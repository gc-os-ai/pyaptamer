import numpy as np
import pandas as pd
import rust_sasa_python as rsp
from PyBioMed import Pyprotein
from PyBioMed.PyDNA.PyDNAac import *
from PyBioMed.PyDNA.PyDNAac import GetDAC
from PyBioMed.PyDNA.PyDNAnac import *
from PyBioMed.PyDNA.PyDNApsenac import *
from PyBioMed.PyDNA.PyDNApsenac import GetPseDNC
from PyBioMed.PyDNA.PyDNAutil import *

AMINOACIDS = [
    "GLY",
    "ALA",
    "VAL",
    "LEU",
    "ILE",
    "THR",
    "SER",
    "MET",
    "CYS",
    "PRO",
    "PHE",
    "TYR",
    "TRP",
    "HIS",
    "LYS",
    "ARG",
    "ASP",
    "GLU",
    "ASN",
    "GLN",
]
FEATURES = {
    "aptamer": _aptamer_feature_extraction,
    "target": _target_feature_extraction,
    "ss": _secondary_structure_analysis,
    "pdb_id": _protein_sasa_extraction,
}


############# PyBioMed Feature Extraction #######################
def _dna_pybiomed(dna):
    resdna = np.hstack(
        [
            list(GetDAC(dna, all_property=True).values()),
            list(GetDCC(dna, all_property=True).values()),
            list(GetDACC(dna, all_property=True).values()),
            list(GetTAC(dna, all_property=True).values()),
            list(GetTACC(dna, all_property=True).values()),
            list(GetTCC(dna, all_property=True).values()),
            list(GetKmer(dna, k=3).values()),
            list(GetPseDNC(dna, all_property=True).values()),
            list(GetPseKNC(dna, all_property=True).values()),
            list(GetSCPseDNC(dna, all_property=True).values()),
            list(GetSCPseTNC(dna, all_property=True).values()),
        ]
    )
    return resdna


def _protein_pybiomed(prot):
    resprot = list(Pyprotein.PyProtein(prot).GetALL().values())
    return resprot


def _template_feature():
    # The resulting dicts from the GetAll_Pair function doesn't match that of a DFrame;
    # To convert it I'll create the following DICT pair - with reference random DNA/Protein set;
    ex_apt = "AGTCGATGG"
    ex_trg = "MWLGRRALL"
    DNA_Frame = _dna_pybiomed(ex_apt)
    PROT_Frame = _protein_pybiomed(ex_trg)
    dna_f = {}
    prot_f = {}
    for p in DNA_Frame.keys():
        dna_f[f"apt_{p}"] = []
    for p in PROT_Frame.keys():
        prot_f[f"trgt_{p}"] = []
    return dna_f, prot_f


def _validate_sequence(apt_seq, ref):
    check = [i for i in apt_seq.upper() if i not in ref]
    if check != []:
        return False
    else:
        return True


def _clean_sequence(sequence):
    ref_table = str.maketrans({"5": "", "3": "", "-": "", ".": "", "U": "T"})
    return str(sequence).translate(ref_table)


def _build_dna_df(df):
    dna_frame, _ = _template_feature()
    dna_frame["Aptamer Sequence"] = []
    for n in range(len(df)):
        apt = _clean_sequence(df["Aptamer Sequence"].iloc[n])
        if _validate_sequence(apt, ref="AUGCT"):
            pass
        else:
            apt = "AAAAAAAAAAA"
        pdna = _dna_pybiomed(apt)
        dna_frame["Aptamer Sequence"].append(apt)
        for k, v in pdna.items():
            dna_frame[f"apt_{k}"].append(v)
    dna_df = pd.DataFrame.from_dict(dna_frame)
    return dna_df


def _build_protein_df(df):
    _, prot_frame = _template_feature()
    prot_frame["Target Sequence"] = []
    for n in range(len(df)):
        trgt = str(df["Target Sequence"].iloc[n])
        pdb = str(df["PDB_ID"].iloc[n])
        prot_frame["Target Sequence"].append(trgt)
        prot_frame["PDB_ID"].append(pdb)
        pprot = _protein_pybiomed(trgt)
        for k, v in pprot.items():
            prot_frame[f"trgt_{k}"].append(v)
    prot_df = pd.DataFrame.from_dict(prot_frame)
    return prot_df


def _validate_shape(dna_df, prot_df):
    dna_shape = dna_df.shape
    prot_shape = prot_df.shape
    if dna_shape == prot_shape:
        return True
    else:
        return False


def _concat_df(dna_df, prot_df):
    if _validate_shape(dna_df, prot_df):
        dna_df.reset_index(inplace=True, drop=True)
        prot_df.reset_index(inplace=True, drop=True)
        df = pd.concat([dna_df, prot_df], axis=1)
    else:
        raise ValueError(
            "Aptamer and Target feature dataframe expected to be of equal shape"
        )
        quit()  # Temporary


#################################################################


############ Protein SASA Extraction ############################
def _residue_exposure_map(pdb_file_path):
    sasa = rsp.calculate_sasa_at_residue_level(pdb_file_path)
    sasa_per_restype = {f"sasa_{i}": 0 for i in AMINOACIDS}
    for i in sasa:
        restype = str(f"sasa_{i[0].split('_')[1]}")
        sasa_per_restype[restype] += float(i[1])
    return sasa_per_restype


def _build_df(dataframe: pd.DataFrame):
    sasa_df = {f"sasa_{i}": [] for i in AMINOACIDS}
    cdir = os.path.dirname(os.path.abspath(__file__))
    for i in list(dataframe["PDB_ID"]):
        file_path = os.path.join(cdir, "../datasets/data/", i)
        sasa_per_restype = _residue_exposure_map(
            file_path
        )  # Here pdb sub-dir path can be modified
        for j in list(sasa_per_restype.keys()):
            sasa_df[j].append(sasa_per_restype[j])
    return pd.DataFrame().from_dict(sasa_df)


#################################################################


############ Aptamer SS Extraction ##############################


def _analyse_ss(df):  # Still have to mod it for apt seqs of len <= 8
    ss_table = {"Aptamer Sequence": df["Aptamer Sequence"], "SS": df["SS"]}
    for n in range(4):
        ss_table[f"S{n}_."] = []
        ss_table[f"S{n}_("] = []
        ss_table[f"S{n}_)"] = []
    for ss in ss_table["SS"]:
        ss_table["S0_."].append(ss.count("."))
        ss_table["S0_("].append(ss.count("("))
        ss_table["S0_)"].append(ss.count(")"))
        l = len(ss)
        d = len(ss) // 4
        for s, i in zip(["S1", "S2", "S3", "S4"], range(0, l - d, d - 1), strict=False):
            ss_table[f"{s}_."].append(ss[i : i + d].count("."))
            ss_table[f"{s}_("].append(ss[i : i + d].count("("))
            ss_table[f"{s}_)"].append(ss[i : i + d].count(")"))

    ndf = pd.DataFrame().from_dict(ss_table)
    return ndf


#################################################################
def _validate_feature_format(X, features):
    return len(X) == len(features)


def _validate_features(features):
    ref = ["aptamer", "target", "ss", "pdb_id"]
    for r, f in zip(ref, features, strict=False):
        if len(features) < 2 or r != f:
            raise ValueError(
                "Expected feature lists:\n"
                "['aptamer', 'target']\n"
                "['aptamer', 'target', 'ss']\n"
                "['aptamer', 'target', 'ss', 'pdb_id']"
            )
    return True


def _aptamer_feature_extraction(x):
    pass


def _target_feature_extraction(x):
    pass


def _secondary_structure_analysis(x):
    pass


def _protein_sasa_extraction(x):
    pass


def _feature_router(x, f):
    # Takes as input a feature and a feature name
    # Based on the pair it routes them to the correct feature function
    return FEATURES[f](x)


def pairs_to_features(X, features):
    # validate if len(X) == len(features)
    # validates if apt and target sequence are present
    # runs feature extraction for each
    features = []
