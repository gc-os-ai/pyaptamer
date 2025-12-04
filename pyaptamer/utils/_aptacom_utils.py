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
import os.path

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

AA_SINGLE = "GAVLITSMCPFYWHKRDENQ"


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


def _validate_sequence(apt_seq, ref):
    check = [i for i in apt_seq.upper() if i not in ref]
    if check != []:
        return False
    else:
        return True


def _clean_aptamer_sequence(sequence):
    ref_table = str.maketrans({"5": "", "3": "", "-": "", ".": "", "U": "T"})
    return str(sequence).translate(ref_table)

def _clean_target_sequence(sequence):
    ref_table = str.maketrans({"-": "A", ".": "A"})
    return str(sequence).translate(ref_table)


def _validate_shape(dna_df, prot_df):
    dna_shape = dna_df.shape
    prot_shape = prot_df.shape
    if dna_shape == prot_shape:
        return True
    else:
        return False


#################################################################


############ Protein SASA Extraction ############################
def _clean_pdb(pdb):
    new_pdb = []
    for i in pdb.split("\n")[:-1]:
        if 'ATOM' in i and len([j for j in i if j in AMINOACIDS]) >= 1:
            new_pdb.append(i)
    return new_pdb

def _save_new_pdb(x,new_pdb):
    new_path = x.replace(".pdb","_clean_.pdb")
    with open(new_path, "w") as op:
        for i in new_pdb:
            op.write(f"{i}\n")
    return new_path


def _verify_file(x):
    if os.path.isfile(x):
        with open(x, "r") as op:
            pdb = op.read()
        return _save_new_pdb(x,_clean_pdb(pdb))
    else: 
        raise ValueError(f'''File not found on path: {x} ''')

def _residue_exposure(x):
    sasa = rsp.calculate_sasa_at_residue_level(x)
    sasa_per_restype = []
    for i in sasa:
        sasa_per_restype.append(float(i[1]))
    return sasa_per_restype


#################################################################


############ Aptamer SS Extraction ##############################


def _analyse_ss(x):  # Still have to mod it for apt seqs of len <= 8
    ss_features = []
    ss_features.append(x.count("."))
    ss_features.append(x.count("("))
    ss_features.append(x.count(")"))
    l = len(x)
    d = len(x) // 4
    for i in range(0, l - d, d - 1): # And add an option to discard this section
        ss_features.append(x[i : i + d].count("."))
        ss_features.append(x[i : i + d].count("("))
        ss_features.append(x[i : i + d].count(")"))

    ss_stack = np.hstack(ss_features)
    return ss_stack


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
    # Add some sort of caveat for apt sequences equal or smaller than 8 units
    seq = _clean_aptamer_sequence(x)
    if _validate_sequence(seq,'AGCT'):
        apt_features = _dna_pybiomed(seq)
        return apt_features
    else: 
        raise ValueError('''Aptamer Sequence is not DNA or has formatting
        'issues (i.e. "-", "5", ".")''')
    


def _target_feature_extraction(x):
    seq = _clean_target_sequence(x)
    if _validate_sequence(seq,AA_SINGLE):
        trg_features = _protein_pybiomed(seq)
        return trg_features
    else: 
        raise ValueError('''Target Sequence might have missing
        or incorrect resiudes''')
    

def _secondary_structure_analysis(x):
    if _validate_sequence(x,"(.)"):
        ss_features = _analyse_ss(x)
        return ss_features
    else:
        raise ValueError('''Secondary Structure must be in dot bracket format  
        [ i.e. ....(...).... ] and aptamer length must be >= 8 nucleotides''')
    

def _protein_sasa_extraction(x):
    pdb_file = _verify_file(x)
    sasa_per_residues = _residue_exposure(pdb_file)


FEATURES = {
    "aptamer": _aptamer_feature_extraction,
    "target": _target_feature_extraction,
    "ss": _secondary_structure_analysis,
    "pdb_id": _protein_sasa_extraction,
}

def _feature_router(x, f):
    # Takes as input a feature and a feature name
    # Based on the pair it routes them to the correct feature function
    return FEATURES[f](x)


def pairs_to_features(X, features):
    # validate if len(X) == len(features)
    # validates if apt and target sequence are present
    # runs feature extraction for each
    row = []
    feats = []
    if _validate_feature_format(X, features):
        for x, f in zip(X, features):
            row.append(_feature_router(x,f))
        feats.append(np.hstack(row))
    else:
        raise ValueError('''Input should be formated as follows:
                         X: ["AGCT", "MLKP","...(.).","/home/Desktop/file.pdb"]
                         features:["aptamer", "target","ss","pdb_id"]
''')
    return feats
