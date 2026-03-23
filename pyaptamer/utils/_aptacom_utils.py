import os.path
import tempfile

import numpy as np
import pandas as pd  # noqa: F401

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
    from PyBioMed.PyDNA.PyDNAac import GetDAC, GetDACC, GetDCC, GetTAC, GetTACC, GetTCC
    from PyBioMed.PyDNA.PyDNAnac import GetKmer
    from PyBioMed.PyDNA.PyDNApsenac import (
        GetPseDNC,
        GetPseKNC,
        GetSCPseDNC,
        GetSCPseTNC,
    )

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
    from PyBioMed import Pyprotein

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
    for line in pdb.splitlines():
        residue_name = line[17:20].strip()
        if line.startswith("ATOM") and residue_name in AMINOACIDS:
            new_pdb.append(line)
    return new_pdb


def _save_new_pdb(new_pdb):
    with tempfile.NamedTemporaryFile("w", suffix=".pdb", delete=False) as op:
        for line in new_pdb:
            op.write(f"{line}\n")
    return op.name


def _verify_file(x):
    if os.path.isfile(x):
        return x
    else:
        raise ValueError(f"""File not found on path: {x} """)


def _extract_residue_sasa_values(sasa):
    sasa_per_restype = []
    for residue in sasa:
        if hasattr(residue, "sasa"):
            sasa_per_restype.append(float(residue.sasa))
        else:
            sasa_per_restype.append(float(residue[1]))
    return sasa_per_restype


def _residue_exposure(x):
    import rust_sasa_python as rsp

    with open(x) as op:
        pdb = op.read()

    cleaned_pdb = _clean_pdb(pdb)
    if not cleaned_pdb:
        raise ValueError(f"No protein ATOM records found in PDB file: {x}")

    cleaned_path = _save_new_pdb(cleaned_pdb)
    try:
        sasa = rsp.calculate_residue_sasa(cleaned_path)
        return _extract_residue_sasa_values(sasa)
    finally:
        if os.path.exists(cleaned_path):
            os.remove(cleaned_path)


#################################################################


############ Aptamer SS Extraction ##############################


def _analyse_ss(x):
    ss_features = [
        x.count("."),
        x.count("("),
        x.count(")"),
    ]

    length = len(x)
    chunk_size = length // 4

    for i in range(0, length - chunk_size, chunk_size - 1):
        chunk = x[i : i + chunk_size]
        ss_features.extend(
            [
                chunk.count("."),
                chunk.count("("),
                chunk.count(")"),
            ]
        )

    return np.array(ss_features)


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
    if _validate_sequence(seq, "AGCT"):
        apt_features = _dna_pybiomed(seq)
        return apt_features
    else:
        raise ValueError("""Aptamer Sequence is not DNA or has formatting
        'issues (i.e. "-", "5", ".")""")


def _target_feature_extraction(x):
    seq = _clean_target_sequence(x)
    if _validate_sequence(seq, AA_SINGLE):
        trg_features = _protein_pybiomed(seq)
        return trg_features
    else:
        raise ValueError("""Target Sequence might have missing
        or incorrect resiudes""")


def _secondary_structure_analysis(x):
    if _validate_sequence(x, "(.)"):
        ss_features = _analyse_ss(x)
        return ss_features
    else:
        raise ValueError("""Secondary Structure must be in dot bracket format  
        [ i.e. ....(...).... ] and aptamer length must be >= 8 nucleotides""")


def _protein_sasa_extraction(x):
    pdb_file = _verify_file(x)
    sasa_per_residues = _residue_exposure(pdb_file)
    return np.array(sasa_per_residues)


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
    """Convert a list of tuples into a numeric feature matrix.

    Each element of ``X`` is a tuple whose entries correspond, in order,
    to the keys listed in ``features``.

    Parameters
    ----------
    X : list of tuple
        Each tuple contains string-valued inputs in the order described
        by ``features``.  For example, with
        ``features=["aptamer", "target"]`` every tuple should be
        ``(aptamer_seq, target_seq)``.
    features : list of str
        Ordered feature keys.  Valid keys are
        ``"aptamer"``, ``"target"``, ``"ss"``, ``"pdb_id"``.

    Returns
    -------
    np.ndarray, shape (n_samples, n_features)
        Numeric feature matrix (float32).
    """
    _validate_features(features)

    feats = []
    for row_tuple in X:
        if len(row_tuple) != len(features):
            raise ValueError(
                f"Expected {len(features)} elements per row "
                f"(one per feature key), got {len(row_tuple)}.\n"
                f"features={features}"
            )
        row = []
        for value, key in zip(row_tuple, features, strict=False):
            row.append(_feature_router(value, key))
        feats.append(np.hstack(row))

    return np.vstack(feats).astype(np.float32)
