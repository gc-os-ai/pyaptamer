__author__ = "Jayant-kernel"
__all__ = [
    "aptacom_dna_features",
    "aptacom_protein_features",
    "aptacom_pairs_to_features",
]

from itertools import product

import numpy as np
import pandas as pd

_DINUC_ORDER = [
    "AA",
    "AC",
    "AG",
    "AT",
    "CA",
    "CC",
    "CG",
    "CT",
    "GA",
    "GC",
    "GG",
    "GT",
    "TA",
    "TC",
    "TG",
    "TT",
]

_DINUC_PROPS = np.array(
    [
        [3.332, -0.80, 0.00, -0.06, -2.00, 35.20],
        [3.305, 6.70, 0.00, 1.54, -2.80, 27.70],
        [3.381, 2.80, 0.00, 1.29, 5.30, 29.80],
        [3.213, 0.80, 0.00, 1.01, 0.00, 34.90],
        [3.374, 1.50, 0.00, -1.06, -2.00, 33.50],
        [3.322, 0.00, 0.00, -1.57, -2.80, 33.67],
        [3.455, 6.00, 0.00, -2.40, 5.30, 29.80],
        [3.381, 2.80, 0.00, 1.29, -5.30, 29.80],
        [3.374, 1.50, 0.00, -1.06, 2.00, 33.50],
        [3.456, 0.00, 0.00, 0.44, 0.00, 36.90],
        [3.322, 0.00, 0.00, -1.57, 2.80, 33.67],
        [3.305, 6.70, 0.00, 1.54, 2.80, 27.70],
        [3.206, 0.00, 0.00, 0.72, 0.00, 38.50],
        [3.213, 0.80, 0.00, 1.01, 0.00, 34.90],
        [3.456, 0.00, 0.00, 0.44, 0.00, 36.90],
        [3.332, -0.80, 0.00, -0.06, 2.00, 35.20],
    ],
    dtype=np.float64,
)

_DINUC_PROPS_NORM = (_DINUC_PROPS - _DINUC_PROPS.mean(axis=0)) / (
    _DINUC_PROPS.std(axis=0) + 1e-9
)
_DINUC_IDX = {d: i for i, d in enumerate(_DINUC_ORDER)}
_TRINUC_ORDER = ["".join(t) for t in product("ACGT", repeat=3)]
_TRINUC_IDX = {t: i for i, t in enumerate(_TRINUC_ORDER)}
DNA_BASES = list("ACGT")

_AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")
_AA_HYDROPHOBICITY = {
    "A": 1.8,
    "C": 2.5,
    "D": -3.5,
    "E": -3.5,
    "F": 2.8,
    "G": -0.4,
    "H": -3.2,
    "I": 4.5,
    "K": -3.9,
    "L": 3.8,
    "M": 1.9,
    "N": -3.5,
    "P": -1.6,
    "Q": -3.5,
    "R": -4.5,
    "S": -0.8,
    "T": -0.7,
    "V": 4.2,
    "W": -0.9,
    "Y": -1.3,
}
_AA_HYDROPHILICITY = {
    "A": -0.5,
    "C": -1.0,
    "D": 3.0,
    "E": 3.0,
    "F": -2.5,
    "G": 0.0,
    "H": -0.5,
    "I": -1.8,
    "K": 3.0,
    "L": -1.8,
    "M": -1.3,
    "N": 0.2,
    "P": 0.0,
    "Q": 0.2,
    "R": 3.0,
    "S": 0.3,
    "T": -0.4,
    "V": -1.5,
    "W": -3.4,
    "Y": -2.3,
}
_AA_RESIDUEMASS = {
    "A": 71.03711,
    "C": 103.00919,
    "D": 115.02694,
    "E": 129.04259,
    "F": 147.06841,
    "G": 57.02146,
    "H": 137.05891,
    "I": 113.08406,
    "K": 128.09496,
    "L": 113.08406,
    "M": 131.04049,
    "N": 114.04293,
    "P": 97.05276,
    "Q": 128.05858,
    "R": 156.10111,
    "S": 87.03203,
    "T": 101.04768,
    "V": 99.06841,
    "W": 186.07931,
    "Y": 163.06333,
}

_CTD_CLASSES = [
    (
        "Hydro",
        {
            "R": 1,
            "K": 1,
            "E": 1,
            "D": 1,
            "Q": 1,
            "N": 1,
            "G": 2,
            "A": 2,
            "S": 2,
            "T": 2,
            "P": 2,
            "H": 2,
            "Y": 2,
            "V": 3,
            "L": 3,
            "I": 3,
            "F": 3,
            "W": 3,
            "M": 3,
            "C": 3,
        },
    ),
    (
        "VDW",
        {
            "G": 1,
            "A": 1,
            "S": 1,
            "T": 1,
            "C": 1,
            "P": 1,
            "V": 1,
            "N": 2,
            "D": 2,
            "E": 2,
            "Q": 2,
            "I": 2,
            "L": 2,
            "M": 3,
            "H": 3,
            "K": 3,
            "F": 3,
            "R": 3,
            "Y": 3,
            "W": 3,
        },
    ),
    (
        "Polar",
        {
            "L": 1,
            "I": 1,
            "F": 1,
            "W": 1,
            "C": 1,
            "M": 1,
            "V": 1,
            "Y": 1,
            "P": 2,
            "A": 2,
            "T": 2,
            "G": 2,
            "S": 2,
            "H": 3,
            "Q": 3,
            "R": 3,
            "K": 3,
            "N": 3,
            "E": 3,
            "D": 3,
        },
    ),
    (
        "Polariz",
        {
            "G": 1,
            "A": 1,
            "S": 1,
            "D": 1,
            "T": 1,
            "C": 2,
            "P": 2,
            "N": 2,
            "V": 2,
            "E": 2,
            "Q": 2,
            "I": 2,
            "L": 2,
            "K": 3,
            "M": 3,
            "H": 3,
            "F": 3,
            "R": 3,
            "Y": 3,
            "W": 3,
        },
    ),
    (
        "Charge",
        {
            "K": 1,
            "R": 1,
            "A": 2,
            "N": 2,
            "C": 2,
            "Q": 2,
            "G": 2,
            "H": 2,
            "I": 2,
            "L": 2,
            "M": 2,
            "F": 2,
            "P": 2,
            "S": 2,
            "T": 2,
            "W": 2,
            "Y": 2,
            "V": 2,
            "D": 3,
            "E": 3,
        },
    ),
    (
        "SS",
        {
            "E": 1,
            "A": 1,
            "L": 1,
            "M": 1,
            "Q": 1,
            "K": 1,
            "R": 1,
            "H": 1,
            "V": 2,
            "I": 2,
            "Y": 2,
            "C": 2,
            "W": 2,
            "F": 2,
            "T": 2,
            "G": 3,
            "N": 3,
            "P": 3,
            "S": 3,
            "D": 3,
        },
    ),
    (
        "SA",
        {
            "A": 1,
            "L": 1,
            "F": 1,
            "C": 1,
            "G": 1,
            "I": 1,
            "V": 1,
            "W": 1,
            "R": 2,
            "K": 2,
            "Q": 2,
            "E": 2,
            "N": 2,
            "D": 2,
            "M": 3,
            "S": 3,
            "P": 3,
            "T": 3,
            "H": 3,
            "Y": 3,
        },
    ),
]


def _kmer_freq(seq, k):
    kmers = ["".join(p) for p in product(DNA_BASES, repeat=k)]
    counts = dict.fromkeys(kmers, 0)
    for i in range(len(seq) - k + 1):
        sub = seq[i : i + k]
        if sub in counts:
            counts[sub] += 1
    total = sum(counts.values()) or 1
    return {kmer: counts[kmer] / total for kmer in kmers}


def _dinuc_composition(seq):
    return _kmer_freq(seq, 2)


def _trinuc_composition(seq):
    return _kmer_freq(seq, 3)


def _dac(seq, lag):
    n_props = _DINUC_PROPS_NORM.shape[1]
    result = {}
    for j in range(n_props):
        mu_j = np.mean(_DINUC_PROPS_NORM[:, j])
        for lag_i in range(1, lag + 1):
            pairs = [
                (seq[i : i + 2], seq[i + lag_i : i + lag_i + 2])
                for i in range(len(seq) - lag_i - 1)
                if seq[i : i + 2] in _DINUC_IDX
                and seq[i + lag_i : i + lag_i + 2] in _DINUC_IDX
            ]
            if not pairs:
                result[f"DAC.P{j}.L{lag_i}"] = 0.0
                continue
            val = np.mean(
                [
                    (_DINUC_PROPS_NORM[_DINUC_IDX[d1], j] - mu_j)
                    * (_DINUC_PROPS_NORM[_DINUC_IDX[d2], j] - mu_j)
                    for d1, d2 in pairs
                ]
            )
            result[f"DAC.P{j}.L{lag_i}"] = float(val)
    return result


def _dcc(seq, lag):
    n_props = _DINUC_PROPS_NORM.shape[1]
    result = {}
    for j1 in range(n_props):
        for j2 in range(n_props):
            if j1 == j2:
                continue
            mu_j1 = np.mean(_DINUC_PROPS_NORM[:, j1])
            mu_j2 = np.mean(_DINUC_PROPS_NORM[:, j2])
            for lag_i in range(1, lag + 1):
                pairs = [
                    (seq[i : i + 2], seq[i + lag_i : i + lag_i + 2])
                    for i in range(len(seq) - lag_i - 1)
                    if seq[i : i + 2] in _DINUC_IDX
                    and seq[i + lag_i : i + lag_i + 2] in _DINUC_IDX
                ]
                if not pairs:
                    result[f"DCC.P{j1}{j2}.L{lag_i}"] = 0.0
                    continue
                val = np.mean(
                    [
                        (_DINUC_PROPS_NORM[_DINUC_IDX[d1], j1] - mu_j1)
                        * (_DINUC_PROPS_NORM[_DINUC_IDX[d2], j2] - mu_j2)
                        for d1, d2 in pairs
                    ]
                )
                result[f"DCC.P{j1}{j2}.L{lag_i}"] = float(val)
    return result


def _dacc(seq, lag):
    feat = {}
    feat.update(_dac(seq, lag))
    feat.update(_dcc(seq, lag))
    return feat


def _trinuc_prop(t, j):
    d1, d2 = t[:2], t[1:]
    v1 = _DINUC_PROPS_NORM[_DINUC_IDX[d1], j] if d1 in _DINUC_IDX else 0.0
    v2 = _DINUC_PROPS_NORM[_DINUC_IDX[d2], j] if d2 in _DINUC_IDX else 0.0
    return (v1 + v2) / 2.0


def _tac(seq, lag):
    n_props = _DINUC_PROPS_NORM.shape[1]
    result = {}
    for j in range(n_props):
        mu_j = np.mean([_trinuc_prop(t, j) for t in _TRINUC_ORDER])
        for lag_i in range(1, lag + 1):
            pairs = [
                (seq[i : i + 3], seq[i + lag_i : i + lag_i + 3])
                for i in range(len(seq) - lag_i - 2)
                if seq[i : i + 3] in _TRINUC_IDX
                and seq[i + lag_i : i + lag_i + 3] in _TRINUC_IDX
            ]
            if not pairs:
                result[f"TAC.P{j}.L{lag_i}"] = 0.0
                continue
            val = np.mean(
                [
                    (_trinuc_prop(t1, j) - mu_j) * (_trinuc_prop(t2, j) - mu_j)
                    for t1, t2 in pairs
                ]
            )
            result[f"TAC.P{j}.L{lag_i}"] = float(val)
    return result


def _tcc(seq, lag):
    n_props = _DINUC_PROPS_NORM.shape[1]
    result = {}
    for j1 in range(n_props):
        for j2 in range(n_props):
            if j1 == j2:
                continue
            mu_j1 = np.mean([_trinuc_prop(t, j1) for t in _TRINUC_ORDER])
            mu_j2 = np.mean([_trinuc_prop(t, j2) for t in _TRINUC_ORDER])
            for lag_i in range(1, lag + 1):
                pairs = [
                    (seq[i : i + 3], seq[i + lag_i : i + lag_i + 3])
                    for i in range(len(seq) - lag_i - 2)
                    if seq[i : i + 3] in _TRINUC_IDX
                    and seq[i + lag_i : i + lag_i + 3] in _TRINUC_IDX
                ]
                if not pairs:
                    result[f"TCC.P{j1}{j2}.L{lag_i}"] = 0.0
                    continue
                val = np.mean(
                    [
                        (_trinuc_prop(t1, j1) - mu_j1) * (_trinuc_prop(t2, j2) - mu_j2)
                        for t1, t2 in pairs
                    ]
                )
                result[f"TCC.P{j1}{j2}.L{lag_i}"] = float(val)
    return result


def _tacc(seq, lag):
    feat = {}
    feat.update(_tac(seq, lag))
    feat.update(_tcc(seq, lag))
    return feat


def _kmer_features(seq, k):
    feat = {}
    for i in range(1, k + 1):
        feat.update(_kmer_freq(seq, i))
    return feat


def _psednc(seq, lag):
    dc = _dinuc_composition(seq)
    n = len(seq)
    n_props = _DINUC_PROPS_NORM.shape[1]
    theta = []
    for lag_i in range(1, lag + 1):
        for j in range(n_props):
            denom = n - lag_i - 1
            if denom <= 0:
                theta.append(0.0)
                continue
            val = (
                sum(
                    _DINUC_PROPS_NORM[_DINUC_IDX[seq[i : i + 2]], j]
                    * _DINUC_PROPS_NORM[_DINUC_IDX[seq[i + lag_i : i + lag_i + 2]], j]
                    for i in range(denom)
                    if seq[i : i + 2] in _DINUC_IDX
                    and seq[i + lag_i : i + lag_i + 2] in _DINUC_IDX
                )
                / denom
            )
            theta.append(val)
    w = 0.05
    total = sum(dc.values()) + w * sum(theta) or 1.0
    result = {f"PseDNC.{d}": dc[d] / total for d in _DINUC_ORDER}
    for i, t in enumerate(theta):
        result[f"PseDNC.theta{i}"] = w * t / total
    return result


def _pseknc(seq, lag, k):
    kf = _kmer_freq(seq, k)
    n = len(seq)
    n_props = _DINUC_PROPS_NORM.shape[1]
    theta = []
    for lag_i in range(1, lag + 1):
        for j in range(n_props):
            denom = n - lag_i - 1
            if denom <= 0:
                theta.append(0.0)
                continue
            val = (
                sum(
                    _DINUC_PROPS_NORM[_DINUC_IDX[seq[i : i + 2]], j]
                    * _DINUC_PROPS_NORM[_DINUC_IDX[seq[i + lag_i : i + lag_i + 2]], j]
                    for i in range(denom)
                    if seq[i : i + 2] in _DINUC_IDX
                    and seq[i + lag_i : i + lag_i + 2] in _DINUC_IDX
                )
                / denom
            )
            theta.append(val)
    w = 0.05
    total = sum(kf.values()) + w * sum(theta) or 1.0
    result = {f"PseKNC.{km}": kf[km] / total for km in kf}
    for i, t in enumerate(theta):
        result[f"PseKNC.theta{i}"] = w * t / total
    return result


def _scpsednc(seq, lag):
    dc = _dinuc_composition(seq)
    n = len(seq)
    n_props = _DINUC_PROPS_NORM.shape[1]
    sc = []
    for lag_i in range(1, lag + 1):
        for j in range(n_props):
            denom = n - lag_i - 1
            if denom <= 0:
                sc.append(0.0)
                continue
            val = (
                sum(
                    _DINUC_PROPS_NORM[_DINUC_IDX[seq[i : i + 2]], j]
                    * _DINUC_PROPS_NORM[_DINUC_IDX[seq[i + lag_i : i + lag_i + 2]], j]
                    for i in range(denom)
                    if seq[i : i + 2] in _DINUC_IDX
                    and seq[i + lag_i : i + lag_i + 2] in _DINUC_IDX
                )
                / denom
            )
            sc.append(val)
    result = {f"SCPseDNC.{d}": dc[d] for d in _DINUC_ORDER}
    for i, s in enumerate(sc):
        result[f"SCPseDNC.sc{i}"] = s
    return result


def _scpseknc(seq, lag):
    tc = _trinuc_composition(seq)
    n = len(seq)
    n_props = _DINUC_PROPS_NORM.shape[1]
    sc = []
    for lag_i in range(1, lag + 1):
        for j in range(n_props):
            denom = n - lag_i - 2
            if denom <= 0:
                sc.append(0.0)
                continue
            val = (
                sum(
                    _DINUC_PROPS_NORM[_DINUC_IDX[seq[i : i + 2]], j]
                    * _DINUC_PROPS_NORM[_DINUC_IDX[seq[i + lag_i : i + lag_i + 2]], j]
                    for i in range(denom)
                    if seq[i : i + 2] in _DINUC_IDX
                    and seq[i + lag_i : i + lag_i + 2] in _DINUC_IDX
                )
                / denom
            )
            sc.append(val)
    result = {f"SCPseTNC.{t}": tc[t] for t in _TRINUC_ORDER}
    for i, s in enumerate(sc):
        result[f"SCPseTNC.sc{i}"] = s
    return result


def aptacom_dna_features(seq, lag=1, k=4):
    """
    Compute AptaCom DNA feature vector for a single sequence.

    Concatenates all 11 AptaCom DNA descriptor groups from the original
    paper: DAC, DCC, DACC, TAC, TCC, TACC, Kmer, PseDNC, PseKNC,
    SCPseDNC, SCPseTNC.

    Parameters
    ----------
    seq : str
        DNA sequence. RNA (U) is converted to DNA (T) automatically.
    lag : int, optional, default=1
        Lag parameter for auto/cross-covariance and pseudo-composition.
    k : int, optional, default=4
        Maximum k-mer size for Kmer and PseKNC feature groups.

    Returns
    -------
    np.ndarray of shape (n_features,), dtype float32
        Concatenated DNA feature vector.

    References
    ----------
    .. [1] Emami, N., et al. AptaCom. Briefings in Bioinformatics, 2022.
       https://doi.org/10.1093/bib/bbac415
    """
    seq = seq.upper().replace("U", "T")
    feat = {}
    feat.update(_dac(seq, lag))
    feat.update(_dcc(seq, lag))
    feat.update(_dacc(seq, lag))
    feat.update(_tac(seq, lag))
    feat.update(_tcc(seq, lag))
    feat.update(_tacc(seq, lag))
    feat.update(_kmer_features(seq, k))
    feat.update(_psednc(seq, lag))
    feat.update(_pseknc(seq, lag, k))
    feat.update(_scpsednc(seq, lag))
    feat.update(_scpseknc(seq, lag))
    return np.array(list(feat.values()), dtype=np.float32)


# ================================================================== #
#  Protein feature helpers                                            #
# ================================================================== #


def _aa_composition(seq):
    n = len(seq) or 1
    return {f"AAC.{aa}": seq.count(aa) / n for aa in _AA_ORDER}


def _dipeptide_composition(seq):
    n = max(len(seq) - 1, 1)
    dipeptides = ["".join(p) for p in product(_AA_ORDER, repeat=2)]
    counts = dict.fromkeys(dipeptides, 0)
    for i in range(len(seq) - 1):
        dp = seq[i : i + 2]
        if dp in counts:
            counts[dp] += 1
    return {f"DPC.{dp}": counts[dp] / n for dp in dipeptides}


def _ctd_composition(seq, cls_map):
    n = len(seq) or 1
    c = {1: 0, 2: 0, 3: 0}
    for aa in seq:
        c[cls_map.get(aa, 2)] += 1
    return [c[1] / n, c[2] / n, c[3] / n]


def _ctd_transition(seq, cls_map):
    n = max(len(seq) - 1, 1)
    t12, t13, t23 = 0, 0, 0
    for i in range(len(seq) - 1):
        pair = tuple(sorted([cls_map.get(seq[i], 2), cls_map.get(seq[i + 1], 2)]))
        if pair == (1, 2):
            t12 += 1
        elif pair == (1, 3):
            t13 += 1
        elif pair == (2, 3):
            t23 += 1
    return [t12 / n, t13 / n, t23 / n]


def _ctd_distribution(seq, cls_map):
    n = len(seq) or 1
    result = []
    for g in [1, 2, 3]:
        positions = [i for i, aa in enumerate(seq) if cls_map.get(aa, 2) == g]
        if not positions:
            result.extend([0.0] * 5)
        else:
            count = len(positions)
            qi = [0, count // 4, count // 2, 3 * count // 4, count - 1]
            result.extend([positions[q] / n for q in qi])
    return result


def _ctd_features(seq):
    """CTD descriptors -- 7 properties x 21 values = 147 features."""
    result = {}
    for name, cls_map in _CTD_CLASSES:
        for i, v in enumerate(_ctd_composition(seq, cls_map)):
            result[f"CTD.{name}.C{i + 1}"] = v
        for i, v in enumerate(_ctd_transition(seq, cls_map)):
            result[f"CTD.{name}.T{i + 1}"] = v
        for i, v in enumerate(_ctd_distribution(seq, cls_map)):
            result[f"CTD.{name}.D{i + 1}"] = v
    return result


def _norm_aa_prop(prop_dict):
    vals = [prop_dict.get(aa, 0.0) for aa in _AA_ORDER]
    mu, sigma = np.mean(vals), np.std(vals) or 1.0
    return {aa: (prop_dict.get(aa, 0.0) - mu) / sigma for aa in _AA_ORDER}


def _paac(seq, lag=6, w=0.05):
    """Pseudo amino acid composition -- (20 + lag) features."""
    aac = _aa_composition(seq)
    n = len(seq)
    norm_props = [
        _norm_aa_prop(_AA_HYDROPHOBICITY),
        _norm_aa_prop(_AA_HYDROPHILICITY),
        _norm_aa_prop(_AA_RESIDUEMASS),
    ]
    theta = []
    for lag_i in range(1, lag + 1):
        val, cnt = 0.0, 0
        for norm_p in norm_props:
            pairs = [
                (norm_p.get(seq[i], 0.0), norm_p.get(seq[i + lag_i], 0.0))
                for i in range(n - lag_i)
                if seq[i] in _AA_ORDER and seq[i + lag_i] in _AA_ORDER
            ]
            if pairs:
                val += np.mean([a * b for a, b in pairs])
                cnt += 1
        theta.append(val / cnt if cnt else 0.0)
    denom = sum(aac.values()) + w * sum(theta) or 1.0
    result = {f"PAAC.{aa}": aac[f"AAC.{aa}"] / denom for aa in _AA_ORDER}
    for i, t in enumerate(theta):
        result[f"PAAC.theta{i}"] = w * t / denom
    return result


def aptacom_protein_features(seq):
    """
    Compute AptaCom protein feature vector for a single protein sequence.

    Implements the main descriptor groups from PyBioMed GetALL():
    AAC (20), DPC (400), CTD (147), PAAC (26).

    Parameters
    ----------
    seq : str
        Protein amino acid sequence (single-letter codes).

    Returns
    -------
    np.ndarray of shape (593,), dtype float32
        Concatenated protein feature vector.

    References
    ----------
    .. [1] Emami, N., et al. AptaCom. Briefings in Bioinformatics, 2022.
       https://doi.org/10.1093/bib/bbac415
    """
    seq = seq.upper()
    feat = {}
    feat.update(_aa_composition(seq))
    feat.update(_dipeptide_composition(seq))
    feat.update(_ctd_features(seq))
    feat.update(_paac(seq))
    return np.array(list(feat.values()), dtype=np.float32)


def aptacom_pairs_to_features(X, lag=1, k=4):
    """
    Convert (aptamer, protein) pairs into the AptaCom feature matrix.

    Concatenates DNA and protein feature vectors for each pair, matching
    the original AptaCom implementation structure.

    Parameters
    ----------
    X : list of tuple of (str, str) or pandas.DataFrame
        Either a list of (aptamer_sequence, protein_sequence) tuples,
        or a DataFrame with columns "aptamer" and "protein".
    lag : int, optional, default=1
        Lag parameter forwarded to aptacom_dna_features.
    k : int, optional, default=4
        Maximum k-mer size forwarded to aptacom_dna_features.

    Returns
    -------
    np.ndarray of shape (n_samples, n_features), dtype float32
        Feature matrix suitable for XGBoost or any sklearn estimator.
    """
    feats = []
    if isinstance(X, pd.DataFrame):
        pairs = zip(X["aptamer"], X["protein"], strict=False)
    else:
        pairs = X
    for aptamer_seq, protein_seq in pairs:
        dna_vec = aptacom_dna_features(aptamer_seq, lag=lag, k=k)
        prot_vec = aptacom_protein_features(protein_seq)
        feats.append(np.concatenate([dna_vec, prot_vec]))
    return np.vstack(feats).astype(np.float32)
