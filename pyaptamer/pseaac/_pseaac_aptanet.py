__author__ = ["nennomp", "satvshr"]
__all__ = ["AptaNetPSeAAC"]

from pyaptamer.pseaac._pseaac_general import PSeAAC

class AptaNetPSeAAC(PSeAAC):
    """
    Compute Pseudo Amino Acid Composition (PseAAC) features for a protein sequence.

    This class generates a numerical feature vector that encodes both the composition
    and local order of amino acids in a protein sequence. The features are derived from
    selected physicochemical properties and sequence-order correlations as described in
    the PseAAC model by Chou.

    The PSeAAC algorithm uses 21 normalized physiochemical (NP) properties of amino
    acids, which we load from a predefined matrix using `aa_props`. These 21 properties
    are grouped into 7 distinct property groups, with each group containing
    3 consecutive properties. Specifically, the groups are arranged in order as follows:
    Group 1 includes properties 1-3, Group 2 includes properties 4-6, and so on, up to
    Group 7, which includes properties 19-21. The properties in order are:

    1. Hydrophobicity
    2. Hydrophilicity
    3. Side-chain Mass
    4. Polarity
    5. Molecular Weight
    6. Melting Point
    7. Transfer Free Energy
    8. Buriability
    9. Bulkiness
    10. Solvation Free Energy
    11. Relative Mutability
    12. Residue Volume
    13. Volume
    14. Amino Acid Distribution
    15. Hydration Number
    16. Isoelectric Point
    17. Compressibility
    18. Chromatographic Index
    19. Unfolding Entropy Change
    20. Unfolding Enthalpy Change
    21. Unfolding Gibbs Free Energy Change

    Each feature vector consists of:

    - 20 normalized amino acid composition features (frequency of each standard
      amino acid)
    - `self.lambda_val` sequence-order correlation features based on physicochemical
      similarity between residues.

    These (20 + `self.lambda_val`) features are computed for each of 7 predefined
    property groups, resulting in a final vector of length (20 + `self.lambda_val`) * 7.

    This is a fixed, stable subclass of `PSeAAC` fine-tuned for AptaNet, using exactly
    21 properties in 7 groups of 3. The fixed configuration ensures that bug fixes in
    `PSeAAC` automatically propagate here without manual mirroring. For a fully
    configurable version see `PSeAAC`.

    Parameters
    ----------
    lambda_val : int, optional, default=30
        The lambda parameter defining the number of sequence-order correlation factors.
        This also determines the minimum length allowed for input protein sequences,
        which should be of length greater than `lambda_val`.
    weight : float, optional, default=0.05
        The weight factor for the sequence-order correlation features.

    Attributes
    ----------
    np_matrix : np.ndarray
        A 20x21 matrix of normalized physicochemical properties for the 20 standard
        amino acids.
    prop_groups : list of list
        List of 7 lists, each containing indices of 3 properties that form a property
        group.

    Methods
    -------
    transform(protein_sequence)
        Generate the PseAAC feature vector for the given protein sequence.
        Inherited from `PSeAAC`.

    References
    ----------
    Shen HB, Chou KC. PseAAC: a flexible web server for generating various kinds of
    protein pseudo amino acid composition. Anal Biochem. 2008 Feb 15;373(2):386-8.
    doi: 10.1016/j.ab.2007.10.012. Epub 2007 Oct 13. PMID: 17976365.

    Example
    -------
    >>> from pyaptamer.pseaac import AptaNetPSeAAC
    >>> pseaac = AptaNetPSeAAC()
    >>> features = pseaac.transform("ACDEFGHIKLMNPQRHIKLMNPQRSTVWHIKLMNPQRSTVWY")
    >>> print(features[:10])
    [0.006 0.006 0.006 0.006 0.006 0.006 0.018 0.018 0.018 0.018]
    """

    def __init__(self, lambda_val=30, weight=0.05):
        super().__init__(
            lambda_val = lambda_val,
            weight = weight,
            prop_indices=None,
            group_props=3,
            custom_groups=None,
        )     