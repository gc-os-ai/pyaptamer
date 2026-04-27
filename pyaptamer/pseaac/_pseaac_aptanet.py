__author__ = ["nennomp", "satvshr"]
__all__ = ["AptaNetPSeAAC"]

from pyaptamer.pseaac._pseaac_general import PSeAAC

class AptaNetPSeAAC(PSeAAC):
    """
    Compute Pseudo Amino Acid Composition (PseAAC) features for a protein sequence,
    using the fixed 21-property, 7-group configuration as used in AptaNet.

    This is a thin subclass of `PSeAAC` with fixed physicochemical property
    configuration matching the original AptaNet implementation. All 21 properties
    are used, grouped into 7 groups of 3.

    For full documentation see `PSeAAC`.

    Parameters
    ----------
    lambda_val : int, optional, default=30
        The lambda parameter defining the number of sequence-order correlation factors.
        This also determines the minimum length allowed for input protein sequences,
        which should be of length greater than `lambda_val`.
    weight : float, optional, default=0.05
        The weight factor for the sequence-order correlation features.

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