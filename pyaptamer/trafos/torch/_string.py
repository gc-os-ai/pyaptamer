"""String-to-string transformations."""

from pyaptamer.trafos.torch._base import BaseTorchTransform


class Reverse(BaseTorchTransform):
    """Reverse a sequence string.
    
    Usage
    -----
    >>> transform = Reverse()
    >>> transform("ATCG")
    'GCTA'
    """

    def __call__(self, x: str) -> str:
        return x[::-1]


class DNAtoRNA(BaseTorchTransform):
    """Convert DNA to RNA (T -> U).
    
    Usage
    -----
    >>> transform = DNAtoRNA()
    >>> transform("ATCGATCG")
    'AUCGAUCG'
    """

    def __call__(self, x: str) -> str:
        return x.replace("T", "U")
