"""String-to-string transformations."""


class Reverse:
    """Reverse a sequence string."""

    def __call__(self, x: str) -> str:
        return x[::-1]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class DNAtoRNA:
    """Convert DNA to RNA (T -> U)."""

    def __call__(self, x: str) -> str:
        return x.replace("T", "U")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
