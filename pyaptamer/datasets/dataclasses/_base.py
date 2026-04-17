"""Base class for aptamer-data containers.

`BaseAptamerDataset` is a minimal abstract root for every in-memory data
container in pyaptamer (paired aptamer-protein, single-sequence, masked-LM,
etc.). It owns only tag plumbing and the `load()` contract. Concrete
subclasses set their own `scitype` and `X_inner_mtype` tags and implement
`load()` and any input-coercion machinery appropriate for their scitype.
"""

__author__ = ["siddharth7113"]
__all__ = ["BaseAptamerDataset"]


from skbase.base import BaseObject


class BaseAptamerDataset(BaseObject):
    """Abstract root for pyaptamer in-memory data containers.

    Subclasses MUST:
        - Override `_tags["scitype"]` with their scientific-type identifier
          (e.g. ``"APIPairs"``, ``"MaskedSequences"``).
        - Override `_tags["X_inner_mtype"]` with the canonical inner mtype(s)
          their `load()` returns.
        - Implement `load()` to return data in their canonical inner mtype.

    The base provides no input-coercion logic — what counts as a valid input
    depends entirely on the scitype, so coercion lives in the concrete subclass.
    """

    _tags = {
        "object_type": "dataset",
        "authors": [],
        "maintainers": [],
        "python_dependencies": None,
        "scitype": None,
        "X_inner_mtype": None,
        "has_y": True,
    }

    def __init__(self):
        super().__init__()

    def load(self):
        """Return X in this dataset's canonical inner mtype.

        Subclasses must override.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement load().")
