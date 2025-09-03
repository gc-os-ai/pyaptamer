import pandas as pd

from pyaptamer.utils.meta_classes import NoNewPublicMethods


class BasePreprocessor(metaclass=NoNewPublicMethods):
    """Base class for all preprocessors."""

    def _ensure_format(self, df, required_cols=None):
        """Ensure input is in the correct df format."""
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame is missing columns: {missing}")

    def transform(self, df):
        """ """
        df = self._ensure_format(df, required_cols=["aptamer", "protein", "y"])

        df = self._transform(df)

        df = self._ensure_format(df, required_cols=["X", "y"])

        return df

    def _transform(self, df):
        """Subclasses must override this with preprocessing logic."""
        raise NotImplementedError("Subclasses must implement `_transform`.")
