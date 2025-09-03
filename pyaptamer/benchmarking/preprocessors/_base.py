import pandas as pd

from pyaptamer.utils.meta_classes import NoNewPublicMethods


class BasePreprocessor(metaclass=NoNewPublicMethods):
    """Base class for all preprocessors."""

    def _ensure_format(self, df):
        """Ensure input is in the correct df format."""
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)
        required_columns = {"X", "y"}
        missing = required_columns - set(df.columns)
        if missing:
            raise ValueError(f"Input DataFrame is missing columns: {missing}")
        return df

    def transform(self, df):
        """ """
        df = self._ensure_format(df)
        return df
