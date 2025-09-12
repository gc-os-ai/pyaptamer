__author__ = "satvshr"
__all__ = ["BasePreprocessor"]

import pandas as pd

from pyaptamer.utils.meta_classes import NoNewPublicMethods


class BasePreprocessor(metaclass=NoNewPublicMethods):
    """
    Base class for all preprocessors.

    A preprocessor takes a pandas DataFrame with columns "aptamer", "protein",
    and "y", and converts it into a DataFrame with columns "X" (features)
    and "y" (target).

    The exact conversion logic depends on the algorithm being processed.
    """

    def _ensure_format(self, df, required_cols, only_required_cols=False):
        """
        Ensure input is a DataFrame with the required columns.

        Parameters
        ----------
        df : pandas.DataFrame or dict-like
            Input data.
        required_cols : iterable of str
            Column names that must be present.
        only_required_cols : bool, default=False
            If True, return a DataFrame with only the required columns.

        Returns
        -------
        df : pandas.DataFrame
            Validated DataFrame.

        Raises
        ------
        ValueError
            If required columns are missing.
        """
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)

        if required_cols is not None:
            missing = set(required_cols) - set(df.columns)
            if missing:
                raise ValueError(f"DataFrame is missing columns: {missing}")

        if only_required_cols and required_cols is not None:
            return df[required_cols]

        return df

    def transform(self, df, only_required_cols=True):
        """
        Transform input DataFrame into feature/target representation.

        Parameters
        ----------
        df : pandas.DataFrame
            Must contain "aptamer", "protein", and "y" columns.
        only_required_cols_cols : bool, default=True
            If True, return only the required columns ["X","y"] at the end.

        Returns
        -------
        df : pandas.DataFrame
            Must contain "X" (features) and "y" (target) columns.
        """
        df = self._ensure_format(df, required_cols=["aptamer", "protein", "y"])
        df = self._transform(df)
        df = self._ensure_format(
            df, required_cols=["X", "y"], only_required_cols=only_required_cols
        )

        return df

    def _transform(self, df):
        """
        Apply subclass-specific preprocessing.

        Parameters
        ----------
        df : pandas.DataFrame
            Input DataFrame with "aptamer", "protein", "y".

        Returns
        -------
        df : pandas.DataFrame
            Transformed DataFrame with "X" and "y".
        """
        raise NotImplementedError("Subclasses must implement `_transform`.")
