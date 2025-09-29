__author__ = "rpgv"
__all__ = ["load_aptacom"]

from datasets import load_dataset
import pandas as pd



def load_aptacom(as_df=False):
    """
    Loads aptacom dataset from hugging face datasets.

    Parameters
    ----------
    as_df: (bool) Requires pandas compatible format; converts
    dataset into pandas dataframe

    
    Returns
    -------
    dataset: Hugging face dataset in a pandas compatible format.
    """
    aptacom = load_dataset('rpgv/AptaCom')['train']
    dataset = aptacom.with_format('pandas')
    if as_df: dataset = pd.DataFrame().from_dict(aptacom);
    

    return dataset
