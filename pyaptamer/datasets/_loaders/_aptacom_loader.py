__author__ = "satvshr"
__all__ = ["load_from_rcsb"]

from datasets import load_dataset
import pandas as pd

from pyaptamer.utils.pdb_to_struct import pdb_to_struct


def load_aptacom(format_='pandas', as_df=False):
    """
    Loads aptacom dataset from hugging face datasets.

    Parameters
    ----------
    format_: defines format compatibility in which dataset 
    is returned (i.e. pandas, dictionary...)

    as_df: (bool) Requires pandas compatible format; converts
    dataset into pandas dataframe

    
    Returns
    -------
    dataset: Hugging face dataset in a pandas compatible format.
    """
    aptacom = load_dataset('rpgv/AptaCom')['train'].with_format(format)
    if as_df and format_=='pandas': aptacom = pd.DataFrame().from_dict(aptacom);
    

    return aptacom
