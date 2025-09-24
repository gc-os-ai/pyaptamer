__author__ = "satvshr"
__all__ = ["load_from_rcsb"]

from datasets import load_dataset

from pyaptamer.utils.pdb_to_struct import pdb_to_struct


def load_aptacom():
    """
    Loads aptacom dataset from hugging face datasets.

    Parameters
    ----------
    None (for now)

    
    Returns
    -------
    dataset: Hugging face dataset in a pandas compatible format.
    """
    aptacom = load_dataset('rpgv/AptaCom')['train'].with_format('pandas')
    

    return aptacom
