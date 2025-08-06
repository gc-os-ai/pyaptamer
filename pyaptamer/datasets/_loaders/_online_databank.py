from Bio.PDB import PDBList

from pyaptamer.utils.pdb_to_struct import pdb_to_struct
from pyaptamer.utils.struct_to_aaseq import struct_to_aaseq


def download_and_extract_sequences(pdb_id):
    """
    Download a PDB file, parse it into a structure, and extract amino acid sequences.

    Parameters
    ----------
    pdb_id : str
        The PDB ID of the structure to download.

    Returns
    -------
    sequences : list of str
        List of amino acid sequences extracted from the structure.
    """
    pdbl = PDBList()
    pdb_file_path = pdbl.retrieve_pdb_file(pdb_id, file_format="pdb")

    structure = pdb_to_struct(pdb_file_path)
    sequences = struct_to_aaseq(structure)

    return sequences
