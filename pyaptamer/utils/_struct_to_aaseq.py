__author__ = "satvshr"
__all__ = ["struct_to_aaseq"]

from Bio.PDB.Polypeptide import PPBuilder


def struct_to_aaseq(structure):
    """
    Extract amino acid sequences from a Biopython Structure.

    Parameters
    ----------
    structure : Bio.PDB.Structure.Structure
        A Biopython Structure object (e.g. from PDBParser).

    Returns
    -------
    sequences : list of str
        List of amino acid sequences, one per polypeptide chain.
    """
    ppb = PPBuilder()
    sequences = [str(pp.get_sequence()) for pp in ppb.build_peptides(structure)]
    return sequences
