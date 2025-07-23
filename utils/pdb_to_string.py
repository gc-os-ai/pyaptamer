from Bio.PDB import PDBParser, PPBuilder


def extract_sequences_from_pdb(pdb_file_path):
    """
    Extracts amino acid sequences from a PDB file using Biopython.

    Args:
        pdb_file_path (str): Path to the PDB file.

    Returns:
        List[str]: List of amino acid sequences (one per polypeptide).
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("structure", pdb_file_path)

    ppb = PPBuilder()
    sequences = [str(pp.get_sequence()) for pp in ppb.build_peptides(structure)]
    return sequences
