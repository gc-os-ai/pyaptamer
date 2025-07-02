import gemmi
from Bio.PDB import PDBIO, MMCIFIO

def pdb_to_mmcif(pdb_path: str, mmcif_path: str):
    """Convert PDB to mmCIF using gemmi."""
    structure = gemmi.read_structure(pdb_path)
    structure.make_mmcif_document().write_file(mmcif_path)

def mmcif_to_pdb(mmcif_path: str, pdb_path: str):
    """Convert mmCIF to PDB using gemmi."""
    structure = gemmi.read_structure(mmcif_path)
    structure.write_pdb(pdb_path)

def convert_format(input_path: str, output_path: str, target_format: str):
    """Convert between PDB and mmCIF formats."""
    ext = target_format.lower()
    if ext in ['mmcif', 'cif']:
        pdb_to_mmcif(input_path, output_path)
    elif ext in ['pdb']:
        mmcif_to_pdb(input_path, output_path)
    else:
        raise ValueError(f"Unsupported target format: {target_format}")
