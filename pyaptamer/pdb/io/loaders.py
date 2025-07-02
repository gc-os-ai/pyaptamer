import os
import gemmi
from Bio.PDB import PDBParser, MMCIFParser

def load_from_file(path: str):
    """Load structure from PDB or mmCIF file, auto-detect format."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")
    ext = os.path.splitext(path)[1].lower()
    if ext == '.cif':
        parser = MMCIFParser()
    elif ext == '.pdb':
        parser = PDBParser()
    else:
        raise ValueError("Unsupported file extension. Use .pdb or .cif")
    return parser.get_structure('structure', path)

def fetch_from_pdbe(pdb_id: str):
    """Download structure from PDBe and load."""
    import requests
    url = f'https://files.rcsb.org/download/{pdb_id.upper()}.cif'
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(f"Failed to fetch PDB ID {pdb_id}")
    filename = f"{pdb_id}.cif"
    with open(filename, 'w') as f:
        f.write(response.text)
    return load_from_file(filename)
