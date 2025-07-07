import os

from Bio.PDB import PDBParser, MMCIFParser

def analyze_structure(input_path):
    ext = os.path.splitext(input_path)[1].lower()
    if ext == '.cif':
        parser = MMCIFParser()
    elif ext == '.pdb':
        parser = PDBParser()
    else:
        raise ValueError("Unsupported file extension")
    structure = parser.get_structure('target', input_path)

    chains = set()
    residues = 0
    atoms = 0
    waters = 0
    ligands = set()
    resolution = None

    # Extract resolution if available
    try:
        if hasattr(structure, 'header') and 'resolution' in structure.header:
            resolution = structure.header['resolution']
    except:
        pass

    for model in structure:
        for chain in model:
            chains.add(chain.id)
            for residue in chain:
                residues += 1
                if residue.get_resname() in ['HOH', 'WAT']:
                    waters += 1
                elif residue.id[0] != ' ':
                    ligands.add(residue.get_resname())
                atoms += len(list(residue.get_atoms()))

    return {
        "format": "mmCIF" if ext == '.cif' else "PDB",
        "chains": list(chains),
        "residues": residues,
        "atoms": atoms,
        "waters": waters,
        "ligands": list(ligands),
        "resolution": resolution,
        "errors": []  # Could add validation here
    }
