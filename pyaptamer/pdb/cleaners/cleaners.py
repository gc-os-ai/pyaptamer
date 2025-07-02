import os
from Bio.PDB import Select

class RemoveWatersSelect(Select):
    def accept_residue(self, residue):
        return residue.get_resname() not in ['HOH', 'WAT']

class RemoveLigandsSelect(Select):
    def accept_residue(self, residue):
        return residue.id[0] == ' '  # standard residues only

def clean_structure(input_path, output_path, remove_waters=True, remove_ligands=True, remove_hydrogens=False, keep_chains=None):
    """Load, clean, and save structure with options."""
    from Bio.PDB import PDBParser, MMCIFParser, PDBIO
    ext = os.path.splitext(input_path)[1].lower()
    if ext == '.cif':
        parser = MMCIFParser()
    elif ext == '.pdb':
        parser = PDBParser()
    else:
        raise ValueError("Unsupported file extension")
    structure = parser.get_structure('target', input_path)

    # Apply cleaning
    from Bio.PDB.PDBIO import Select
    class CustomSelect(Select):
        def accept_residue(self, residue):
            if remove_waters and residue.get_resname() in ['HOH', 'WAT']:
                return False
            if remove_ligands and residue.id[0] != ' ':
                return False
            if keep_chains and residue.get_parent().id not in keep_chains:
                return False
            return True

    # Save cleaned structure
    io = PDBIO()
    io.set_structure(structure)
    io.save(output_path, select=CustomSelect())
