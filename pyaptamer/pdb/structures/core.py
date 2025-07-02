from Bio.PDB import PDBParser, MMCIFParser

class AptamerStructure:
    def __init__(self, structure):
        self.structure = structure

    @classmethod
    def load(cls, path):
        ext = os.path.splitext(path)[1].lower()
        if ext == '.cif':
            parser = MMCIFParser()
        elif ext == '.pdb':
            parser = PDBParser()
        else:
            raise ValueError("Unsupported file extension")
        structure = parser.get_structure('structure', path)
        return cls(structure)

    def count_water_molecules(self):
        count = 0
        for model in self.structure:
            for chain in model:
                for residue in chain:
                    if residue.get_resname() in ['HOH', 'WAT']:
                        count += 1
        return count
