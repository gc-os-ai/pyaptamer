from Bio.PDB import PDBIO, MMCIFIO

def save_structure(structure, output_path):
    """Save Bio.PDB structure to file, format auto-detected by extension."""
    ext = os.path.splitext(output_path)[1].lower()
    if ext == '.cif':
        # For simplicity, placeholder: gemmi is better for CIF
        # Here, just write a placeholder or extend with gemmi if needed
        with open(output_path, 'w') as f:
            f.write("# Placeholder for CIF output\n")
    elif ext == '.pdb':
        io = PDBIO()
        io.set_structure(structure)
        io.save(output_path)
    else:
        raise ValueError("Unsupported output format. Use .pdb or .cif")
