# PyAptamer: Quick Start

## 1. Download Sample PDB Files

First, download some PDB files to work with:

```bash
python download_user_pdbs.py
```

- You will see a list of suggested PDB IDs.
- Enter your own IDs separated by commas, or just press Enter to download the examples.


## 2. Process a PDB File

Next, run the main workflow:

```bash
python -m pyaptamer.process_pdb
```

- A file dialog will appear. Select any `.pdb` or `.cif` file from the `data/pdb` folder (or elsewhere).
- The script will:
    - Analyze the structure (chains, residues, atoms, waters, ligands)
    - Clean the structure (remove waters/ligands)
    - Convert the cleaned structure to mmCIF
    - Verify the conversion and print a summary


## Example Output

```
===== PyAptamer PDB Processing Workflow =====

Selected file: /home/avinab/pyaptamer/data/pdb/1bna.pdb

🔬 Analyzing structure...
• Chains: B, A
• Residues: 104
• Atoms: 566
• Waters: 80
• Ligands: []

🧹 Cleaning structure...
✅ Saved cleaned structure: cleaned_structure.pdb

🔄 Converting format...
✅ Converted to mmCIF: converted_structure.cif

🔍 Verifying conversion...
• Format: mmCIF
• Waters in converted: 0

🎉 Workflow completed successfully!
```


## Requirements

- Python 3.8+
- Biopython
- gemmi
- tkinter (for GUI file dialog)


## Troubleshooting

- If the file dialog does not appear, make sure you are running on a machine with a graphical desktop and have `tkinter` installed (`sudo apt-get install python3-tk` on Ubuntu).
- If you get import errors, ensure you are running from the project root and using the `-m` flag as shown above.