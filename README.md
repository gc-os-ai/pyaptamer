# PyAptamer: Quick Start

## 1.Prerequisites

To use this package, **you must be added as a collaborator**.
Ask the repository owner to invite you if you don’t have access.

- Get the HTTPS URL from the green "Code" button on the repo page (`https://github.com/avinab/pyaptamer`).
- Open a terminal and run:

```bash
git clone https://github.com/avinab/pyaptamer.git
```

- When prompted:
    - Enter **your own GitHub username**.
    - For password, use **your own Personal Access Token (PAT)** if you have two-factor authentication enabled.
(Generate one at [GitHub Settings > Developer settings > Personal access tokens](https://github.com/settings/tokens))
- Change into the project directory:

```bash
cd pyaptamer
```

**Now you can proceed to install requirements and use the package!**


## 2. Download Sample PDB Files

First, download some PDB files to work with:

```bash
python download_user_pdbs.py
```

- You will see a list of suggested PDB IDs.
- Enter your own IDs separated by commas, or just press Enter to download the examples.


## 3. Process a PDB File

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