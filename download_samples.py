import os
from Bio.PDB import PDBList

def download_user_pdbs():
    pdb_list = PDBList()
    os.makedirs("data/pdb", exist_ok=True)

    # Suggested PDB IDs (small, medium, large, DNA, RNA, ligand-bound, etc.)
    suggestions = {
        "1crn": "Small protein (crambin)",
        "1a2d": "DNA-binding protein",
        "4hhb": "Hemoglobin (tetramer, large)",
        "1bna": "DNA dodecamer",
        "2hbb": "Hemoglobin beta chain",
        "1c0a": "Protein-ligand complex",
        "6lu7": "SARS-CoV-2 main protease"
    }

    print("Some example PDB IDs you can try:")
    for pdb_id, desc in suggestions.items():
        print(f"  {pdb_id:6} - {desc}")

    user_input = input(
        "\nEnter PDB IDs separated by commas (or press Enter to download the above examples): "
    ).strip()

    if not user_input:
        pdb_ids = list(suggestions.keys())
        print("\nNo input detected. Downloading example PDBs:")
    else:
        pdb_ids = [pdb_id.strip().lower() for pdb_id in user_input.split(",") if pdb_id.strip()]

    for pdb_id in pdb_ids:
        print(f"\nDownloading {pdb_id}...")
        pdb_path = pdb_list.retrieve_pdb_file(
            pdb_id,
            pdir="data/pdb",
            file_format="pdb",
            overwrite=True
        )
        # Rename to consistent format
        new_path = f"data/pdb/{pdb_id}.pdb"
        os.rename(pdb_path, new_path)
        print(f"✅ {pdb_id}.pdb downloaded as {new_path}")

if __name__ == "__main__":
    download_user_pdbs()
