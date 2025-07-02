import os
import sys
import requests

# Add project root to Python path
sys.path.insert(0, os.path.abspath('.'))

from pyaptamer.pdb.io.loaders import load_from_file
from pyaptamer.pdb.cleaners.cleaners import clean_structure
from pyaptamer.pdb.structures.core import AptamerStructure
from pyaptamer.pdb.io.converters import pdb_to_mmcif, mmcif_to_pdb

def test_pdb_file(pdb_path):
    print(f"\n🔬 Testing {os.path.basename(pdb_path)}")
    
    # Load structure
    structure = load_from_file(pdb_path)
    apt_struct = AptamerStructure(structure)
    print(f"Waters before cleaning: {apt_struct.count_water_molecules()}")

    # Clean structure
    cleaned_path = pdb_path.replace('.pdb', '_cleaned.pdb')
    clean_structure(pdb_path, cleaned_path, remove_waters=True, remove_ligands=True)
    print(f"✅ Cleaned file saved: {os.path.basename(cleaned_path)}")

    # Reload cleaned structure
    cleaned_structure = load_from_file(cleaned_path)
    apt_struct_cleaned = AptamerStructure(cleaned_structure)
    print(f"Waters after cleaning: {apt_struct_cleaned.count_water_molecules()}")

    # Convert to mmCIF
    cif_path = cleaned_path.replace('.pdb', '.cif')
    pdb_to_mmcif(cleaned_path, cif_path)
    print(f"✅ Converted to mmCIF: {os.path.basename(cif_path)}")

    # Convert back to PDB
    roundtrip_path = cif_path.replace('.cif', '_roundtrip.pdb')
    mmcif_to_pdb(cif_path, roundtrip_path)
    print(f"✅ Converted back to PDB: {os.path.basename(roundtrip_path)}")

    print("Workflow test completed for", os.path.basename(pdb_path))

if __name__ == "__main__":
    pdb_dir = 'data/pdb'
    os.makedirs(pdb_dir, exist_ok=True)
    
    # List of sample PDB IDs to ensure we have
    sample_pdb_ids = ['1a2d', '1crn', '4hhb']
    
    # Download any missing sample PDB files
    for pdb_id in sample_pdb_ids:
        pdb_file = os.path.join(pdb_dir, f'{pdb_id}.pdb')
        if not os.path.exists(pdb_file):
            url = f'https://files.rcsb.org/download/{pdb_id}.pdb'
            response = requests.get(url)
            if response.status_code == 200:
                with open(pdb_file, 'w') as f:
                    f.write(response.text)
                print(f"Downloaded {pdb_id}.pdb")
            else:
                print(f"Failed to download {pdb_id}.pdb")
    
    # Get all PDB files in the directory
    pdb_files = [os.path.join(pdb_dir, f) for f in os.listdir(pdb_dir) if f.endswith('.pdb')]
    
    if not pdb_files:
        print("No PDB files found in data/pdb. Exiting.")
    else:
        for pdb_file in pdb_files:
            test_pdb_file(pdb_file)
