import sys
import os
sys.path.insert(0, os.path.abspath('.'))  # Add current directory to path

from pyaptamer.pdb import (
    analyzers, 
    cleaners, 
    converters,
    utils
)

def test_pdb_file(pdb_path):
    print(f"\n🔬 Testing {os.path.basename(pdb_path)}")
    
    # 1. Analyze structure
    analysis = analyzers.analyze_structure(pdb_path)
    print(f"Analysis:\n{analysis}")
    
    # 2. Clean structure
    cleaned_path = pdb_path.replace(".pdb", "_cleaned.pdb")
    cleaners.clean_structure(
        pdb_path, 
        cleaned_path,
        remove_waters=True,
        remove_ligands=True
    )
    print(f"✅ Cleaned file saved: {os.path.basename(cleaned_path)}")
    
    # 3. Convert to mmCIF
    cif_path = cleaned_path.replace(".pdb", ".cif")
    converters.convert_format(cleaned_path, cif_path, "mmcif")
    print(f"✅ Converted to mmCIF: {os.path.basename(cif_path)}")
    
    return analysis

if __name__ == "__main__":
    samples = [
        "data/pdb/1crn.pdb",
        "data/pdb/1a2d.pdb",
        "data/pdb/4hhb.pdb"
    ]
    
    for sample in samples:
        if os.path.exists(sample):
            test_pdb_file(sample)
        else:
            print(f"⚠️ Missing sample: {sample}")
