import os
from pyaptamer.pdb import (
    select_structure_file,
    analyze_structure,
    clean_structure,
    convert_format
)

def main():
    print("===== PyAptamer PDB Processing Workflow =====")

    # 1. Select file interactively
    input_file = select_structure_file()
    if not input_file:
        print("No file selected. Exiting.")
        return

    input_dir = os.path.dirname(input_file)
    input_base = os.path.splitext(os.path.basename(input_file))[0]

    print(f"\nSelected file: {input_file}")

    # 2. Analyze structure
    print("\n🔬 Analyzing structure...")
    analysis = analyze_structure(input_file)
    print(f"• Chains: {', '.join(analysis['chains'])}")
    print(f"• Residues: {analysis['residues']}")
    print(f"• Atoms: {analysis['atoms']}")
    print(f"• Waters: {analysis['waters']}")
    print(f"• Ligands: {analysis['ligands']}")

    # 3. Clean structure
    print("\n🧹 Cleaning structure...")
    cleaned_file = os.path.join(input_dir, f"{input_base}_cleaned.pdb")
    clean_structure(
        input_file,
        cleaned_file,
        remove_waters=True,
        remove_ligands=True
    )
    print(f"✅ Saved cleaned structure: {cleaned_file}")

    # 4. Convert to mmCIF
    print("\n🔄 Converting format...")
    cif_file = os.path.join(input_dir, f"{input_base}_cleaned.cif")
    convert_format(cleaned_file, cif_file, "mmcif")
    print(f"✅ Converted to mmCIF: {cif_file}")

    # 5. Verify conversion
    print("\n🔍 Verifying conversion...")
    converted_analysis = analyze_structure(cif_file)
    print(f"• Format: {converted_analysis['format']}")
    print(f"• Waters in converted: {converted_analysis['waters']}")

    print("\n🎉 Workflow completed successfully!")

if __name__ == "__main__":
    main()
