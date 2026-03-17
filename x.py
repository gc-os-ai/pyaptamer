from pathlib import Path

from pyaptamer.data.loader import MoleculeLoader

# adjust this to your repo root if needed
root_path = Path(__file__).parent.parent  # or Path.cwd()

pdb_paths = [
    "C:\\Users\\satvm\\pyaptamer\\pyaptamer\\datasets\\data\\1gnh.pdb",
    "C:\\Users\\satvm\\pyaptamer\\pyaptamer\\datasets\\data\\5nu7.pdb",
]

loader = MoleculeLoader(pdb_paths)
df = loader.to_df_seq()

print(df)
