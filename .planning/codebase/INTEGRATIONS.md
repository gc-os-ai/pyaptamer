# External Integrations

**Analysis Date:** 2026-04-13

## APIs & External Services

**Bioinformatics Databases:**
- UniProt API - Used for mapping PDB IDs to sequences.
  - Integration method: REST API via `requests.get`.
  - Auth: None detection (public access).
  - Used in: `pyaptamer/utils/_pdb_to_seq_uniprot.py`.

**Data Repositories:**
- Hugging Face Datasets - Used for loading remote datasets.
  - Integration method: `datasets` library and `requests`.
  - Auth: None detected for public datasets.
  - Used in: `pyaptamer/datasets/_loaders/_hf_to_dataset_loader.py`.
- Hugging Face Model Hub - Used for downloading pretrained weights.
  - Integration method: `torch.hub.load_state_dict_from_url`.
  - Used in: `pyaptamer/aptatrans/_model.py` (AptaTrans pretrained weights).

## Data Storage

**Files:**
- Local PDB/CSV files - Used for dataset storage in `pyaptamer/datasets/data/`.
- Pretrained weights - Stored in `.pt` files (e.g., `pyaptamer/aptatrans/weights/pretrained.pt`).

## CI/CD & Deployment

**CI Pipeline:**
- GitHub Actions - Detected via `.github` directory.
- Workflows: `release.yml` (detected in README).

## Environment Configuration

**Development:**
- Required tools: `pre-commit`, `ruff`, `pytest`.
- Data downloading: Requires internet access to fetch PDBs and UniProt sequences if not cached.

---

*Integration audit: 2026-04-13*
