
---

# 📄 `docs/aptatrans/overview.md`

```md
# AptaTrans Overview

AptaTrans is a deep learning model used to predict interactions between aptamers and proteins.

## Key Idea

The model takes two sequences:

- Aptamer sequence
- Protein sequence

It processes them and predicts interaction strength or compatibility.

## Workflow

1. Encode sequences into embeddings
2. Generate interaction map
3. Apply convolutional layers
4. Produce prediction output

## Features

- Uses sequence embeddings
- Convolution-based architecture
- Designed for biological sequence interaction tasks

## Notes

- Input sequences must be properly encoded
- Model expects tensor inputs
- Output shape depends on task configuration