Here is a clean and precise developer-facing markdown documentation for `Structure.py`, designed to follow the logic of the attached `Structure_IO.svg` diagram and explain how the class works and interacts with residue metadata. The goal is to ensure that any developer reading this can quickly understand the control flow, data shape, and design intent of the module.

---

# `Structure.py` — Developer Documentation

This module defines the `Structure` class, which encapsulates the molecular configuration of residue-based molecules (e.g., aptamers). It provides abstractions over residue metadata, rotatable bonds, and backbone structure, ultimately constructing AMBER-style inputs like `.lib` and `.frcmod` files to support force field parametrization in molecular simulations.

The class is central to the input preprocessing pipeline and is responsible for converting high-level residue specifications into concrete AMBER configuration strings.

---

## Class: `Structure`

### Purpose

The `Structure` class acts as a configuration holder for all relevant molecular properties of the input aptamer or peptide sequence. This includes:

- Loading of residue parameter files (`.lib`, `.frcmod`)
- Mapping residue names to atom-level aliases
- Defining rotation and backbone behavior for energy minimization and conformational sampling

---

![yaptamer/maws/docs/assets/Structure_IO.svg

## Input Overview

Each input parameter plays a role in downstream dictionary definitions and `.init_string` generation.

```text
residue_names → Residue types in the molecule
residue_length → Number of atoms in each residue
rotating_elements → Rotatable bond definitions per residue
backbone_elements → Backbone torsion definitions per residue
connect → Defines connectivity logic (e.g., end-to-end linkage)
residue_path → File path to load residue data (.lib/.frcmod)
alias → Defines how each residue is represented as 4 parts: [start, middle, ..., end]
```

---

## Key Internal Attributes

### 1. `self.init_string`

A multi-line string that contains LEaP-compatible commands for loading force field parameters for each residue:

```bash
loadoff ./A.lib
loadamberparams ./A.frcmod
```

### 2. `self.residue_length`

Maps each residue to its atom count. Used when dealing with negative indices (e.g., `-1` means last atom in residue).

### 3. `self.connect`

Defines how two residues connect. Default is:

```python
[[0, -1], [-2, 0], 1.6, 1.6]
```

Where `[0, -1]` are indices of atoms forming a bond, and `1.6` is the bond length placeholder.

### 4. `self.alias`

Stores how residues should be decomposed for edge cases like sequence capping or tail modifications. Defaults to `[residue]*4`.

### 5. `self.rotating_elements`

For each residue, stores a list of `[start, bond, end]` atom indices that define a rotatable unit. Can be `None` or `[None]` if undefined.

### 6. `self.backbone_elements`

Stores the atoms involved in backbone torsion angles in the form:

```python
[[start, mid_pre, bond], [mid_post, end]]
```

---

## Flow of `__init__` (Step-by-Step)

### 1. Load Force Field Files

If `residue_path` is provided, build `init_string` by injecting each residue's `.lib` and `.frcmod`.

### 2. Store Residue Lengths

If `residue_length` is given, index it against `residue_names`. Required for index correction logic later.

### 3. Define Connectivity

If custom `connect` is passed, override default per residue.

### 4. Alias Table Construction

Sets `self.alias[residue] = [residue]*4` by default, unless overridden. Later used in `.translate()` for sequence transformation.

### 5. Rotatable Bonds

Initializes each residue to `[None]`. If a `rotating_elements` list is passed, each tuple is resolved into residue-specific triplets. Handles negative indices by converting them into positive offsets from residue end using `residue_length`.

### 6. Backbone Definition

Similarly, backbone torsions are stored as resolved index triplets, handling negative indices and defaulting to `None` if absent.

---

## Method: `add_rotation`

```python
add_rotation(residue_name, rotations, basestring)
```

Used to dynamically add new rotatable definitions to an existing residue. It appends them to the internal `rotating_elements` dict.

---

## Method: `translate`

```python
translate("A B C D")
```

This uses `self.alias` to convert a sequence like `"A B C D"` into a mapped string like `"A_mid_start A_mid A_mid A_mid_end"`. It is primarily for custom sequence representation across the molecule, especially useful in linker or terminal residue customization.

---

## Notes

- **Negative Indexing**: Many of the rotation and backbone values support Python-style negative indexing. These are normalized based on `residue_length`.
- **Missing Data**: All key dictionaries (`residue_length`, `connect`, `alias`, etc.) are `defaultdicts`, meaning missing keys return a default value rather than throwing an error.

---

## Example Use Case

```python
Structure(
    residue_names=["A", "T"],
    residue_length=[12, 14],
    rotating_elements=[
        ("A", 1, 2, 5),
        ("T", -4, -3, -1)
    ],
    backbone_elements=[
        ("A", 0, 1, 2, 3, 4),
        ("T", 0, 2, 3, 4, 5)
    ],
    residue_path="./residues",
    alias=[
        ("A", "A_N", "A_M", "A_M", "A_C"),
        ("T", "T_N", "T_M", "T_M", "T_C")
    ]
)
```

---
