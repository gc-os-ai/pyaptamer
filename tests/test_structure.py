# tests/test_structure.py

import pytest
from pyaptamer.maws.structure import Structure

def test_fill_defaults_and_init_string(tmp_path):
    """
    Test that when only residue_names is provided (with a custom residue_path),
    all per-residue dicts get the correct defaults, and init_string contains
    exactly two lines per residue in the right order.
    """
    # Create a Structure with two residues and point residue_path at tmp_path
    s = Structure(
        residue_names=["A", "B"],
        residue_path=str(tmp_path),
    )

    # --- Defaults from _fill_defaults() ---
    # residue_length should default to 0 for each residue
    assert s.residue_length["A"] == 0
    assert s.residue_length["B"] == 0

    # connect should default to [[0,-1],[-2,0],1.6,1.6]
    assert s.connect["A"] == [[0, -1], [-2, 0], 1.6, 1.6]
    assert s.connect["B"] == [[0, -1], [-2, 0], 1.6, 1.6]

    # alias should default to [name]*4
    assert s.alias["A"] == ["A", "A", "A", "A"]
    assert s.alias["B"] == ["B", "B", "B", "B"]

    # rotating_elements and backbone_elements should default to empty lists
    assert s.rotating_elements["A"] == []
    assert s.rotating_elements["B"] == []
    assert s.backbone_elements["A"] == []
    assert s.backbone_elements["B"] == []

    # --- init_string via _build_init_string() ---
    # Expect 2 lines per residue: one loadoff, one loadamberparams
    lines = s.init_string.splitlines()
    assert len(lines) == 2 * len(s.residue_names)

    # Check content and order
    for idx, name in enumerate(["A", "B"]):
        loadoff_line = lines[2 * idx]
        params_line = lines[2 * idx + 1]
        expected_path = f"{tmp_path}/{name}"
        assert loadoff_line == f"loadoff {expected_path}.lib"
        assert params_line == f"loadamberparams {expected_path}.frcmod"


def test_validation_raises_for_unknown_mappings():
    """
    If any of the provided dicts contain keys not in residue_names,
    __post_init__ should immediately raise a ValueError naming the bad keys.
    """
    # Unknown key in residue_length
    with pytest.raises(ValueError) as exc:
        Structure(
            residue_names=["X"],
            residue_length={"Y": 3},
        )
    assert "residue_length: unknown residue(s): Y" in str(exc.value)

    # Unknown key in connect
    with pytest.raises(ValueError) as exc2:
        Structure(
            residue_names=["A"],
            connect={"B": [[0,1],[2,3],1.6,1.6]},
        )
    assert "connect: unknown residue(s): B" in str(exc2.value)


def test_custom_initial_mappings_are_respected(tmp_path):
    """
    Passing custom residue_length, connect, alias, rotating_elements,
    and backbone_elements at init-time should be preserved (and not
    overwritten by defaults).
    """
    names = ["Z"]
    path = tmp_path / "res"
    path.mkdir()
    # create dummy lib/frcmod files for init_string
    (path / "Z.lib").write_text("")
    (path / "Z.frcmod").write_text("")

    # Prepare custom dicts
    length = {"Z": 42}
    connect = {"Z": [[9,8],[7,6],2.0,2.1]}
    alias = {"Z": ["n0","n1","n2","n3"]}
    rotating = {"Z": [[1, 2, 3]]}
    backbone = {"Z": [[4,5,6],[7,8]]}

    s = Structure(
        residue_names=names,
        residue_path=str(path),
        residue_length=length,
        connect=connect,
        alias=alias,
        rotating_elements=rotating,
        backbone_elements=backbone,
    )

    # Check that nothing was clobbered by _fill_defaults
    assert s.residue_length["Z"] == 42
    assert s.connect["Z"] == [[9,8],[7,6],2.0,2.1]
    assert s.alias["Z"] == ["n0","n1","n2","n3"]
    assert s.rotating_elements["Z"] == [[1,2,3]]
    assert s.backbone_elements["Z"] == [[4,5,6],[7,8]]


def test_add_rotation_behaviour():
    """
    Test that add_rotation appends new entries to rotating_elements[residue]:
    - a single string becomes a one-element list
    - a list of strings extends the list
    - a list of ints (numeric triplet) is also extended (flattened) per current implementation
    """
    s = Structure(residue_names=["R"])
    # initially empty
    assert s.rotating_elements["R"] == []

    # add a single tag
    result = s.add_rotation("R", "phi")
    assert result == ["phi"], "Single-string should be wrapped and appended"

    # add multiple tags
    result = s.add_rotation("R", ["psi", "omega"])
    assert result[-2:] == ["psi", "omega"], "List of strings should extend"

    # add a numeric triplet (current behavior: extend flattens them)
    result = s.add_rotation("R", [1, 2, 3])
    assert result[-3:] == [1, 2, 3], "Numeric triplet is flattened by extend()"


def test_translate_single_and_multi():
    """
    Test translate() for:
    - a singleton sequence → alias[name][0]
    - a two-residue and three-residue chain
    """
    s = Structure(residue_names=["A","B"])
    # override alias mapping for clarity
    s.alias = {
        "A": ["A0", "A1", "A2", "A3"],
        "B": ["B0", "B1", "B2", "B3"],
    }

    # single residue → N-term form alias[0]
    assert s.translate("A") == "A0"

    # two residues → N-term of A, C-term of B
    assert s.translate("A B") == "A1 B3"

    # three residues → N-term of A, mid of B, C-term of A
    assert s.translate("A B A") == "A1 B2 A3"
