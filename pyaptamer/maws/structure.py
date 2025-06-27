"""
structure.py

Defines the `Structure` dataclass to model chemical residue-based molecular
structures, including rotation and backbone definitions.  Integrates with OpenMM.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TypeAlias, Sequence, Union

# ── compact, self-documenting type aliases ───────────────────────────
RotSpec: TypeAlias = tuple[str, int, int, int | None]  # (res, i, j, k)
BackboneSpec: TypeAlias = tuple[str, int, int, int, int, int]  # (res, s, mp, b, mpo, e)

# heterogenous entry types -------------------------------------------------------
RotationTriplet: TypeAlias = list[int | None]  # [i, j, k] indices
RotationEntry: TypeAlias = Union[RotationTriplet, str]  # numeric OR tag string
RotationList: TypeAlias = list[RotationEntry]

ConnectEntry: TypeAlias = Union[list[int], float]  # inner bond list or distance
ConnectSpec: TypeAlias = list[ConnectEntry]


log = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# main dataclass
# ----------------------------------------------------------------------
@dataclass(slots=True)
class Structure:
    """
    Defines chemical properties for molecules based on residues.

    Attributes
    ----------
    residue_names : list[str]
        All residue identifiers handled by this Structure.
    residue_length : dict[str, int]
        Number of atoms per residue.
    rotating_elements : dict[str, list[list[int | None]]]
        [start, bond, end] triplets that may rotate.
    backbone_elements : dict[str, list[list[int]]]
        Two substructures per residue: [[s, mp, b], [mpo, e]]
    connect : dict[str, list[list[int | float]]]
        Bond indices and default distances for linking residues.
    alias : dict[str, list[str]]
        Four-entry translation for each residue (N-term, mid, mid, C-term).
    init_string : str
        TLeap commands for loading each residue force-field file.
    """

    # ------------------------------- core data ------------------------
    residue_names: list[str]
    residue_path: str = "."
    residue_length: dict[str, int] = field(default_factory=dict)
    connect: dict[str, ConnectSpec] = field(default_factory=dict)
    alias: dict[str, list[str]] = field(default_factory=dict)
    rotating_elements: dict[str, RotationList] = field(default_factory=dict)
    backbone_elements: dict[str, list[list[int]]] = field(default_factory=dict)

    # ------------------------------- derived --------------------------
    init_string: str = field(init=False)

    # ------------------------------------------------------------------
    # post-init = light validation + default filling
    # ------------------------------------------------------------------
    def __post_init__(self) -> None:  # noqa: D401
        """Validate inputs and build defaults, then compose `init_string`."""
        self._validate_residue_names()
        self._fill_defaults()
        self.init_string = self._build_init_string()

    # ==================================================================
    # public helpers
    # ==================================================================
    def add_rotation(self, residue: str, rot: str | Sequence[str]) -> RotationList:
        """
        Append one or many rotation specifications to *residue*.

        Parameters
        ----------
        residue : str
            Residue to modify.
        rot : str | list[str]
            Either a single rotation spec or a list of them.

        Returns
        -------
        list[list[int | None]]
            Updated rotation table for the residue.
        """
        if isinstance(rot, str):
            rot = [rot]
        self.rotating_elements[residue].extend(rot)
        return self.rotating_elements[residue]

    def translate(self, sequence: str) -> str:
        """
        Convert a space-separated residue string to its alias form.

        Notes
        -----
        * single residue  → alias[0]
        * multi-residue   → N-term alias[1], middle alias[2], C-term alias[3]
        """
        parts = sequence.split()
        if len(parts) == 1:
            return self.alias[parts[0]][0]

        return " ".join(
            [self.alias[parts[0]][1]]
            + [self.alias[p][2] for p in parts[1:-1]]
            + [self.alias[parts[-1]][3]]
        )

    # ==================================================================
    # private helpers
    # ==================================================================
    def _validate_residue_names(self) -> None:
        """Raise early if any dict key refers to an unknown residue."""
        known = set(self.residue_names)
        for name_set, label in [
            (self.residue_length, "residue_length"),
            (self.connect, "connect"),
            (self.alias, "alias"),
            (self.rotating_elements, "rotating_elements"),
            (self.backbone_elements, "backbone_elements"),
        ]:
            unknown = set(name_set) - known
            if unknown:
                raise ValueError(f"{label}: unknown residue(s): {', '.join(unknown)}")

    def _fill_defaults(self) -> None:
        """Populate missing per-residue entries with safe defaults."""
        for name in self.residue_names:
            self.residue_length.setdefault(name, 0)
            self.connect.setdefault(name, [[0, -1], [-2, 0], 1.6, 1.6])
            self.alias.setdefault(name, [name] * 4)
            self.rotating_elements.setdefault(name, [])
            self.backbone_elements.setdefault(name, [])

    def _build_init_string(self) -> str:
        """Compose TLeap commands for each residue."""
        lines: list[str] = []
        for name in self.residue_names:
            path = f"{self.residue_path}/{name}"
            log.debug("Building force-field load string for %s", path)
            lines.append(f"loadoff {path}.lib\nloadamberparams {path}.frcmod")
        return "\n".join(lines)


# ----------------------------------------------------------------------
#  quick check
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # 1. build the tiniest valid Structure (one residue, aliases required)
    s = Structure(
        residue_names=["ALA"],
        alias={"ALA": ["ALA", "ALA", "ALA", "ALA"]},
    )

    # 2. sanity: defaults were filled
    assert s.connect["ALA"][2:] == [1.6, 1.6]

    # 3. public helpers still work
    assert s.translate("ALA") == "ALA"
    s.add_rotation("ALA", "chi1")
    assert s.rotating_elements["ALA"][-1] == "chi1"

    print("check passed")
