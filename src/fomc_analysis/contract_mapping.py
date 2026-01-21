"""
contract_mapping
================

Classes and utilities for working with the mapping between market
contracts and the phrases that count as a mention of each contract.
The mapping can be loaded from a YAML or JSON file and provides
methods to test whether a given text contains a mention and to
produce a clean set of phrases for counting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Mapping

import yaml


@dataclass
class ContractSpec:
    """Specification for a single contract.

    Parameters
    ----------
    synonyms: List[str]
        List of lowercase phrase variants that count as a mention.
    threshold: int
        Minimum count required for contract to resolve YES. Default is 1.
    scope: str
        Which text to search: "powell_only" or "full_transcript". Default is "powell_only".
    match_mode: str
        How to match phrases: "strict_literal" or "variants". Default is "strict_literal".
    count_unit: str
        What to count: "token" or "phrase". Default is "token".
    description: Optional[str]
        Human-readable description of the contract.
    """

    synonyms: List[str]
    threshold: int = 1
    scope: str = "powell_only"
    match_mode: str = "strict_literal"
    count_unit: str = "token"
    description: Optional[str] = None


@dataclass
class ContractMapping:
    """Represents a mapping of contracts to their specifications.

    Parameters
    ----------
    mapping: Mapping[str, List[str]]
        A dictionary mapping contract names to a list of lowercase
        phrase variants that count as a mention.  All variants
        should already be lowercase; the code lowercases the text it
        searches against.
    descriptions: Optional[Mapping[str, str]]
        Optional descriptions for each contract.  These are not used
        programmatically but can be helpful when generating reports.
    specs: Optional[Mapping[str, ContractSpec]]
        Optional full specifications for each contract including
        threshold, scope, match_mode, and count_unit.
    """

    mapping: Mapping[str, List[str]]
    descriptions: Optional[Mapping[str, str]] = field(default_factory=dict)
    specs: Optional[Mapping[str, ContractSpec]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # normalise synonyms to lower case and strip whitespace
        normalised = {}
        for contract, synonyms in self.mapping.items():
            normalised[contract] = [s.lower().strip() for s in synonyms]
        self.mapping = normalised

    def contracts(self) -> Iterable[str]:
        """Return an iterable of contract names."""
        return self.mapping.keys()

    def phrases_for(self, contract: str) -> List[str]:
        """Return the list of phrase variants for a contract."""
        return list(self.mapping.get(contract, []))

    def get_spec(self, contract: str) -> Optional[ContractSpec]:
        """Return the ContractSpec for a contract, or None if not found."""
        return self.specs.get(contract) if self.specs else None

    def get_threshold(self, contract: str) -> int:
        """Return the threshold for a contract (default: 1)."""
        spec = self.get_spec(contract)
        return spec.threshold if spec else 1

    def get_scope(self, contract: str) -> str:
        """Return the scope for a contract (default: "powell_only")."""
        spec = self.get_spec(contract)
        return spec.scope if spec else "powell_only"

    def get_match_mode(self, contract: str) -> str:
        """Return the match_mode for a contract (default: "strict_literal")."""
        spec = self.get_spec(contract)
        return spec.match_mode if spec else "strict_literal"

    def get_count_unit(self, contract: str) -> str:
        """Return the count_unit for a contract (default: "token")."""
        spec = self.get_spec(contract)
        return spec.count_unit if spec else "token"

    def match_in_text(self, contract: str, text: str) -> bool:
        """Return True if any variant of a contract appears in the text.

        Parameters
        ----------
        contract: str
            The name of the contract.
        text: str
            The lower‑case text to search for a mention.

        Returns
        -------
        bool
            True if any of the contract's synonyms is a substring of
            ``text``; False otherwise.
        """
        text = text.lower()
        for phrase in self.mapping.get(contract, []):
            if phrase and phrase in text:
                return True
        return False

    def count_in_text(self, contract: str, text: str) -> int:
        """Count how many times any variant of a contract appears in text.

        This function counts overlapping mentions; if multiple
        variants overlap (e.g. "AI" inside "artificial intelligence") the
        counts may be inflated.  For most cases this simple count is
        sufficient.  If you need more sophisticated counting (e.g.
        using regex word boundaries) you can extend this method.
        """
        text = text.lower()
        count = 0
        for phrase in self.mapping.get(contract, []):
            if not phrase:
                continue
            # naive count of substring occurrences
            idx = text.find(phrase)
            while idx != -1:
                count += 1
                idx = text.find(phrase, idx + len(phrase))
        return count


def load_mapping_from_file(path: str | Path) -> ContractMapping:
    """Load a contract mapping from a YAML or JSON file.

    Parameters
    ----------
    path: str or Path
        Path to a YAML or JSON file defining the contract mapping.

    Returns
    -------
    ContractMapping
        An instance initialised with the mapping defined in the file.

    Notes
    -----
    The YAML/JSON structure should be a mapping from contract names
    to keys ``synonyms`` and optionally ``description``, ``threshold``,
    ``scope``, ``match_mode``, and ``count_unit``.  See
    ``configs/contract_mapping.yaml`` for an example.
    """
    path = Path(path)
    data = yaml.safe_load(path.read_text())
    mapping: Dict[str, List[str]] = {}
    descriptions: Dict[str, str] = {}
    specs: Dict[str, ContractSpec] = {}

    for contract, entry in data.items():
        synonyms = entry.get("synonyms", [])
        mapping[contract] = synonyms

        # Parse optional fields
        threshold = entry.get("threshold", 1)
        scope = entry.get("scope", "powell_only")
        match_mode = entry.get("match_mode", "strict_literal")
        count_unit = entry.get("count_unit", "token")
        description = entry.get("description")

        if description:
            descriptions[contract] = description

        # Create ContractSpec
        specs[contract] = ContractSpec(
            synonyms=synonyms,
            threshold=threshold,
            scope=scope,
            match_mode=match_mode,
            count_unit=count_unit,
            description=description,
        )

    return ContractMapping(mapping, descriptions, specs)