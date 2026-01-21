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
class ContractMapping:
    """Represents a mapping of contracts to lists of phrase variants.

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
    """

    mapping: Mapping[str, List[str]]
    descriptions: Optional[Mapping[str, str]] = field(default_factory=dict)

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
    to keys ``synonyms`` and optionally ``description``.  See
    ``configs/contract_mapping.yaml`` for an example.
    """
    path = Path(path)
    data = yaml.safe_load(path.read_text())
    mapping: Dict[str, List[str]] = {}
    descriptions: Dict[str, str] = {}
    for contract, entry in data.items():
        synonyms = entry.get("synonyms", [])
        mapping[contract] = synonyms
        if "description" in entry:
            descriptions[contract] = entry["description"]
    return ContractMapping(mapping, descriptions)