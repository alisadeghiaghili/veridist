"""The balanced-tree merge the executors used before they merged results as they arrive.

It is kept as the oracle for the streaming merge: the engine now folds each partition's
state into one running value, and the exactness of the states makes that value equal, bit for
bit, to the one this tree gives for the same states in the order of their partition ordinals.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TypeVar

MergedT = TypeVar("MergedT")


def merge_tree(
    items: Sequence[MergedT], merge: Callable[[MergedT, MergedT], MergedT], identity: MergedT
) -> MergedT:
    """Merge ``items`` (in the order of their partition ordinals) as a balanced binary tree.

    Each level merges neighbours ``(0, 1)``, ``(2, 3)``, and so on; an odd item at the end
    moves up unchanged, and the levels are as many as halving the count to one takes. The
    shape depends on the number of items only. No items give ``identity``.
    """

    level = list(items) or [identity]
    for _ in range((len(level) - 1).bit_length()):
        level = [
            merge(level[index], level[index + 1]) if index + 1 < len(level) else level[index]
            for index in range(0, len(level), 2)
        ]
    return level[0]
