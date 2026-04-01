"""
Task sampler for Meta-RL on Points24.
Generates task distributions based on card difficulty tiers.
"""
import random
import itertools
from typing import List, Tuple, Optional, Dict


# Pre-computed sets of 4-card combos that are known to be solvable for 24
# We categorize by difficulty based on card values
DIFFICULTY_TIERS = {
    "easy": {
        "description": "Low numbers (1-5), more straightforward arithmetic",
        "card_range": (1, 5),
    },
    "medium": {
        "description": "Mixed numbers (1-8), moderate complexity",
        "card_range": (1, 8),
    },
    "hard": {
        "description": "Full range (1-10), includes larger numbers and face cards",
        "card_range": (1, 10),
    },
}


def _check_24_solvable(cards: List[int], target: int = 24) -> bool:
    """
    Check if a set of 4 cards can be combined to make the target (24).
    Uses brute-force enumeration of all possible expressions.
    """
    if len(cards) != 4:
        return False

    ops = ['+', '-', '*', '/']

    # Try all permutations of cards
    for perm in itertools.permutations(cards):
        a, b, c, d = [float(x) for x in perm]
        # Try all combinations of 3 operators
        for op1, op2, op3 in itertools.product(ops, repeat=3):
            # Try all 5 possible parenthesizations:
            # 1. ((a op1 b) op2 c) op3 d
            # 2. (a op1 (b op2 c)) op3 d
            # 3. (a op1 b) op2 (c op3 d)
            # 4. a op1 ((b op2 c) op3 d)
            # 5. a op1 (b op2 (c op3 d))
            expressions = []
            try:
                expressions.append(_apply_op(_apply_op(_apply_op(a, b, op1), c, op2), d, op3))
            except (ZeroDivisionError, ValueError):
                expressions.append(None)
            try:
                expressions.append(_apply_op(_apply_op(a, _apply_op(b, c, op2), op1), d, op3))
            except (ZeroDivisionError, ValueError):
                expressions.append(None)
            try:
                expressions.append(_apply_op(_apply_op(a, b, op1), _apply_op(c, d, op3), op2))
            except (ZeroDivisionError, ValueError):
                expressions.append(None)
            try:
                expressions.append(_apply_op(a, _apply_op(_apply_op(b, c, op2), d, op3), op1))
            except (ZeroDivisionError, ValueError):
                expressions.append(None)
            try:
                expressions.append(_apply_op(a, _apply_op(b, _apply_op(c, d, op3), op2), op1))
            except (ZeroDivisionError, ValueError):
                expressions.append(None)

            for val in expressions:
                if val is not None and abs(val - target) < 1e-9:
                    return True
    return False


def _apply_op(a: float, b: float, op: str) -> float:
    if op == '+':
        return a + b
    elif op == '-':
        return a - b
    elif op == '*':
        return a * b
    elif op == '/':
        if abs(b) < 1e-9:
            raise ZeroDivisionError
        return a / b
    raise ValueError(f"Unknown op: {op}")


class TaskSampler:
    """
    Samples card configurations grouped by difficulty for meta-RL.
    Pre-computes solvable configurations for each tier.
    """

    def __init__(self, tiers: Optional[List[str]] = None, precompute: bool = True,
                 max_per_tier: int = 500, seed: int = 42):
        """
        Args:
            tiers: List of difficulty tier names. Default: all tiers.
            precompute: Whether to precompute solvable combos at init.
            max_per_tier: Max number of solvable combos to cache per tier.
            seed: Random seed for reproducibility.
        """
        self.rng = random.Random(seed)
        self.tiers = tiers or list(DIFFICULTY_TIERS.keys())
        self.max_per_tier = max_per_tier
        self.solvable_cache: Dict[str, List[Tuple[int, ...]]] = {}

        if precompute:
            self._precompute_solvable()

    def _precompute_solvable(self):
        """Pre-compute solvable 4-card combos for each tier."""
        for tier in self.tiers:
            lo, hi = DIFFICULTY_TIERS[tier]["card_range"]
            solvable = []
            # Generate random combos and check solvability
            seen = set()
            attempts = 0
            max_attempts = self.max_per_tier * 20
            while len(solvable) < self.max_per_tier and attempts < max_attempts:
                cards = tuple(sorted([self.rng.randint(lo, hi) for _ in range(4)]))
                attempts += 1
                if cards in seen:
                    continue
                seen.add(cards)
                if _check_24_solvable(list(cards)):
                    solvable.append(cards)
            self.solvable_cache[tier] = solvable

    def sample_task(self, tier: Optional[str] = None) -> Tuple[int, ...]:
        """
        Sample a single task (4-card combo) from a tier.
        Args:
            tier: Difficulty tier. If None, sample uniformly from all tiers.
        Returns:
            Tuple of 4 card numbers.
        """
        if tier is None:
            tier = self.rng.choice(self.tiers)
        cached = self.solvable_cache.get(tier, [])
        if cached:
            return self.rng.choice(cached)
        # Fallback: random from range
        lo, hi = DIFFICULTY_TIERS[tier]["card_range"]
        return tuple(self.rng.randint(lo, hi) for _ in range(4))

    def sample_task_batch(self, n: int, tier: Optional[str] = None) -> List[Tuple[int, ...]]:
        """Sample n tasks from a tier (or mixed tiers)."""
        return [self.sample_task(tier) for _ in range(n)]

    def sample_diverse_batch(self, n_per_tier: int) -> Dict[str, List[Tuple[int, ...]]]:
        """Sample n_per_tier tasks from each tier."""
        return {tier: self.sample_task_batch(n_per_tier, tier) for tier in self.tiers}
