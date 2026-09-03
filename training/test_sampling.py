"""Check that download.py's reservoir is uniform over the whole stream, not a prefix.

    uv run python training/test_sampling.py

The bug being guarded against: keeping the first N usable lines makes the sample a slice of
wherever the file happens to start. A correct reservoir draws position indices roughly
uniformly across the file, so the mean kept index should sit near the middle of the stream.
Not part of the submission.
"""

import random
from collections import Counter

POPULATION = 100_000
RESERVOIR = 1_000
TRIALS = 200


def reservoir_indices(population: int, size: int, rng: random.Random) -> list[int]:
    """The same accept/replace rule as download.py, run over plain indices."""
    held: list[int] = []
    for usable, i in enumerate(range(population)):
        slot = -1
        if len(held) < size:
            slot = len(held)
            held.append(i)
        else:
            j = rng.randrange(usable + 1)
            if j < size:
                slot = j
        if 0 <= slot < len(held):
            held[slot] = i
    return held


rng = random.Random(0)
means = []
bucket_counts: Counter[int] = Counter()
for _ in range(TRIALS):
    kept = reservoir_indices(POPULATION, RESERVOIR, rng)
    means.append(sum(kept) / len(kept))
    for idx in kept:
        bucket_counts[idx * 10 // POPULATION] += 1

observed = sum(means) / len(means)
expected = (POPULATION - 1) / 2
print(f"mean kept index: {observed:,.0f}   expected for a uniform sample: {expected:,.0f}")
print(f"deviation: {abs(observed - expected) / expected * 100:.2f}%")

print("\ncoverage by decile of the file (uniform => ~10% each):")
total = sum(bucket_counts.values())
worst = 0.0
for decile in range(10):
    share = bucket_counts[decile] / total * 100
    worst = max(worst, abs(share - 10.0))
    bar = "#" * int(share * 4)
    print(f"  {decile * 10:>3}-{decile * 10 + 10:>3}%  {share:5.2f}%  {bar}")

print(f"\nworst decile deviation from 10%: {worst:.2f} points")
ok = abs(observed - expected) / expected < 0.02 and worst < 1.5
print("PASS - reservoir is uniform across the file" if ok else "FAIL - sampling is skewed")
print("(a first-N prefix would put the mean near "
      f"{(RESERVOIR - 1) / 2:,.0f} and 100% in the first decile)")
