"""Does the corpus-field primitive generalize outside ARC entirely?

The claim under test: "find coordinates where a nuisance is a group action,
harvest invariants by harmonic analysis on the group" is not an ARC trick --
it is the same move classical cryptanalysis has used for a century on Caesar
ciphers. There the nuisance is a cyclic shift over Z_26 (not Z_10), and the
DFT-shift theorem gives an exact, closed-form recovery instead of just an
invariant: a circular shift by s in the letter-frequency profile multiplies
the k-th DFT coefficient by exp(-2*pi*i*k*s/26). So the PHASE of a stable
harmonic -- not just its magnitude, which is all corpusfield.py used -- names
the shift directly. This is the same power_spectrum machinery with the phase
term kept instead of discarded.

Two honest tests, both able to lose:

1. Self-consistency: encrypt one English text at every shift 0..25 and
   recover the shift via DFT phase alone (no frequency table, pure math).
   Must be exact if the DFT-shift theorem is doing real work.
2. Realistic cryptanalysis: calibrate the reference phase on ONE English
   text (this repo's README) and attack ciphertexts made from a DIFFERENT
   text (this repo's docstrings) at random shifts -- the real-world setting,
   where the attacker never sees the plaintext being decoded. Benchmarked
   against the classical chi-squared shift search (the standard technique)
   for both accuracy and the efficiency gain (O(1) closed form vs O(26)
   search) predicted by "the right lens for a Z_n nuisance is its harmonic
   analysis."
"""

from __future__ import annotations

import math
import random
import re
from pathlib import Path

ALPHABET = "abcdefghijklmnopqrstuvwxyz"
N = 26

# canonical relative English letter frequencies (Lewand 2000), used ONLY by
# the classical chi-squared baseline -- the DFT-phase method never touches it
ENGLISH_FREQ = {
    "e": 12.70, "t": 9.06, "a": 8.17, "o": 7.51, "i": 6.97, "n": 6.75, "s": 6.33,
    "h": 6.09, "r": 5.99, "d": 4.25, "l": 4.03, "c": 2.78, "u": 2.76, "m": 2.41,
    "w": 2.36, "f": 2.23, "g": 2.02, "y": 1.97, "p": 1.93, "b": 1.29, "v": 0.98,
    "k": 0.77, "j": 0.15, "x": 0.15, "q": 0.10, "z": 0.07,
}


def clean(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.lower())


def caesar_shift(text: str, s: int) -> str:
    return "".join(ALPHABET[(ALPHABET.index(c) + s) % N] for c in text)


def letter_profile(text: str) -> list[float]:
    counts = [0.0] * N
    for c in text:
        counts[ALPHABET.index(c)] += 1.0
    total = sum(counts) or 1.0
    return [c / total for c in counts]


def dft(profile: list[float], k: int) -> complex:
    return sum(x * complex(math.cos(2 * math.pi * k * i / N), -math.sin(2 * math.pi * k * i / N)) for i, x in enumerate(profile))


def recover_shift_by_phase(cipher_profile: list[float], ref_phase_k1: float, k: int = 1) -> int:
    """DFT-shift theorem: X_cipher[k] = X_plain[k] * exp(-2*pi*i*k*s/N), so
    phase(X_cipher[k]) - phase(X_plain[k]) = -2*pi*k*s/N (mod 2*pi)."""
    c = dft(cipher_profile, k)
    delta = ref_phase_k1 - math.atan2(c.imag, c.real)
    s = round((delta * N) / (2 * math.pi * k)) % N
    return s % N


def recover_shift_by_cross_correlation(cipher_profile: list[float], ref_profile: list[float]) -> int:
    """Full-spectrum version: circular cross-correlation of the cipher profile
    against the reference, computed via ALL 26 DFT harmonics at once
    (Wiener-Khinchin), not just the phase of k=1. This is the direct analog
    of corpusfield.power_spectrum keeping several k -- a single harmonic
    discards most of the profile's information; the full spectrum keeps it."""
    best_s, best_score = 0, -float("inf")
    for s in range(N):
        score = sum(ref_profile[i] * cipher_profile[(i + s) % N] for i in range(N))
        if score > best_score:
            best_score, best_s = score, s
    return best_s


def recover_shift_by_chi_squared(cipher_text: str) -> int:
    """Classical baseline: try all 26 shifts, pick the one whose de-shifted
    letter frequency best matches the canonical English table."""
    best_s, best_score = 0, float("inf")
    for s in range(N):
        candidate = caesar_shift(cipher_text, -s)
        counts = {c: 0 for c in ALPHABET}
        for c in candidate:
            counts[c] += 1
        total = len(candidate) or 1
        score = sum(
            ((counts[c] / total * 100) - ENGLISH_FREQ[c]) ** 2 / ENGLISH_FREQ[c] for c in ALPHABET
        )
        if score < best_score:
            best_score, best_s = score, s
    return best_s


def main() -> None:
    readme = clean(Path("README.md").read_text())
    # a disjoint text sample: every docstring in the package, excluding README
    docstrings = []
    for path in sorted(Path("arc_jgs2").glob("*.py")):
        text = path.read_text()
        docstrings.extend(re.findall(r'"""(.*?)"""', text, re.DOTALL))
    other_text = clean(" ".join(docstrings))
    print(f"reference text (README.md): {len(readme)} letters")
    print(f"attack text (arc_jgs2/*.py docstrings): {len(other_text)} letters")

    # --- 1. self-consistency: no frequency table, pure DFT-shift theorem ---
    ref_profile = letter_profile(readme)
    ref_phase = math.atan2(dft(ref_profile, 1).imag, dft(ref_profile, 1).real)

    print("\n1. self-consistency (same text, every shift, phase-only recovery):")
    exact = 0
    for s in range(N):
        cipher = caesar_shift(readme, s)
        recovered = recover_shift_by_phase(letter_profile(cipher), ref_phase)
        exact += recovered == s
    print(f"   exact recovery: {exact}/{N}")

    # --- 2. realistic cryptanalysis: different text, random shifts ---------
    print("\n2. cross-text cryptanalysis (reference=README, attack=docstrings):")
    rng = random.Random(0)
    # a second reference: the SAME canonical table the chi-squared baseline
    # uses, converted to a phase -- isolates "is the harmonic lens sound"
    # from "was README a representative enough corpus to calibrate it"
    canonical_profile = [ENGLISH_FREQ[c] for c in ALPHABET]
    canonical_phase = math.atan2(dft(canonical_profile, 1).imag, dft(canonical_profile, 1).real)

    phase_hits = phase_canonical_hits = xcorr_hits = xcorr_canonical_hits = chi2_hits = 0
    n_trials = 100
    chunk_len = 300
    for _ in range(n_trials):
        start = rng.randrange(0, max(1, len(other_text) - chunk_len))
        plain_chunk = other_text[start : start + chunk_len]
        true_s = rng.randrange(N)
        cipher = caesar_shift(plain_chunk, true_s)
        cipher_profile = letter_profile(cipher)

        phase_hits += recover_shift_by_phase(cipher_profile, ref_phase) == true_s
        phase_canonical_hits += recover_shift_by_phase(cipher_profile, canonical_phase) == true_s
        xcorr_hits += recover_shift_by_cross_correlation(cipher_profile, ref_profile) == true_s
        xcorr_canonical_hits += recover_shift_by_cross_correlation(cipher_profile, canonical_profile) == true_s
        chi2_hits += recover_shift_by_chi_squared(cipher) == true_s

    print(f"   DFT phase (k=1 only), reference = README:          {phase_hits}/{n_trials} exact")
    print(f"   DFT phase (k=1 only), reference = canonical table: {phase_canonical_hits}/{n_trials} exact")
    print(f"   full-spectrum cross-correlation, reference=README: {xcorr_hits}/{n_trials} exact")
    print(f"   full-spectrum cross-correlation, ref=canonical:    {xcorr_canonical_hits}/{n_trials} exact")
    print(f"   chi-squared vs canonical English table (search):   {chi2_hits}/{n_trials} exact")

    import timeit

    sample_cipher = caesar_shift(other_text[:300], 7)
    sample_profile = letter_profile(sample_cipher)
    t_phase = timeit.timeit(lambda: recover_shift_by_phase(sample_profile, ref_phase), number=200)
    t_xcorr = timeit.timeit(lambda: recover_shift_by_cross_correlation(sample_profile, ref_profile), number=200)
    t_chi2 = timeit.timeit(lambda: recover_shift_by_chi_squared(sample_cipher), number=200)
    print(
        f"\n   cost per attack: DFT phase {t_phase / 200 * 1e3:.3f} ms   "
        f"full-spectrum xcorr {t_xcorr / 200 * 1e3:.3f} ms   "
        f"chi-squared search {t_chi2 / 200 * 1e3:.3f} ms  ({t_chi2 / t_xcorr:.0f}x vs xcorr)"
    )


if __name__ == "__main__":
    main()
