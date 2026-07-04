"""Port the corpus-cylinder geometry lens onto MNIST-1D as a cheap testbed.

`linefeatures.py` lifts a discrete, categorical ARC grid to a 3D polyline (a
color-wheel cylinder) so that Frenet curvature/torsion -- shape invisible in
the raw symbol sequence -- can be read off and checked against solver
difficulty. MNIST-1D signals are already continuous curves, so there is
nothing to lift *out of*; but they are still only 1D, and classic time-series
analysis has its own lift: a Takens delay embedding

    p(t) = (x(t), x(t + tau), x(t + 2*tau))

turns a 1D signal into a 3D curve whose shape reflects the signal's local
dynamics (this is the standard nonlinear-dynamics trick for recovering
attractor geometry from a scalar time series). We reuse the exact same
`frenet_stats` Frenet invariants from `linefeatures.py` on this embedding --
same question, same math, different lift.

Question: does delay-embedded curvature/torsion of the class *templates*
predict which digit classes a real classifier actually confuses on the noisy,
augmented dataset -- the MNIST-1D analog of "does cylinder geometry predict
ARC solver difficulty"? And, matching the honesty standard applied there: does
it beat a trivial baseline (raw Euclidean distance between templates)?
"""

from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path

MNIST1D_REPO = Path("/home/ath/mnist1d")
if str(MNIST1D_REPO) not in sys.path:
    sys.path.insert(0, str(MNIST1D_REPO))

from .linefeatures import PolyGeom, frenet_stats  # noqa: E402


def delay_embed(signal: list[float], tau: int = 1) -> list[tuple[float, float, float]]:
    """Takens delay embedding of a 1D signal into 3D: (x[t], x[t+tau], x[t+2*tau])."""
    n = len(signal)
    return [(signal[t], signal[t + tau], signal[t + 2 * tau]) for t in range(n - 2 * tau)]


def template_geometry(signal: list[float], tau: int = 1) -> PolyGeom:
    return frenet_stats(delay_embed(signal, tau=tau))


def _geom_vector(g: PolyGeom) -> tuple[float, ...]:
    d = asdict(g)
    keys = ("mean_turn", "mean_curv", "mean_torsion", "mean_abs_torsion", "var_turn")
    return tuple(d[k] for k in keys)


def _euclid(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5


def run(tau: int = 1, seed: int = 0) -> dict:
    import numpy as np
    from mnist1d.data import get_dataset, get_dataset_args, get_templates
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    templates = get_templates()
    class_ids = [int(c) for c in templates["y"]]
    geoms = {c: template_geometry(list(templates["x"][c]), tau=tau) for c in class_ids}
    geom_vecs = {c: _geom_vector(geoms[c]) for c in class_ids}
    raw_vecs = {c: tuple(templates["x"][c]) for c in class_ids}

    args = get_dataset_args()
    data = get_dataset(args, path=str(MNIST1D_REPO / "mnist1d_data.pkl"), download=True, verbose=False)

    scaler = StandardScaler().fit(data["x"])
    clf = LogisticRegression(max_iter=2000, random_state=seed)
    clf.fit(scaler.transform(data["x"]), data["y"])
    pred = clf.predict(scaler.transform(data["x_test"]))

    n_classes = len(class_ids)
    confusion = np.zeros((n_classes, n_classes), dtype=float)
    for true_y, pred_y in zip(data["y_test"], pred):
        confusion[true_y, pred_y] += 1
    row_sums = confusion.sum(axis=1, keepdims=True)
    confusion_rate = np.divide(confusion, row_sums, out=np.zeros_like(confusion), where=row_sums > 0)
    accuracy = float((pred == data["y_test"]).mean())

    pairs = [(i, j) for i in range(n_classes) for j in range(n_classes) if i != j]
    off_diag_confusion = [confusion_rate[i, j] + confusion_rate[j, i] for i, j in pairs]
    geom_dist = [_euclid(geom_vecs[i], geom_vecs[j]) for i, j in pairs]
    raw_dist = [_euclid(raw_vecs[i], raw_vecs[j]) for i, j in pairs]

    def spearman(xs: list[float], ys: list[float]) -> float:
        def rank(vals: list[float]) -> list[float]:
            order = sorted(range(len(vals)), key=lambda i: vals[i])
            ranks = [0.0] * len(vals)
            for r, i in enumerate(order):
                ranks[i] = float(r)
            return ranks

        rx, ry = rank(xs), rank(ys)
        n = len(xs)
        mx, my = sum(rx) / n, sum(ry) / n
        cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
        vx = sum((a - mx) ** 2 for a in rx) ** 0.5
        vy = sum((b - my) ** 2 for b in ry) ** 0.5
        return cov / (vx * vy) if vx > 0 and vy > 0 else 0.0

    # geometry predicts confusion inversely: closer shape -> more confusion
    geom_corr = spearman([-d for d in geom_dist], off_diag_confusion)
    raw_corr = spearman([-d for d in raw_dist], off_diag_confusion)

    ranked_pairs = sorted(zip(pairs, off_diag_confusion, geom_dist, raw_dist), key=lambda t: -t[1])

    # Frenet invariants are even in curvature and odd (chirality-sensitive) in
    # torsion, so an exact amplitude-mirror pair (x -> -x) should show identical
    # curvature/turn stats and torsion of matching magnitude but opposite sign.
    # Surface any such pairs -- a correctness signal for the embedding, and a
    # real structural fact about the template set, independent of the
    # confusion-prediction question above.
    mirror_pairs = []
    for i, j in pairs:
        if i >= j:
            continue
        gi, gj = geoms[i], geoms[j]
        same_shape = (
            abs(gi.mean_turn - gj.mean_turn) < 1e-6
            and abs(gi.mean_curv - gj.mean_curv) < 1e-6
            and abs(gi.mean_abs_torsion - gj.mean_abs_torsion) < 1e-6
        )
        if same_shape:
            mirror_pairs.append((i, j, gi.mean_torsion, gj.mean_torsion))

    return {
        "tau": tau,
        "classifier_test_accuracy": accuracy,
        "n_confusable_pairs": sum(1 for c in off_diag_confusion if c > 0),
        "spearman_geom_dist_vs_confusion": geom_corr,
        "spearman_raw_dist_vs_confusion": raw_corr,
        "top_confused_pairs": [
            {"classes": p, "confusion_rate": c, "geom_dist": g, "raw_dist": r}
            for p, c, g, r in ranked_pairs[:8]
        ],
        "mirror_symmetric_template_pairs": [
            {"classes": (i, j), "torsion_i": ti, "torsion_j": tj} for i, j, ti, tj in mirror_pairs
        ],
        "template_geometry": {c: asdict(geoms[c]) for c in class_ids},
    }


def main() -> None:
    import json

    result = run()
    print(f"classifier test accuracy: {result['classifier_test_accuracy']:.3f}")
    print(f"confusable class pairs (>0 confusion): {result['n_confusable_pairs']} / 90\n")
    print(f"spearman(geom_dist, confusion)  = {result['spearman_geom_dist_vs_confusion']:+.3f}  (closer shape -> more confusion, expect negative corr with distance)")
    print(f"spearman(raw_dist,  confusion)  = {result['spearman_raw_dist_vs_confusion']:+.3f}  (baseline: raw amplitude distance)\n")
    print("most-confused class pairs:")
    for row in result["top_confused_pairs"]:
        a, b = row["classes"]
        print(f"  {a} <-> {b}  confusion_rate={row['confusion_rate']:.3f}  geom_dist={row['geom_dist']:.3f}  raw_dist={row['raw_dist']:.3f}")

    if result["mirror_symmetric_template_pairs"]:
        print("\nexact amplitude-mirror template pairs found (curvature matches, torsion is +/- symmetric):")
        for row in result["mirror_symmetric_template_pairs"]:
            a, b = row["classes"]
            print(f"  {a} <-> {b}  torsion: {row['torsion_i']:+.3f} / {row['torsion_j']:+.3f}")

    out_path = Path("runs/mnist1d-lens")
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {out_path / 'result.json'}")


if __name__ == "__main__":
    main()
