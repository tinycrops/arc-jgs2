"""Animations of the corpus-cylinder / wave-room results.

Four animations, each visualizing a claim already measured statically
elsewhere in this repo -- nothing here is a new experiment:

1. corpus_disk_spin.gif -- the rotation orbit made continuous. The 10
   discrete color-invariant formations (README, tests/test_corpusfield.py)
   are the frames at every 36 degrees of a single continuous rigid spin of
   the corpus disk.
2. wave_room_spin.gif -- the interference pattern from analyze_wave_field.py
   rotating. u(x) is a function of point positions only, so rotating every
   source by theta rotates the whole field by theta -- exactly (not just at
   the 10 color-cyclic angles; that symmetry holds for ANY continuous theta).
   Rendered by rotating the precomputed magnitude image, not resimulating
   316k sources per frame.
3. wave_room_buildup.gif -- sources switch on in RADIUS order (corpus
   position order), the literal "light from a point source, run backward"
   framing from the original request. Built via a cumulative FFT convolution
   at reduced resolution so each frame is cheap.
4. wave_k_sweep.gif -- one real ARC grid's far field as the wavenumber k
   sweeps from long to short wavelength: the "dial" from the wave-room
   section of the README, showing fringe spacing shrink as k grows.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import rotate as nd_rotate

from arc_jgs2.corpusfield import corpus_slots, disk_points, grid_slots
from arc_jgs2.loaders import load_tasks

DATA = Path("data/ARC-AGI/data/training")
OUT = Path("runs/corpus-field")

ARC_COLORS = [
    "#111111", "#1E93FF", "#F93C31", "#4FCC30", "#FFDC00",
    "#999999", "#E53AA3", "#FF851B", "#87D8F1", "#921231",
]
ARC_RGB = np.array([[int(h[i : i + 2], 16) for i in (1, 3, 5)] for h in ARC_COLORS], dtype=np.uint8)
SURFACE_RGB = np.array([255, 255, 255], dtype=np.uint8)


def _save_gif(frames: list[np.ndarray], path: Path, fps: int) -> None:
    """Quantize every frame to ONE shared palette with no dithering. Smooth
    gradients (our wave fields) blow up under per-frame dithered palettes;
    a shared palette + flat quantization cuts file size by an order of
    magnitude with no visible loss on this content."""
    images = [Image.fromarray(f) for f in frames]
    palette_src = images[len(images) // 2].quantize(colors=200, method=Image.MEDIANCUT)
    frames_p = [im.quantize(palette=palette_src, dither=Image.Dither.NONE) for im in images]
    frames_p[0].save(
        path, save_all=True, append_images=frames_p[1:], duration=int(1000 / fps), loop=0, optimize=True
    )
    print(f"wrote {path}  ({len(frames)} frames, {path.stat().st_size / 1e6:.1f} MB)")


# --- 1. corpus disk spin -----------------------------------------------------


def render_disk_spin(n_frames: int = 90, size: int = 640, fps: int = 24) -> None:
    tasks = load_tasks(DATA)
    slots, _ = corpus_slots(tasks)
    n = len(slots)
    idx = np.arange(n)
    r = (n - 1 - idx) / (n - 1)
    ang0 = np.radians(-90 + (np.asarray(slots) % 10) * 36)
    x0, y0 = r * np.cos(ang0), r * np.sin(ang0)
    colors = ARC_RGB[np.asarray(slots) % 10]

    half = 1.05
    edges = np.linspace(-half, half, size + 1)

    frames = []
    for i, theta in enumerate(np.linspace(0, 360, n_frames, endpoint=False)):
        t = math.radians(theta)
        x = x0 * math.cos(t) - y0 * math.sin(t)
        y = x0 * math.sin(t) + y0 * math.cos(t)
        gx = np.clip(np.digitize(x, edges) - 1, 0, size - 1)
        gy = np.clip(np.digitize(y, edges) - 1, 0, size - 1)

        img = np.full((size, size, 3), 255, dtype=np.uint8)
        # paint back-to-front by radius so near-origin (later, denser) wins ties visually
        order = np.argsort(r)
        img[gy[order], gx[order]] = colors[order]
        frames.append(np.flipud(img))
        if i % 15 == 0:
            print(f"  disk_spin frame {i}/{n_frames}")

    _save_gif(frames, OUT / "corpus_disk_spin.gif", fps)


# --- 2. wave room spin --------------------------------------------------------


def render_wave_spin(n_frames: int = 72, fps: int = 20, downsample: int = 2) -> None:
    data = np.load(OUT / "wave_room.npz")
    mag = data["mag0"].astype(np.float64)[::downsample, ::downsample]
    lo, hi = np.quantile(mag, 0.02), np.quantile(mag, 0.998)
    log_mag = np.log1p(mag)
    log_lo, log_hi = math.log1p(lo), math.log1p(hi)

    # simple one-hue sequential colormap sampled by hand (avoids a Normalize/
    # cmap object per frame): white -> mid blue -> near-black blue
    stops = np.array(
        [[255, 255, 255], [205, 226, 251], [134, 182, 239], [57, 135, 229], [28, 92, 171], [13, 47, 92]],
        dtype=np.float64,
    )

    def colorize(a: np.ndarray) -> np.ndarray:
        t = np.clip((a - log_lo) / (log_hi - log_lo), 0, 1) * (len(stops) - 1)
        i0 = np.clip(t.astype(int), 0, len(stops) - 2)
        frac = (t - i0)[..., None]
        return (stops[i0] * (1 - frac) + stops[i0 + 1] * frac).astype(np.uint8)

    frames = []
    for i, theta in enumerate(np.linspace(0, 360, n_frames, endpoint=False)):
        rotated = nd_rotate(log_mag, theta, reshape=False, order=1, cval=log_mag.min())
        frames.append(colorize(rotated))
        if i % 15 == 0:
            print(f"  wave_spin frame {i}/{n_frames}")

    _save_gif(frames, OUT / "wave_room_spin.gif", fps)


# --- 3. wave room build-up ----------------------------------------------------


def render_wave_buildup(n_frames: int = 60, room_n: int = 512, fps: int = 15) -> None:
    tasks = load_tasks(DATA)
    slots, _ = corpus_slots(tasks)
    n = len(slots)

    # disk coords at rotation 0; index order already IS radius order (position
    # 0 -> rim, n-1 -> origin), so revealing a growing PREFIX is exactly the
    # "light from a point source, run backward" propagation from the request
    pts = np.asarray(disk_points(slots))
    x, y = pts[:, 0], pts[:, 1]

    wavelen = 0.08
    k_room = 2 * math.pi / wavelen
    half = 1.4
    m = 2 * room_n
    dx = 2 * half / room_n
    edges = np.linspace(-half, half, room_n + 1)

    freq_idx = np.fft.fftfreq(m, d=1.0) * m
    ox, oy = np.meshgrid(freq_idx * dx, freq_idx * dx, indexing="ij")
    rho = np.hypot(ox, oy)
    kernel_fft = np.fft.fft2(np.exp(1j * k_room * rho) / np.sqrt(rho + 0.5 * dx))

    # cumulative source histogram, revealed in `n_frames` equal index chunks
    bounds = np.linspace(0, n, n_frames + 1).astype(int)
    cum_hist = np.zeros((room_n, room_n))
    all_mags = []
    frames_raw = []
    for i in range(n_frames):
        lo_i, hi_i = bounds[i], bounds[i + 1]
        chunk, _, _ = np.histogram2d(x[lo_i:hi_i], y[lo_i:hi_i], bins=[edges, edges])
        cum_hist += chunk
        pad = np.zeros((m, m), dtype=complex)
        pad[:room_n, :room_n] = cum_hist
        u = np.fft.ifft2(np.fft.fft2(pad) * kernel_fft)[:room_n, :room_n]
        mag = np.log1p(np.abs(u))
        frames_raw.append(mag)
        all_mags.append(mag)
        if i % 10 == 0:
            print(f"  wave_buildup frame {i}/{n_frames}  ({hi_i}/{n} sources on)")

    lo_v, hi_v = np.quantile(all_mags[-1], 0.02), np.quantile(all_mags[-1], 0.998)
    stops = np.array(
        [[255, 255, 255], [205, 226, 251], [134, 182, 239], [57, 135, 229], [28, 92, 171], [13, 47, 92]],
        dtype=np.float64,
    )

    def colorize(a: np.ndarray) -> np.ndarray:
        t = np.clip((a - lo_v) / (hi_v - lo_v), 0, 1) * (len(stops) - 1)
        i0 = np.clip(t.astype(int), 0, len(stops) - 2)
        frac = (t - i0)[..., None]
        return (stops[i0] * (1 - frac) + stops[i0 + 1] * frac).astype(np.uint8)

    frames = [np.flipud(colorize(m)) for m in frames_raw]
    # hold the final frame a bit longer so the completed room is visible
    frames += [frames[-1]] * (fps)
    _save_gif(frames, OUT / "wave_room_buildup.gif", fps)


# --- 4. single-grid wavenumber sweep ------------------------------------------


def render_grid_ksweep(task_id: str = "007bbfb7", n_frames: int = 48, room_n: int = 360, fps: int = 16) -> None:
    tasks = load_tasks(DATA)
    task = next(t for t in tasks if t.task_id == task_id)
    grid = task.train[0].input
    slots = grid_slots(grid)
    n = len(slots)
    idx = np.arange(n)
    r = (n - 1 - idx) / (n - 1) if n > 1 else np.zeros(1)
    ang = np.radians(-90 + (np.asarray(slots) % 10) * 36)
    x, y = r * np.cos(ang), r * np.sin(ang)

    half = 1.3
    px = np.linspace(-half, half, room_n)
    gx, gy = np.meshgrid(px, px, indexing="ij")

    ks = np.geomspace(2 * math.pi * 1.5, 2 * math.pi * 40, n_frames)
    frames = []
    for i, k in enumerate(ks):
        u = np.zeros((room_n, room_n), dtype=complex)
        for xj, yj in zip(x, y):
            rho = np.hypot(gx - xj, gy - yj)
            u += np.exp(1j * k * rho) / np.sqrt(rho + 1e-3)
        mag = np.log1p(np.abs(u))
        lo_v, hi_v = np.quantile(mag, 0.02), np.quantile(mag, 0.998)
        t = np.clip((mag - lo_v) / (hi_v - lo_v), 0, 1)
        gray = 255.0 * (1 - t)  # float; must clip/cast AFTER the +40 offset, not before
        rgb = np.stack([gray, gray, np.clip(gray + 40, 0, 255)], axis=-1).astype(np.uint8)
        frames.append(np.flipud(rgb))
        if i % 10 == 0:
            print(f"  grid_ksweep frame {i}/{n_frames}  k={k:.1f}")

    frames = frames + frames[::-1]  # ping-pong: long wavelength <-> short, looping
    _save_gif(frames, OUT / f"wave_k_sweep_{task_id}.gif", fps)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    render_disk_spin()
    render_wave_spin()
    render_wave_buildup()
    render_grid_ksweep()


if __name__ == "__main__":
    main()
