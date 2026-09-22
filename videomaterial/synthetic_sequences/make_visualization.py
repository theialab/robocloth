#!/usr/bin/env python3
"""Compose [rendered frame | hemisphere inset] per frame, encode an MP4 and a contact sheet.

The inset is the equal-area disk (r = sqrt(2) sin(theta/2)) of the moving element's hemisphere:
full path (colour = time), current pose ●, mirror direction ★, fixed element ◎, polar-range circle,
theta rings at 15/30/45/60/75 deg, and phi ticks. Reads only metadata.json + frames.jsonl + PNGs.
"""
from __future__ import annotations
import argparse, json, math, subprocess, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from synthetic_sequences import trajectories as T  # noqa: E402


def inset_figure(meta, recs, i, size_px):
    tr = meta["trajectory"]; moving = tr["moving_element"]
    key = "light" if moving == "light" else "camera"
    th = np.array([r[key]["theta_deg"] for r in recs]); ph = np.array([r[key]["phi_deg"] for r in recs])
    P = T.ang_to_disk(th, ph)
    fixed = tr["fixed_element_deg"]; mirror = tr["mirror_direction_deg"]
    tmin, tmax = tr["polar_range_deg"]
    dpi = 100; fig = plt.figure(figsize=(size_px / dpi, size_px / dpi), dpi=dpi)
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.96]); ax.set_aspect("equal"); ax.axis("off")
    R = T.disk_radius(90.0); ax.set_xlim(-R * 1.08, R * 1.08); ax.set_ylim(-R * 1.08, R * 1.08)
    for t_ in (15, 30, 45, 60, 75):
        c = plt.Circle((0, 0), T.disk_radius(t_), fill=False, lw=0.6, color="#9AA1B2", ls=":")
        ax.add_patch(c)
    ax.add_patch(plt.Circle((0, 0), R, fill=False, lw=1.2, color="#5A6070"))
    ax.add_patch(plt.Circle((0, 0), T.disk_radius(tmax), fill=False, lw=1.0, color="#4A56C9", alpha=0.7))
    for a in range(0, 360, 90):
        x, y = T.ang_to_disk(90.0, a); ax.text(x * 1.05, y * 1.05, f"φ{a}°", fontsize=7, ha="center", va="center", color="#5A6070")
    ax.text(0, T.disk_radius(90) * 0.93, "θ=90°", fontsize=6, ha="center", color="#9AA1B2")
    ax.plot(P[:, 0], P[:, 1], "-", lw=1.0, color="#B0B5C4", zorder=1)
    ax.scatter(P[:, 0], P[:, 1], c=np.linspace(0, 1, len(P)), cmap="viridis", s=7, zorder=2)
    fx, fy = T.ang_to_disk(fixed["theta"], fixed["phi"]); mx, my = T.ang_to_disk(mirror["theta"], mirror["phi"])
    ax.plot(fx, fy, marker="o", ms=11, mfc="none", mec="#0E7C6B", mew=2, zorder=3)
    ax.plot(mx, my, marker="*", ms=13, color="#A23B57", mec="k", mew=0.4, zorder=3)
    ax.plot(P[i, 0], P[i, 1], marker="o", ms=10, color="#E8A04C", mec="k", mew=0.8, zorder=4)
    lab = f"{tr['class']}  ·  moving: {moving}\nframe {recs[i]['index']:02d}/{tr['frames']-1}   θ {th[i]:5.1f}°  φ {ph[i]:6.1f}°"
    ax.text(-R * 1.05, -R * 1.02, lab, fontsize=8, ha="left", va="bottom", color="#1B1E24", family="monospace")
    ax.text(R * 1.05, -R * 1.02, "◎ fixed  ★ mirror  ● now", fontsize=7, ha="right", va="bottom", color="#5A6070")
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy(); plt.close(fig)
    return Image.fromarray(img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sequence_dir"); ap.add_argument("--fps", type=int, default=15); ap.add_argument("--crf", type=int, default=18)
    ap.add_argument("--contact", type=int, default=9)
    a = ap.parse_args()
    d = Path(a.sequence_dir); meta = json.loads((d / "metadata.json").read_text())
    recs = [json.loads(l) for l in (d / "frames.jsonl").read_text().splitlines() if l.strip()]
    recs.sort(key=lambda r: r["index"])
    comp_dir = d / "composite_png"; comp_dir.mkdir(exist_ok=True)
    H = meta["camera"]["resolution"][1]
    frames = []
    for i, r in enumerate(recs):
        left = Image.open(d / r["files"]["png"]).convert("RGB")
        right = inset_figure(meta, recs, i, H)
        canvas = Image.new("RGB", (left.width + right.width, H), (252, 252, 250))
        canvas.paste(left, (0, 0)); canvas.paste(right, (left.width, 0))
        canvas.save(comp_dir / f"frame_{r['index']:03d}.png"); frames.append(canvas)
    name = f"{d.name}_visualization.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(a.fps), "-i", str(comp_dir / "frame_%03d.png"),
                    "-c:v", "libx264", "-preset", "slow", "-crf", str(a.crf), "-pix_fmt", "yuv420p",
                    "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-movflags", "+faststart", str(d / name)], check=True)
    # contact sheet: a.contact frames evenly spaced, 3 per row
    idx = np.linspace(0, len(frames) - 1, a.contact).round().astype(int)
    w, h = frames[0].size; s = 0.5; cw, ch = int(w * s), int(h * s); cols = 3; rows = math.ceil(len(idx) / cols)
    sheet = Image.new("RGB", (cols * cw, rows * ch), (252, 252, 250))
    for k, j in enumerate(idx):
        sheet.paste(frames[j].resize((cw, ch)), ((k % cols) * cw, (k // cols) * ch))
    sheet.save(d / "contact_sheet.png")
    print(f"wrote {d / name} and contact_sheet.png ({len(frames)} frames)")


if __name__ == "__main__":
    main()
