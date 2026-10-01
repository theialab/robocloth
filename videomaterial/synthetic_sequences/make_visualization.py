#!/usr/bin/env python3
"""Compose [rendered frame | hemisphere inset] per frame, encode an MP4 and a contact sheet.

The inset is the equal-area disk (r = sqrt(2) sin(theta/2)) of the moving element's hemisphere:
full path (colour = time), current pose ●, mirror direction ★, fixed element ◎, polar-range circle,
theta rings at 15/30/45/60/75 deg, and phi ticks. Reads only metadata.json + frames.jsonl + PNGs.
Joint (B3, camera+light) sequences show BOTH paths on one disk: light path and current light ●
(warm), camera path and current camera ■ (cool), and the mirror direction of the current light ★
(it moves with the light).

``--rows-contact`` additionally writes ``<name>_rows_contact.png``: frames 0, 10, ..., 80 with the
material render in the top row and each twin (``<twin>_png``) in a row below. ``--out-dir`` writes
everything outside the sequence directory (default: inside it, as before).
"""
from __future__ import annotations
import argparse, json, math, shutil, subprocess, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from synthetic_sequences import trajectories as T  # noqa: E402


def _disk_axes(size_px):
    dpi = 100; fig = plt.figure(figsize=(size_px / dpi, size_px / dpi), dpi=dpi)
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.96]); ax.set_aspect("equal"); ax.axis("off")
    R = T.disk_radius(90.0); ax.set_xlim(-R * 1.08, R * 1.08); ax.set_ylim(-R * 1.08, R * 1.08)
    for t_ in (15, 30, 45, 60, 75):
        ax.add_patch(plt.Circle((0, 0), T.disk_radius(t_), fill=False, lw=0.6, color="#9AA1B2", ls=":"))
    ax.add_patch(plt.Circle((0, 0), R, fill=False, lw=1.2, color="#5A6070"))
    for a in range(0, 360, 90):
        x, y = T.ang_to_disk(90.0, a); ax.text(x * 1.05, y * 1.05, f"φ{a}°", fontsize=7, ha="center", va="center", color="#5A6070")
    ax.text(0, T.disk_radius(90) * 0.93, "θ=90°", fontsize=6, ha="center", color="#9AA1B2")
    return fig, ax, R


def _fig_to_image(fig):
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy(); plt.close(fig)
    return Image.fromarray(img)


def joint_inset_figure(meta, recs, i, size_px):
    """B3: light AND camera paths on one equal-area disk, current poses, the moving mirror direction."""
    tr = meta["trajectory"]
    lth = np.array([r["light"]["theta_deg"] for r in recs]); lph = np.array([r["light"]["phi_deg"] for r in recs])
    cth = np.array([r["camera"]["theta_deg"] for r in recs]); cph = np.array([r["camera"]["phi_deg"] for r in recs])
    PL, PC = T.ang_to_disk(lth, lph), T.ang_to_disk(cth, cph)
    fig, ax, R = _disk_axes(size_px)
    tmax = max(tr["polar_range_deg"]["light"][1], tr["polar_range_deg"]["camera"][1])
    ax.add_patch(plt.Circle((0, 0), T.disk_radius(tmax), fill=False, lw=1.0, color="#4A56C9", alpha=0.7))
    ax.plot(PL[:, 0], PL[:, 1], "-", lw=1.4, color="#E3A869", alpha=0.8, zorder=1)
    ax.plot(PC[:, 0], PC[:, 1], "-", lw=1.4, color="#6C8FD1", alpha=0.8, zorder=1)
    ax.scatter(PL[:, 0], PL[:, 1], c=np.linspace(0, 1, len(PL)), cmap="Oranges", vmin=-0.4, vmax=1.0, s=6, zorder=2)
    ax.scatter(PC[:, 0], PC[:, 1], c=np.linspace(0, 1, len(PC)), cmap="Blues", vmin=-0.4, vmax=1.0, s=6, zorder=2)
    ax.plot(*PL[0], marker="o", ms=5, mfc="none", mec="#B4661A", mew=1.2, zorder=3)
    ax.plot(*PC[0], marker="s", ms=5, mfc="none", mec="#2D55A8", mew=1.2, zorder=3)
    mth, mph = T.mirror_direction(lth[i], lph[i]); mx, my = T.ang_to_disk(mth, mph)
    ax.plot(mx, my, marker="*", ms=13, color="#A23B57", mec="k", mew=0.4, zorder=4)
    ax.plot(PL[i, 0], PL[i, 1], marker="o", ms=10, color="#F0A030", mec="k", mew=0.8, zorder=5)
    ax.plot(PC[i, 0], PC[i, 1], marker="s", ms=9, color="#3A6FD8", mec="k", mew=0.8, zorder=5)
    lab = (f"{tr['group']}  ·  light {tr['light_class']} · camera {tr['camera_class']}\n"
           f"frame {recs[i]['index']:02d}/{tr['frames']-1}\n"
           f"light  θ {lth[i]:5.1f}°  φ {lph[i]:6.1f}°\ncamera θ {cth[i]:5.1f}°  φ {cph[i]:6.1f}°")
    ax.text(-R * 1.05, -R * 1.02, lab, fontsize=7, ha="left", va="bottom", color="#1B1E24", family="monospace")
    ax.text(R * 1.05, R * 1.04, "● light  ■ camera\n★ mirror of light\n○□ start", fontsize=7, ha="right", va="top", color="#5A6070")
    return _fig_to_image(fig)


def inset_figure(meta, recs, i, size_px):
    tr = meta["trajectory"]; moving = tr["moving_element"]
    if moving == "camera+light":
        return joint_inset_figure(meta, recs, i, size_px)
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


def project(meta, rec, point):
    """Pixel of a world point under the metadata camera (convention of check_metadata.py). None if behind."""
    K = np.array(meta["camera"]["intrinsics_K"]); c2w = np.array(rec["camera"]["c2w"], dtype=np.float64)
    w2c = np.linalg.inv(c2w); pc = w2c[:3, :3] @ np.asarray(point, dtype=np.float64) + w2c[:3, 3]
    if pc[2] <= 1e-6:
        return None
    return (K[0, 2] - K[0, 0] * pc[0] / pc[2], K[1, 2] - K[1, 1] * pc[1] / pc[2])


def draw_light_overlay(img, meta, rec):
    """Yellow ring at the light's image position; if it is outside the frame, an arrow on the border."""
    d = ImageDraw.Draw(img); W, H = img.size
    uv = project(meta, rec, rec["light"]["position"])
    if uv is None:
        return img
    u, v = uv
    if 0 <= u < W and 0 <= v < H:
        r = 9; d.ellipse([u - r, v - r, u + r, v + r], outline=(255, 210, 60), width=3)
        d.text((u + 12, v - 8), "light", fill=(255, 210, 60))
    else:  # arrow from the centre towards the off-screen light, clipped to the border
        cx, cy = W / 2, H / 2; dx, dy = u - cx, v - cy
        t = min((W / 2 - 20) / abs(dx) if dx else 9e9, (H / 2 - 20) / abs(dy) if dy else 9e9)
        ex, ey = cx + dx * t, cy + dy * t
        d.line([ex - dx * t * 0.12, ey - dy * t * 0.12, ex, ey], fill=(255, 210, 60), width=3)
        d.ellipse([ex - 6, ey - 6, ex + 6, ey + 6], fill=(255, 210, 60))
        d.text((min(max(ex - 30, 4), W - 70), min(max(ey - 22, 4), H - 16)), "light ↗", fill=(255, 210, 60))
    return img


def ffmpeg_exe():
    """System ffmpeg, else the binary bundled with imageio-ffmpeg (the Leonardo render env has only that)."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as e:
        raise RuntimeError("no ffmpeg on PATH and imageio_ffmpeg is not installed") from e


def rows_contact_sheet(d, recs, out_path, every=10, scale=0.5):
    """Frames 0, every, 2*every, ...: material row on top, one row per twin (<twin>_png) below."""
    pick = [r for r in recs if r["index"] % every == 0]
    keys = ["png"] + sorted(k for k in recs[0]["files"] if k.endswith("_png") and k != "png" and recs[0]["files"][k])
    first = Image.open(d / recs[0]["files"]["png"]); cw, ch = int(first.width * scale), int(first.height * scale)
    lab_w, head = 120, 22
    sheet = Image.new("RGB", (lab_w + len(pick) * cw, head + len(keys) * ch), (252, 252, 250))
    dr_ = ImageDraw.Draw(sheet)
    for j, r in enumerate(pick):
        dr_.text((lab_w + j * cw + 6, 5), f"frame {r['index']:02d}", fill=(27, 30, 36))
    for i, k in enumerate(keys):
        dr_.text((6, head + i * ch + ch // 2 - 6), "material 314" if k == "png" else k.replace("_png", ""), fill=(27, 30, 36))
        for j, r in enumerate(pick):
            sheet.paste(Image.open(d / r["files"][k]).convert("RGB").resize((cw, ch)), (lab_w + j * cw, head + i * ch))
    sheet.save(out_path)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sequence_dir"); ap.add_argument("--fps", type=int, default=15); ap.add_argument("--crf", type=int, default=18)
    ap.add_argument("--contact", type=int, default=9)
    ap.add_argument("--out-dir", default=None, help="write composites / mp4 / sheets here instead of the sequence dir")
    ap.add_argument("--name", default=None, help="file name stem (default: the sequence dir name)")
    ap.add_argument("--rows-contact", action="store_true", help="also write <name>_rows_contact.png (material row + twin rows)")
    ap.add_argument("--keep-composites", action="store_true", help="with --out-dir: keep the per-frame composite PNGs")
    a = ap.parse_args()
    d = Path(a.sequence_dir); meta = json.loads((d / "metadata.json").read_text())
    recs = [json.loads(l) for l in (d / "frames.jsonl").read_text().splitlines() if l.strip()]
    recs.sort(key=lambda r: r["index"])
    od = d if a.out_dir is None else Path(a.out_dir); od.mkdir(parents=True, exist_ok=True)
    stem = a.name or d.name
    if a.rows_contact:
        print(f"wrote {rows_contact_sheet(d, recs, od / f'{stem}_rows_contact.png')}")
    comp_dir = (d / "composite_png") if a.out_dir is None else (od / f"{stem}_composite_png"); comp_dir.mkdir(exist_ok=True)
    H = meta["camera"]["resolution"][1]
    frames = []
    for i, r in enumerate(recs):
        left = Image.open(d / r["files"]["png"]).convert("RGB")
        if meta.get("mode") == "ball":
            left = draw_light_overlay(left, meta, r)
        right = inset_figure(meta, recs, i, H)
        canvas = Image.new("RGB", (left.width + right.width, H), (252, 252, 250))
        canvas.paste(left, (0, 0)); canvas.paste(right, (left.width, 0))
        canvas.save(comp_dir / f"frame_{r['index']:03d}.png"); frames.append(canvas)
    name = f"{stem}_visualization.mp4"
    subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", "-framerate", str(a.fps), "-pattern_type", "glob", "-i", str(comp_dir / "frame_*.png"),
                    "-c:v", "libx264", "-preset", "slow", "-crf", str(a.crf), "-pix_fmt", "yuv420p",
                    "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-movflags", "+faststart", str(od / name)], check=True)
    # contact sheet: a.contact frames evenly spaced, 3 per row
    idx = np.linspace(0, len(frames) - 1, a.contact).round().astype(int)
    w, h = frames[0].size; s = 0.5; cw, ch = int(w * s), int(h * s); cols = 3; rows = math.ceil(len(idx) / cols)
    sheet = Image.new("RGB", (cols * cw, rows * ch), (252, 252, 250))
    for k, j in enumerate(idx):
        sheet.paste(frames[j].resize((cw, ch)), ((k % cols) * cw, (k // cols) * ch))
    sheet_path = (d / "contact_sheet.png") if a.out_dir is None else (od / f"{stem}_contact_sheet.png")
    sheet.save(sheet_path)
    if a.out_dir is not None and not a.keep_composites:
        shutil.rmtree(comp_dir, ignore_errors=True)
    print(f"wrote {od / name} and {sheet_path.name} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
