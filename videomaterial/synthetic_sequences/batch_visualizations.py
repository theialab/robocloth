#!/usr/bin/env python3
"""Render the exp-015 trajectory visualisations: 2 clips per class for B1 and B2 (ball scene), compose
videos, write index.html + all_classes_contact_sheet.png per setup; optionally the material-314 clips."""
from __future__ import annotations
import argparse, json, subprocess, sys, time
from pathlib import Path
from PIL import Image

HERE = Path(__file__).resolve().parent
CLASSES = {"spline": (11, 12), "highlight_sweep": (21, 22), "ring": (31, 32), "spiral": (41, 42), "lissajous": (51, 52)}
SETUP_DIR = {"B1": "B1_Fixed_camera", "B2": "B2_Fixed_light"}


def run(cmd, log):
    with open(log, "a") as f:
        f.write("$ " + " ".join(map(str, cmd)) + "\n")
    r = subprocess.run(list(map(str, cmd)), capture_output=True, text=True)
    with open(log, "a") as f:
        f.write(r.stdout[-3000:] + r.stderr[-3000:] + "\n")
    if r.returncode != 0:
        raise RuntimeError(f"failed: {cmd}\n{r.stderr[-2000:]}")
    return r.stdout


def index_html(root: Path, setup: str):
    clips = sorted(p for p in root.iterdir() if p.is_dir() and (p / "RENDER_COMPLETE").exists())
    rows = []
    for c in clips:
        m = json.loads((c / "metadata.json").read_text()); tr = m["trajectory"]
        mp4 = next(c.glob("*_visualization.mp4"), None)
        rows.append(f"""<section><h2>{c.name}</h2>
<p>class {tr['class']} · seed {m['seeds']['trajectory']} · mode {m['mode']} · spp {m['render']['spp']} · arc {tr.get('arc_length_deg',0):.0f}° · max step {tr.get('max_step_deg',0):.2f}° · θ {tr['theta_deg_range_actual'][0]:.1f}–{tr['theta_deg_range_actual'][1]:.1f}° · φ coverage {tr['phi_coverage_deg']:.0f}°</p>
{'<video controls loop muted playsinline src="'+c.name+'/'+mp4.name+'"></video>' if mp4 else ''}
<a href="{c.name}/contact_sheet.png"><img src="{c.name}/contact_sheet.png" alt="contact sheet"></a></section>""")
    html = f"""<!doctype html><meta charset="utf-8"><title>{setup} trajectory visualisations</title>
<style>body{{font-family:system-ui,sans-serif;max-width:1400px;margin:2rem auto;padding:0 1rem;color:#1B1E24}}
video,img{{max-width:100%;display:block}} section{{margin:2rem 0;border-top:1px solid #ddd;padding-top:1rem}} h2{{font-size:1.1rem}}</style>
<h1>{setup} — {SETUP_DIR[setup]} — trajectory visualisations</h1>
<p>Same geometry as the material scene: ground at y = 0 with a glossy patch of the sample's size at the origin (roughness 0.15) so the light's reflection appears exactly where the material sample would show its specular peak; a grey ball to the side for shading and shadow; the exp-005 point light (intensity 20, r = 3). The yellow ring (or border arrow) marks the light's position. Right panel: equal-area disk of the moving element's hemisphere — full path (colour = time), ● current pose, ★ mirror direction of the fixed element, ◎ fixed element. Fixed element at θ 45°, φ 90°, r 3. Camera fov_y 35°, 832×480, 81 frames at 15 fps. Seeds and every parameter are in each clip's metadata.json (schema v2).</p>
<p><a href="all_classes_contact_sheet.png">all_classes_contact_sheet.png</a></p>
{''.join(rows)}"""
    (root / "index.html").write_text(html)
    # Markdown twin for VS Code's built-in preview (renders images and <video> tags with relative paths;
    # no HTTP server needed on the fileserver)
    md = [f"# {setup} — {SETUP_DIR[setup]} — trajectory visualisations", "",
          "Ground at y = 0 with a glossy sample-sized patch at the origin (the highlight lands where the material sample would show its specular peak) and a grey ball to the side; exp-005 point light. Yellow ring / border arrow = light position. Right panel: equal-area disk of the moving element's hemisphere — path (colour = time), ● current pose, ★ mirror direction, ◎ fixed element. Fixed element θ 45° φ 90° r 3; camera fov_y 35°, 832×480, 81 frames at 15 fps. Seeds and parameters: each clip's `metadata.json`.", "",
          "![all classes](all_classes_contact_sheet.png)", ""]
    for c in clips:
        m = json.loads((c / "metadata.json").read_text()); tr = m["trajectory"]
        mp4 = next(c.glob("*.mp4"), None)
        md += [f"## {c.name}", "",
               f"class {tr['class']} · seed {m['seeds']['trajectory']} · mode {m['mode']} · spp {m['render']['spp']} · arc {tr.get('arc_length_deg',0):.0f}° · max step {tr.get('max_step_deg',0):.2f}° · θ {tr['theta_deg_range_actual'][0]:.1f}–{tr['theta_deg_range_actual'][1]:.1f}° · φ coverage {tr['phi_coverage_deg']:.0f}°", ""]
        if mp4:
            md += [f'<video controls loop muted width="900" src="{c.name}/{mp4.name}"></video>', ""]
        if (c / "contact_sheet.png").exists():
            md += [f"![{c.name} contact sheet]({c.name}/contact_sheet.png)", ""]
    (root / "README.md").write_text("\n".join(md))
    # combined sheet: middle frame of each clip
    tiles = []
    for c in clips:
        f = sorted((c / "composite_png").glob("frame_*.png"))
        if f:
            tiles.append((c.name, Image.open(f[len(f) // 2]).convert("RGB")))
    if tiles:
        w, h = tiles[0][1].size; s = 0.4; cw, ch = int(w * s), int(h * s); cols = 2
        rows_n = (len(tiles) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * cw, rows_n * (ch + 18)), (252, 252, 250))
        from PIL import ImageDraw
        dr = ImageDraw.Draw(sheet)
        for k, (name, im) in enumerate(tiles):
            x, y = (k % cols) * cw, (k // cols) * (ch + 18)
            dr.text((x + 4, y + 2), name, fill=(27, 30, 36)); sheet.paste(im.resize((cw, ch)), (x, y + 18))
        sheet.save(root / "all_classes_contact_sheet.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path("/media/raid/cloth/VideoMaterial/data/Robocloth_synthetic_sequence"))
    ap.add_argument("--setups", default="B1,B2"); ap.add_argument("--spp", type=int, default=32)
    ap.add_argument("--material", action="store_true", help="also render spline_01 with material 314 at 64 spp")
    ap.add_argument("--only-index", action="store_true")
    a = ap.parse_args()
    py = sys.executable
    for setup in a.setups.split(","):
        root = a.data_root / SETUP_DIR[setup] / "trajectory_visualizations"; root.mkdir(parents=True, exist_ok=True)
        log = root / "batch.log"
        if not a.only_index:
            for cls, seeds in CLASSES.items():
                for n, seed in enumerate(seeds, 1):
                    out = root / f"{cls}_{n:02d}"
                    if (out / "RENDER_COMPLETE").exists() and next(out.glob("*_visualization.mp4"), None):
                        continue
                    t0 = time.time()
                    run([py, HERE / "render_sequence.py", "--setup", setup, "--cls", cls, "--seed", seed, "--mode", "ball",
                         "--spp", a.spp, "--batch-spp", 8, "--output", out, "--split", "visualization"], log)
                    run([py, HERE / "make_visualization.py", out], log)
                    print(f"{setup} {out.name} done in {time.time()-t0:.0f}s", flush=True)
            if a.material:
                out = root / "spline_01_material314"
                if not (out / "RENDER_COMPLETE").exists():
                    t0 = time.time()
                    run([py, HERE / "render_sequence.py", "--setup", setup, "--cls", "spline", "--seed", CLASSES["spline"][0], "--mode", "material",
                         "--spp", 64, "--batch-spp", 4, "--output", out, "--split", "visualization"], log)
                    run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "15", "-i", out / "frames_png" / "frame_%03d.png", "-c:v", "libx264",
                         "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out / "spline_01_material314.mp4"], log)
                    print(f"{setup} material314 done in {time.time()-t0:.0f}s", flush=True)
        index_html(root, setup)
        print(f"{setup}: index written to {root/'index.html'}", flush=True)


if __name__ == "__main__":
    main()
