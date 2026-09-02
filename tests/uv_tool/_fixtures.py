"""Tiny box-mesh fixtures for the generate_uv tests.

The box spans x in [0, 1], y in [0, 2], z in [0, 3] -- deliberately asymmetric
so that a wrong importer/exporter axis conversion shows up as moved corners.
write_obj() / write_ply() need only the standard library. Run this file under
Blender or the bpy wheel to also produce the GLB / glTF variants:

    blender --background --python tests/uv_tool/_fixtures.py -- <out_dir>
    python tests/uv_tool/_fixtures.py <out_dir>

The glTF files are authored so that the coordinates *stored in the file* are
the canonical corners (Blender's exporter rotates Z-up to Y-up, so the mesh is
built pre-rotated).
"""
import os
import sys

CORNERS = [(float(x), float(y), float(z)) for x in (0, 1) for y in (0, 2) for z in (0, 3)]
# Quad faces; corner index = 4*ix + 2*iy + iz.
FACES = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]


def write_obj(path):
    """ASCII OBJ, positions + quad faces only (no normals, no UVs)."""
    with open(path, "w") as f:
        f.write("# box fixture\n")
        for v in CORNERS:
            f.write("v %g %g %g\n" % v)
        for face in FACES:
            f.write("f " + " ".join(str(i + 1) for i in face) + "\n")
    return path


def write_ply(path):
    """ASCII PLY, positions + quad faces only (no normals, no UVs)."""
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\ncomment box fixture\n")
        f.write("element vertex %d\nproperty float x\nproperty float y\nproperty float z\n" % len(CORNERS))
        f.write("element face %d\nproperty list uchar int vertex_indices\nend_header\n" % len(FACES))
        for v in CORNERS:
            f.write("%g %g %g\n" % v)
        for face in FACES:
            f.write("%d " % len(face) + " ".join(str(i) for i in face) + "\n")
    return path


def write_gltf_fixtures(out_dir):
    """Needs bpy. Writes box.glb and box_separate.gltf (+ .bin); returns both paths."""
    import bpy
    bpy.ops.wm.read_factory_settings(use_empty=True)
    mesh = bpy.data.meshes.new("box")
    # Blender -> glTF export maps (x, y, z) to (x, z, -y); pre-apply the inverse.
    mesh.from_pydata([(x, -z, y) for (x, y, z) in CORNERS], [], FACES)
    mesh.update()
    obj = bpy.data.objects.new("box", mesh)
    bpy.context.scene.collection.objects.link(obj)
    glb = os.path.join(out_dir, "box.glb")
    gltf = os.path.join(out_dir, "box_separate.gltf")
    bpy.ops.export_scene.gltf(filepath=glb, export_format="GLB")
    bpy.ops.export_scene.gltf(filepath=gltf, export_format="GLTF_SEPARATE")
    return glb, gltf


def main(argv):
    args = argv[argv.index("--") + 1:] if "--" in argv else argv[1:]
    if len(args) != 1:
        print("usage: _fixtures.py <out_dir>   (under Blender: ... --python _fixtures.py -- <out_dir>)")
        return 2
    out_dir = os.path.abspath(args[0])
    os.makedirs(out_dir, exist_ok=True)
    write_obj(os.path.join(out_dir, "box.obj"))
    write_ply(os.path.join(out_dir, "box.ply"))
    try:
        write_gltf_fixtures(out_dir)
    except Exception as exc:  # Blender exits 0 on uncaught exceptions; report explicitly
        print(f"FIXTURES_ERROR {exc!r}", flush=True)
        return 1
    print("FIXTURES_OK", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
