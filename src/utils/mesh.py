"""Optional GLB isosurfaces from the reconstructed radiodensity field."""
import json
from pathlib import Path
import struct

import numpy as np


def export_ct_mesh(volume, output_path, spacing, origin, level=None):
    """Export a native-grid isosurface; spacing and voxel-center origin are in meters.

    The default level is half the peak radiodensity. A zero border closes surfaces
    at the CT boundary. Native XYZ (Z up) is rotated to glTF XYZ (Y up).
    The input CT is never modified, smoothed, or resampled.
    """
    from skimage.measure import marching_cubes

    output_path = Path(output_path)
    if output_path.suffix.lower() != ".glb":
        raise ValueError("Mesh output must use the .glb extension")
    volume = np.asarray(volume, dtype=np.float32)
    spacing = np.asarray(spacing, dtype=np.float64)
    origin = np.asarray(origin, dtype=np.float64)
    if volume.ndim != 3 or min(volume.shape) < 2:
        raise ValueError("Mesh export needs a 3D CT with at least two voxels per axis")
    if not np.isfinite(volume).all() or volume.min() < 0 or volume.max() <= 0:
        raise ValueError("Mesh export needs a finite, nonnegative, nonempty radiodensity field")
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or (spacing <= 0).any():
        raise ValueError("Voxel spacing must contain three positive finite values")
    if origin.shape != (3,) or not np.isfinite(origin).all():
        raise ValueError("Voxel origin must contain three finite values")
    level = float(volume.max()) / 2 if level is None else float(level)
    if not np.isfinite(level) or not 0 < level < float(volume.max()):
        raise ValueError("Mesh level must be positive and below the peak radiodensity")

    vertices, faces, normals, _ = marching_cubes(
        np.pad(volume, 1), level=level, spacing=tuple(spacing),
        gradient_direction="ascent", allow_degenerate=False,
    )
    vertices += origin - spacing  # Remove the padded voxel, keeping voxel-center coordinates.
    # marching_cubes normals are in index space, so account for anisotropic voxels.
    normals = normals / spacing
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    vertices = np.column_stack((vertices[:, 0], vertices[:, 2], -vertices[:, 1]))
    normals = np.column_stack((normals[:, 0], normals[:, 2], -normals[:, 1]))
    _write_glb(output_path, vertices, faces, normals, level)
    return {"vertices": len(vertices), "faces": len(faces), "level": level}


def _write_glb(path, vertices, faces, normals, level):
    """Write glTF 2.0 with only NumPy and the standard library."""
    arrays = [np.ascontiguousarray(vertices, dtype="<f4"),
              np.ascontiguousarray(normals, dtype="<f4"),
              np.ascontiguousarray(faces, dtype="<u4").ravel()]
    views = []
    offset = 0
    for array, target in zip(arrays, (34962, 34962, 34963)):
        views.append({"buffer": 0, "byteOffset": offset,
                      "byteLength": array.nbytes, "target": target})
        offset += array.nbytes
    doc = {
        "asset": {"version": "2.0", "generator": "SAX-NeRF"},
        "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "NORMAL": 1},
                                     "indices": 2, "material": 0, "mode": 4}],
                    "extras": {"radiodensity_level": level}}],
        "materials": [{"pbrMetallicRoughness": {
            "baseColorFactor": [.72, .74, .77, .5],
            "metallicFactor": 0., "roughnessFactor": .8},
            "alphaMode": "BLEND", "doubleSided": True}],
        "buffers": [{"byteLength": offset}], "bufferViews": views,
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": len(vertices),
             "type": "VEC3", "min": arrays[0].min(axis=0).tolist(),
             "max": arrays[0].max(axis=0).tolist()},
            {"bufferView": 1, "componentType": 5126, "count": len(normals), "type": "VEC3"},
            {"bufferView": 2, "componentType": 5125, "count": arrays[2].size, "type": "SCALAR"},
        ],
    }
    metadata = json.dumps(doc, separators=(",", ":")).encode("utf-8")
    metadata += b" " * (-len(metadata) % 4)
    length = 12 + 8 + len(metadata) + 8 + offset
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(struct.pack("<III", 0x46546C67, 2, length))
        handle.write(struct.pack("<II", len(metadata), 0x4E4F534A))
        handle.write(metadata)
        handle.write(struct.pack("<II", offset, 0x004E4942))
        for array in arrays:
            handle.write(array.tobytes())
