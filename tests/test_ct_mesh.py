"""CPU checks of CT geometry, closed boundaries, and GLB output."""
import json
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

from src.utils.mesh import export_ct_mesh


def read_glb(path):
    data = path.read_bytes()
    magic, version, length = struct.unpack_from("<III", data)
    assert (magic, version, length) == (0x46546C67, 2, len(data))
    json_size, json_kind = struct.unpack_from("<II", data, 12)
    assert json_kind == 0x4E4F534A and json_size % 4 == 0
    doc = json.loads(data[20:20 + json_size])
    bin_size, bin_kind = struct.unpack_from("<II", data, 20 + json_size)
    assert bin_kind == 0x004E4942 and bin_size == doc["buffers"][0]["byteLength"]
    binary = data[28 + json_size:]
    arrays = []
    for a in doc["accessors"]:
        view = doc["bufferViews"][a["bufferView"]]
        offset = view.get("byteOffset", 0) + a.get("byteOffset", 0)
        dtype = {5126: "<f4", 5125: "<u4"}[a["componentType"]]
        width = {"VEC3": 3, "SCALAR": 1}[a["type"]]
        arrays.append(np.frombuffer(binary, dtype=dtype, count=a["count"] * width,
                                    offset=offset).reshape(-1, width))
    return arrays[0], arrays[1], arrays[2].reshape(-1, 3)


class CTMeshTest(unittest.TestCase):
    def test_anisotropic_ellipsoid_and_unchanged_input(self):
        shape = np.array([29, 35, 41])
        spacing = np.array([.04, .06, .1])
        origin = np.array([.7, -.9, 1.2])
        center = origin + (shape - 1) * spacing / 2
        radii = np.array([.22, .45, .72])
        xyz = np.stack(np.meshgrid(*[o + np.arange(n) * d for o, n, d in
                                    zip(origin, shape, spacing)], indexing="ij"), axis=-1)
        volume = np.exp(-np.log(2) * np.sum(((xyz - center) / radii) ** 2, axis=-1)).astype("f4")
        snapshot = volume.copy()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "volume.glb"
            info = export_ct_mesh(volume, path, spacing, origin, level=.5)
            vertices, normals, faces = read_glb(path)
        transformed_center = center[[0, 2, 1]] * [1, 1, -1]
        bounds = np.stack([vertices.min(axis=0), vertices.max(axis=0)])
        expected = transformed_center + np.array([-1, 1])[:, None] * radii[[0, 2, 1]]
        np.testing.assert_allclose(bounds, expected, atol=spacing.max())
        np.testing.assert_array_equal(volume, snapshot)
        np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1, atol=1e-5)
        self.assertTrue((np.sum(normals * (vertices - transformed_center), axis=1) > 0).all())
        analytic_normals = (vertices - transformed_center) / radii[[0, 2, 1]] ** 2
        analytic_normals /= np.linalg.norm(analytic_normals, axis=1, keepdims=True)
        self.assertTrue((np.sum(normals * analytic_normals, axis=1) > .98).all())
        triangles = vertices[faces]
        face_normals = np.cross(triangles[:, 1] - triangles[:, 0],
                                triangles[:, 2] - triangles[:, 0])
        self.assertTrue((np.sum(face_normals * (triangles.mean(axis=1) - transformed_center), axis=1) > 0).all())
        self.assertEqual((info["vertices"], info["faces"]), (len(vertices), len(faces)))

    def test_surface_touching_grid_boundary_is_closed(self):
        volume = np.ones((5, 7, 9), dtype="f4")
        spacing = np.array([.2, .3, .4])
        origin = np.array([-.4, 1.2, 3.4])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "volume.glb"
            export_ct_mesh(volume, path, spacing, origin)
            vertices, _, faces = read_glb(path)
        lo = origin - spacing / 2
        hi = origin + (np.array(volume.shape) - .5) * spacing
        expected = np.array([[lo[0], lo[2], -hi[1]], [hi[0], hi[2], -lo[1]]])
        np.testing.assert_allclose([vertices.min(axis=0), vertices.max(axis=0)], expected, atol=1e-6)
        edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
        _, counts = np.unique(edges, axis=0, return_counts=True)
        self.assertTrue((counts == 2).all())
        np.testing.assert_array_equal(volume, np.ones_like(volume))

    def test_signed_background_preserves_positive_isosurface_and_input(self):
        # LeakyReLU checkpoints (e.g. Pelvis) can predict negative background.
        volume = np.full((5, 7, 9), -1., dtype="f4")
        volume[2, 3, 4] = 3.
        snapshot = volume.copy()
        spacing = np.array([.2, .3, .4])
        origin = np.array([-.4, 1.2, 3.4])
        center = origin + np.array([2, 3, 4]) * spacing
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "volume.glb"
            info = export_ct_mesh(volume, path, spacing, origin)
            vertices, normals, faces = read_glb(path)
        # At level 1.5, interpolation from 3 to -1 crosses 0.375 voxels
        # from the center. Clipping negatives to zero would change the surface.
        center = center[[0, 2, 1]] * [1, 1, -1]
        expected = center + np.array([-1, 1])[:, None] * (.375 * spacing[[0, 2, 1]])
        np.testing.assert_allclose([vertices.min(0), vertices.max(0)], expected, atol=1e-6)
        np.testing.assert_array_equal(volume, snapshot)
        np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1, atol=1e-5)
        self.assertEqual(info["level"], 1.5)
        self.assertEqual((len(vertices), len(faces)), (6, 8))

    def test_invalid_fields_and_geometry_do_not_write_a_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "volume.glb"
            cases = [dict(volume=np.zeros((3, 3, 3))),
                     dict(volume=-np.ones((3, 3, 3))),
                     dict(volume=np.full((3, 3, 3), np.nan)),
                     dict(volume=np.ones((1, 3, 3))),
                     dict(spacing=[1, 0, 1]), dict(origin=[0, np.inf, 0]),
                     dict(level=0), dict(level=1)]
            for changes in cases:
                with self.subTest(changes=list(changes)):
                    params = dict(volume=np.ones((3, 3, 3)), spacing=[1, 1, 1], origin=[0, 0, 0])
                    params.update(changes)
                    with self.assertRaises(ValueError):
                        export_ct_mesh(output_path=path, **params)
                    self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
