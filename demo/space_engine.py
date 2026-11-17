"""Hosted SAX-NeRF inference with native CT assets and short GPU operations.

Import this module after ``spaces`` in a ZeroGPU Space. Meshes, slice browsing,
and packaged/cached projections use only CPU operations. The network is loaded
on the CPU by ``prepare_model``; only ``render`` moves it to CUDA. The original
local demo's subprocess worker remains independent of this hosted adapter.
"""
from __future__ import annotations

import copy
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import threading
import time
import uuid

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
MODEL_REVISION = "ee2140bc088ae569a051442dcf0062f25d64d160"
RESOLUTIONS = (128, 192, 256, 384, 512)


class SpaceEngine:
    """The three-view demo's CPU assets and original neural ray renderer."""

    def __init__(self, cache_dir, hosted_dir=None, weights_dir=None):
        self.cache = Path(cache_dir).resolve()
        self.hosted = Path(hosted_dir).resolve() if hosted_dir else ROOT / "hosted"
        self.weights_dir = Path(weights_dir).resolve() if weights_dir else None
        manifest = self.hosted / "scenes.json"
        if not manifest.is_file():
            raise FileNotFoundError(f"Hosted scene metadata is missing: {manifest}")
        document = json.loads(manifest.read_text())
        self.scenes = {}
        for item in document["scenes"]:
            meta = copy.deepcopy(item)
            scene, key = meta["scene"], meta["key"]
            if not all(re.fullmatch(r"[A-Za-z0-9_-]+", value) for value in (scene, key)):
                raise ValueError("Hosted scene identifiers must be simple filenames")
            meta.update(volume=f"volumes/{scene}-{key}.npy",
                        mesh=f"meshes/{scene}-{key}-preview.glb",
                        reference_ct_used=False)
            self.scenes[scene] = meta
        presets = json.loads((ROOT / "catalog.json").read_text())
        self.presets = {item["id"]: item for item in presets}
        for directory in ("volumes", "meshes", "images", "projections", "hub"):
            (self.cache / directory).mkdir(parents=True, exist_ok=True)
        self._prepared = None
        self._lock_pid = os.getpid()
        self._lock = threading.RLock()

    def _guard(self):
        # A ZeroGPU child must not inherit a lock held by its parent's thread.
        if self._lock_pid != os.getpid():
            self._lock_pid = os.getpid()
            self._lock = threading.RLock()
        return self._lock

    def _meta(self, scene):
        if scene not in self.scenes:
            raise ValueError("Unknown scene; choose one of the 15 scene thumbnails")
        return self.scenes[scene]

    @staticmethod
    def _temporary(path):
        return path.with_name(path.name + "." + uuid.uuid4().hex + ".partial")

    def _copy_asset(self, source, destination):
        if destination.is_file():
            return
        if not source.is_file():
            raise FileNotFoundError(f"The hosted scene asset is missing: {source}")
        temporary = self._temporary(destination)
        try:
            shutil.copyfile(source, temporary)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)

    def _volume(self, scene):
        meta = self._meta(scene)
        path = self.cache / meta["volume"]
        if not path.is_file():
            source = self.hosted / "volumes" / (scene + ".npy.gz")
            if not source.is_file():
                raise FileNotFoundError(f"The native reconstructed CT is missing: {source}")
            temporary = self._temporary(path)
            try:
                with gzip.open(source, "rb") as stream, temporary.open("wb") as output:
                    shutil.copyfileobj(stream, output, length=8 * 1024 * 1024)
                volume = np.load(temporary, mmap_mode="r", allow_pickle=False)
                if list(volume.shape) != meta["shape"] or volume.dtype != np.float32:
                    raise ValueError("The hosted CT must preserve the native float32 reconstruction")
                del volume
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        volume = np.load(path, mmap_mode="r", allow_pickle=False)
        if list(volume.shape) != meta["shape"] or volume.dtype != np.float32:
            raise ValueError("The cached CT does not match its native scene geometry")
        return volume

    def load(self, scene):
        """Prepare the native CT and preview mesh without CUDA."""
        with self._guard():
            meta = self._meta(scene)
            self._volume(scene)
            self._copy_asset(self.hosted / "meshes" / (scene + ".glb"),
                             self.cache / meta["mesh"])
            return copy.deepcopy(meta)

    def slice(self, scene, axis=1, index=0):
        """Return the original native-grid slice and its physical image aspect."""
        with self._guard():
            meta = self._meta(scene)
            axis = int(axis)
            if axis not in (0, 1, 2):
                raise ValueError("Slice axis must be X, Y or Z")
            total = meta["shape"][axis]
            index = max(0, min(total - 1, int(index)))
            relative = f"images/{scene}-{meta['key']}-slice-{axis}-{index}.png"
            path = self.cache / relative
            plane = meta["slice_planes"][axis]
            if not path.is_file():
                volume = self._volume(scene)
                remaining = [i for i in range(3) if i != axis]
                image = np.take(volume, index, axis=axis)
                image = image.transpose(remaining.index(plane["v"]), remaining.index(plane["u"]))
                if plane["v_sign"] > 0:
                    image = image[::-1, :]
                if plane["u_sign"] < 0:
                    image = image[:, ::-1]
                pixels = np.clip(image / meta["density_max"], 0, 1)
                temporary = self._temporary(path)
                try:
                    Image.fromarray(np.round(pixels * 255).astype(np.uint8)).save(temporary, format="PNG")
                    temporary.replace(path)
                finally:
                    temporary.unlink(missing_ok=True)
            size = meta["physical_size"]
            return dict(scene=scene, axis=axis, index=index, total=total, image=relative,
                        aspect_ratio=size[plane["u"]] / size[plane["v"]])

    @staticmethod
    def _view(azimuth=35, elevation=15, resolution=192, exposure=1, invert=False):
        azimuth, elevation, exposure = float(azimuth), float(elevation), float(exposure)
        if not all(math.isfinite(value) for value in (azimuth, elevation, exposure)):
            raise ValueError("View parameters must be finite")
        resolution = int(resolution)
        if resolution not in RESOLUTIONS:
            raise ValueError("Unsupported projection resolution")
        return dict(azimuth=azimuth % 360, elevation=max(-85., min(85., elevation)),
                    resolution=resolution, exposure=max(.1, min(4., exposure)), invert=bool(invert))

    def _projection_paths(self, scene, view):
        meta = self._meta(scene)
        signature = hashlib.sha256(
            f"{meta['key']}:{view['azimuth']:.4f}:{view['elevation']:.4f}:{view['resolution']}".encode()
        ).hexdigest()[:16]
        relative = f"projections/{scene}-{signature}.npy"
        return signature, relative, self.cache / relative, self.cache / "projections" / f"{scene}-{signature}.json"

    def _default_projection(self, scene, view, raw_path, sidecar):
        if not (f"{view['azimuth']:.4f}" == "35.0000"
                and f"{view['elevation']:.4f}" == "15.0000" and view["resolution"] == 192):
            return
        source = self.hosted / "projections" / (scene + ".npy")
        info = source.with_suffix(".json")
        if source.is_file() and info.is_file():
            self._copy_asset(source, raw_path)
            self._copy_asset(info, sidecar)

    def _projection_result(self, scene, view, signature, relative, raw_path, sidecar, cached):
        image = np.load(raw_path, allow_pickle=False)
        resolution = view["resolution"]
        if image.shape != (resolution, resolution) or not np.isfinite(image).all():
            raise ValueError("The cached X-ray projection has invalid dimensions or values")
        compute_ms = float(json.loads(sidecar.read_text())["compute_ms"])
        display_key = hashlib.sha256(
            f"{signature}:{view['exposure']:.4f}:{view['invert']}".encode()
        ).hexdigest()[:16]
        display = f"images/{scene}-{display_key}-projection.png"
        path = self.cache / display
        if not path.is_file():
            positive = np.maximum(image, 0)
            pixels = np.clip(positive / max(float(positive.max()), 1e-8) * view["exposure"], 0, 1)
            if view["invert"]:
                pixels = 1 - pixels
            temporary = self._temporary(path)
            try:
                Image.fromarray(np.round(pixels * 255).astype(np.uint8)).save(temporary, format="PNG")
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        return dict(scene=scene, image=display, download=relative, resolution=resolution,
                    azimuth=view["azimuth"], elevation=view["elevation"], compute_ms=compute_ms,
                    cached=cached, backend="SAX-NeRF neural ray rendering")

    def cached_render(self, scene, **view):
        """Return a stored neural projection, applying display controls on CPU."""
        with self._guard():
            view = self._view(**view)
            signature, relative, path, sidecar = self._projection_paths(scene, view)
            if not path.is_file() or not sidecar.is_file():
                self._default_projection(scene, view, path, sidecar)
            if not path.is_file() or not sidecar.is_file():
                return None
            return self._projection_result(scene, view, signature, relative, path, sidecar, cached=True)

    def prepare_model(self, scene):
        """Load this published checkpoint into the original network on CPU."""
        with self._guard():
            self._meta(scene)
            if self._prepared is not None and self._prepared["scene"] == scene:
                return self._prepared
            import gc
            import torch
            from src.config.configloading import load_config
            from src.encoder import get_encoder
            from src.network import get_network

            self._prepared = None
            gc.collect()
            cfg = load_config(str(REPO / "config/Lineformer" / (scene + "_50.yaml")))
            local = self.weights_dir / (scene + ".tar") if self.weights_dir else None
            if local is not None and local.is_file():
                weights = local
            else:
                from huggingface_hub import hf_hub_download
                weights = hf_hub_download("CaiYuanhao/SAX-NeRF", scene + ".tar",
                                          revision=MODEL_REVISION, cache_dir=str(self.cache / "hub"))
            options = dict(cfg["network"])
            network = get_network(options.pop("net_type"))
            encoder = get_encoder(**cfg["encoder"])
            model = network(encoder, **options)
            fine = network(encoder, **options) if cfg["render"]["n_fine"] > 0 else None
            # These are the project's trusted published checkpoints, including
            # their original NumPy/optimizer metadata, rather than user uploads.
            checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
            model.load_state_dict(checkpoint["network"])
            model.eval()
            if fine is not None:
                fine.load_state_dict(checkpoint["network_fine"])
                fine.eval()
            self._prepared = dict(scene=scene, cfg=cfg, model=model, fine=fine)
            del checkpoint
            gc.collect()
            return self._prepared

    def render(self, scene, **view):
        """Render a real neural X-ray; call only inside a GPU allocation."""
        with self._guard():
            view = self._view(**view)
            cached = self.cached_render(scene, **view)
            if cached is not None:
                return cached
            import torch
            from src.dataset.tigre import ConeGeometry
            from src.render import render

            if not torch.cuda.is_available():
                raise RuntimeError("Neural X-ray rendering requires an active CUDA/ZeroGPU allocation")
            # A reused ZeroGPU child may not see later changes to its parent's
            # one-scene model cache. Ensure the requested scene in this process.
            prepared = self.prepare_model(scene)
            model = prepared["model"].cuda()
            fine = prepared["fine"].cuda() if prepared["fine"] is not None else None
            meta = self._meta(scene)
            geo = ConeGeometry(meta["geometry"])
            resolution = view["resolution"]
            theta, phi = math.radians(view["azimuth"]), math.radians(view["elevation"])
            inverse = np.asarray(meta["display_rotation"]).T
            eye_world = np.array([math.cos(phi) * math.cos(theta), math.sin(phi), -math.cos(phi) * math.sin(theta)])
            right_world = np.array([-math.sin(theta), 0., -math.cos(theta)])
            up_world = np.cross(right_world, -eye_world)
            eye_direction, right, up = (inverse @ value for value in (eye_world, right_world, up_world))
            forward = -eye_direction
            rows, columns = torch.meshgrid(torch.arange(resolution, device="cuda"),
                                            torch.arange(resolution, device="cuda"), indexing="ij")
            u = (columns + .5 - resolution / 2) * float(geo.sDetector[0]) / resolution
            v = (resolution / 2 - rows - .5) * float(geo.sDetector[1]) / resolution
            direction = (torch.tensor(forward, dtype=torch.float32, device="cuda")
                         + (u / geo.DSD)[..., None] * torch.tensor(right, dtype=torch.float32, device="cuda")
                         + (v / geo.DSD)[..., None] * torch.tensor(up, dtype=torch.float32, device="cuda"))
            direction = torch.nn.functional.normalize(direction.reshape(-1, 3), dim=-1)
            origins = torch.tensor(eye_direction * geo.DSO, dtype=torch.float32, device="cuda").expand_as(direction)
            half = torch.tensor(geo.sVoxel / 2, dtype=torch.float32, device="cuda")
            safe = torch.where(direction.abs() < 1e-8, torch.full_like(direction, 1e-8), direction)
            first, second = (-half - origins) / safe, (half - origins) / safe
            near = torch.minimum(first, second).max(dim=-1).values.clamp_min(0)
            far = torch.maximum(first, second).min(dim=-1).values
            hits = far > near
            rays = torch.cat((origins[hits], direction[hits], near[hits, None], far[hits, None]), dim=-1)
            result = torch.zeros(resolution * resolution, dtype=torch.float32, device="cuda")
            options = dict(prepared["cfg"]["render"])
            options.update(perturb=False, raw_noise_std=0.)
            values = []
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.no_grad():
                for start in range(0, len(rays), 1024):
                    values.append(render(rays[start:start + 1024], model, fine, **options)["acc"])
            if values:
                result[hits] = torch.cat(values)
            torch.cuda.synchronize()
            compute_ms = (time.perf_counter() - started) * 1000
            image = result.reshape(resolution, resolution).cpu().numpy()
            if not np.isfinite(image).all():
                raise ValueError("The rendered X-ray projection contains nonfinite values")
            signature, relative, path, sidecar = self._projection_paths(scene, view)
            temporary, temporary_info = self._temporary(path), self._temporary(sidecar)
            try:
                with temporary.open("wb") as stream:
                    np.save(stream, image, allow_pickle=False)
                temporary_info.write_text(json.dumps(dict(compute_ms=compute_ms)))
                temporary.replace(path)
                temporary_info.replace(sidecar)
            finally:
                temporary.unlink(missing_ok=True)
                temporary_info.unlink(missing_ok=True)
            return self._projection_result(scene, view, signature, relative, path, sidecar, cached=False)

    def mesh(self, scene):
        """Extract the native-grid surface on CPU, preserving scanner units."""
        with self._guard():
            meta = self._meta(scene)
            relative = f"meshes/{scene}-{meta['key']}-native.glb"
            path = self.cache / relative
            if not path.is_file():
                from src.utils.mesh import export_ct_mesh
                threshold = meta.get("mesh_threshold_fraction",
                                     self.presets.get(scene, {}).get("mesh_threshold_fraction", .5))
                temporary = self._temporary(path).with_suffix(".glb")
                try:
                    export_ct_mesh(self._volume(scene), temporary, meta["spacing"], meta["origin"],
                                   level=meta["density_max"] * threshold)
                    temporary.replace(path)
                finally:
                    temporary.unlink(missing_ok=True)
            return dict(file=relative)
