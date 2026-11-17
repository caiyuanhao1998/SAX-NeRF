"""SAX-NeRF inference worker. Runs in the original PyTorch/CUDA environment.

The UI can use a different Python environment. Standard output is reserved for
the JSON protocol; model imports and compiler messages go to standard error.
"""
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
PROTOCOL = sys.stdout
sys.stdout = sys.stderr
sys.path.insert(0, str(REPO))


def respond(value):
    print(json.dumps(value, allow_nan=False), file=PROTOCOL, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--gpu-id", default="0")
    args = parser.parse_args()
    cache = args.cache_dir.resolve()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    os.environ.setdefault("MPLBACKEND", "Agg")

    import numpy as np
    from PIL import Image
    import torch
    from src.config.configloading import load_config
    from src.dataset.tigre import ConeGeometry
    from src.encoder import get_encoder
    from src.network import get_network
    from src.render import render
    from src.utils.mesh import export_ct_mesh

    if not torch.cuda.is_available():
        raise RuntimeError("SAX-NeRF rendering requires CUDA. Select a CUDA-enabled model environment.")
    torch.set_num_threads(2)
    for folder in ("scenes", "volumes", "meshes", "images", "projections"):
        (cache / folder).mkdir(parents=True, exist_ok=True)
    catalog = {item["id"]: item for item in json.loads((ROOT / "catalog.json").read_text())}
    current = None

    def fingerprint(scene, dataset, weights):
        digest = hashlib.sha256()
        digest.update(b"sax-local-demo-native-ct-v1")
        for path in (Path(dataset), Path(weights)):
            info = path.stat()
            digest.update(f"{path.resolve()}:{info.st_size}:{info.st_mtime_ns}".encode())
        for path in (REPO / "config/Lineformer" / (scene + "_50.yaml"),
                     REPO / "src/network/Lineformer.py", REPO / "src/utils/mesh.py", Path(__file__), ROOT / "catalog.json"):
            digest.update(path.read_bytes())
        return digest.hexdigest()[:16]

    def ensure_model(ctx):
        if ctx.get("model") is not None:
            return
        cfg = ctx["cfg"]
        net_options = dict(cfg["network"])
        network = get_network(net_options.pop("net_type"))
        encoder = get_encoder(**cfg["encoder"])
        model = network(encoder, **net_options).cuda()
        fine = network(encoder, **net_options).cuda() if cfg["render"]["n_fine"] > 0 else None
        checkpoint = torch.load(ctx["weights"], map_location="cpu")
        model.load_state_dict(checkpoint["network"])
        model.eval()
        if fine is not None:
            fine.load_state_dict(checkpoint["network_fine"])
            fine.eval()
        ctx.update(model=model, fine=fine)
        del checkpoint
        gc.collect()

    def context(request):
        nonlocal current
        scene = request["scene"]
        if scene not in catalog:
            raise ValueError("Unknown scene")
        dataset, weights = Path(request["dataset"]), Path(request["weights"])
        key = fingerprint(scene, dataset, weights)
        if current and current["scene"] == scene and current["key"] == key:
            return current
        current = None
        gc.collect()
        torch.cuda.empty_cache()
        cfg = load_config(str(REPO / "config/Lineformer" / (scene + "_50.yaml")))
        meta_file = cache / "scenes" / f"{scene}-{key}.json"
        if meta_file.is_file():
            meta = json.loads(meta_file.read_text())
        else:
            with dataset.open("rb") as stream:
                data = pickle.load(stream)
            fields = ("DSD", "DSO", "nDetector", "dDetector", "nVoxel", "dVoxel",
                      "offOrigin", "offDetector", "accuracy", "mode", "filter")
            geometry = {name: data[name].tolist() if isinstance(data[name], np.ndarray) else data[name]
                        for name in fields}
            shape = [int(n) for n in data["nVoxel"]]
            if tuple(shape) != data["image"].shape:
                raise ValueError("The dataset geometry and CT dimensions disagree")
            del data
            gc.collect()
            geo = ConeGeometry(geometry)
            origin = (-(geo.nVoxel - 1) * geo.dVoxel / 2).astype(np.float32)
            meta = dict(scene=scene, key=key, name=scene.title(), geometry=geometry, shape=shape,
                        spacing=geo.dVoxel.tolist(), origin=origin.tolist(),
                        physical_size=geo.sVoxel.tolist(), reference_ct_used=False,
                        display_rotation=catalog[scene]["display_rotation"],
                        volume=f"volumes/{scene}-{key}.npy", mesh=f"meshes/{scene}-{key}-preview.glb")
            # A signed permutation keeps every displayed pixel on its original
            # voxel while presenting the scene upright, as on the project page.
            rotation = np.asarray(meta["display_rotation"])
            world_axes = np.abs(rotation).argmax(axis=1).tolist()
            signs = rotation[np.arange(3), world_axes].astype(int).tolist()
            planes = []
            for axis in range(3):
                horizontal = axis == world_axes[1]
                u_world = 0 if horizontal or axis == world_axes[2] else 2
                v_world = 2 if horizontal else 1
                planes.append(dict(u=world_axes[u_world], v=world_axes[v_world],
                                   u_sign=signs[u_world] * (-1 if u_world == 2 else 1),
                                   v_sign=signs[v_world] * (-1 if v_world == 2 else 1)))
            meta["slice_planes"] = planes
        ctx = dict(scene=scene, key=key, cfg=cfg, weights=str(weights), meta=meta,
                   geo=ConeGeometry(meta["geometry"]), model=None, fine=None)
        volume_path = cache / meta["volume"]
        if not volume_path.is_file():
            ensure_model(ctx)
            shape = tuple(meta["shape"])
            spacing = np.asarray(meta["spacing"])
            # Match TIGREDataset.get_voxels exactly, without allocating its entire
            # XYZ grid or loading training/reference projections onto the GPU.
            half = np.asarray(shape) * spacing / 2 - spacing / 2
            axes = [np.linspace(-s, s, n).astype(np.float32) for s, n in zip(half, shape)]
            temporary = volume_path.with_suffix(".partial.npy")
            volume = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32, shape=shape)
            flat = volume.reshape(-1)
            chunk = int(cfg["render"]["netchunk"])
            field = ctx["fine"] if ctx["fine"] is not None else ctx["model"]
            with torch.no_grad():
                for start in range(0, flat.size, chunk):
                    indices = np.arange(start, min(start + chunk, flat.size))
                    x = indices // (shape[1] * shape[2])
                    y = (indices // shape[2]) % shape[1]
                    z = indices % shape[2]
                    points = np.column_stack((axes[0][x], axes[1][y], axes[2][z]))
                    values = field(torch.from_numpy(points).cuda()).reshape(-1).cpu().numpy()
                    if not np.isfinite(values).all():
                        raise ValueError("The reconstructed CT contains nonfinite values")
                    flat[start:start + len(values)] = values
            volume.flush()
            del flat, volume
            temporary.replace(volume_path)
        volume = np.load(volume_path, mmap_mode="r")
        ctx["volume"] = volume
        if "density_max" not in meta:
            meta["density_min"], meta["density_max"] = float(volume.min()), float(volume.max())
        if meta["density_max"] <= 0:
            raise ValueError("The reconstructed CT has no positive radiodensity")
        mesh_path = cache / meta["mesh"]
        if not mesh_path.is_file():
            stride = max(1, math.ceil(max(meta["shape"]) / 192))
            info = export_ct_mesh(volume[::stride, ::stride, ::stride], mesh_path,
                                  spacing=np.asarray(meta["spacing"]) * stride, origin=meta["origin"],
                                  level=meta["density_max"] * catalog[scene]["mesh_threshold_fraction"])
            meta.update(mesh_vertices=info["vertices"], mesh_faces=info["faces"], mesh_stride=stride)
        meta_file.write_text(json.dumps(meta, indent=2))
        current = ctx
        return ctx

    def public_meta(ctx):
        return ctx["meta"]

    def slice_image(request):
        ctx = context(request)
        axis = int(request.get("axis", 1))
        if axis not in (0, 1, 2):
            raise ValueError("Slice axis must be X, Y or Z")
        total = ctx["meta"]["shape"][axis]
        index = max(0, min(total - 1, int(request.get("index", total // 2))))
        path = f"images/{ctx['scene']}-{ctx['key']}-slice-{axis}-{index}.png"
        if not (cache / path).is_file():
            plane = ctx["meta"]["slice_planes"][axis]
            remaining = [i for i in range(3) if i != axis]
            image = np.take(ctx["volume"], index, axis=axis)
            image = image.transpose(remaining.index(plane["v"]), remaining.index(plane["u"]))
            if plane["v_sign"] > 0:
                image = image[::-1, :]
            if plane["u_sign"] < 0:
                image = image[:, ::-1]
            pixels = np.clip(image / ctx["meta"]["density_max"], 0, 1)
            Image.fromarray(np.round(pixels * 255).astype(np.uint8)).save(cache / path)
        plane = ctx["meta"]["slice_planes"][axis]
        size = ctx["meta"]["physical_size"]
        return dict(scene=ctx["scene"], axis=axis, index=index, total=total, image=path,
                    aspect_ratio=size[plane["u"]] / size[plane["v"]])

    def projection(request):
        ctx = context(request)
        ensure_model(ctx)
        resolution = int(request.get("resolution", 192))
        if resolution not in (128, 192, 256, 384, 512):
            raise ValueError("Unsupported projection resolution")
        azimuth = float(request.get("azimuth", 35)) % 360
        elevation = max(-85., min(85., float(request.get("elevation", 15))))
        exposure = max(.1, min(4., float(request.get("exposure", 1))))
        invert = bool(request.get("invert", False))
        if not all(math.isfinite(value) for value in (azimuth, elevation, exposure)):
            raise ValueError("View parameters must be finite")
        signature = hashlib.sha256(f"{ctx['key']}:{azimuth:.4f}:{elevation:.4f}:{resolution}".encode()).hexdigest()[:16]
        raw_path = f"projections/{ctx['scene']}-{signature}.npy"
        sidecar = cache / "projections" / f"{ctx['scene']}-{signature}.json"
        if (cache / raw_path).is_file() and sidecar.is_file():
            image = np.load(cache / raw_path)
            compute_ms = json.loads(sidecar.read_text())["compute_ms"]
            cached = True
        else:
            theta, phi = math.radians(azimuth), math.radians(elevation)
            # Match the browser's display-space camera, including image roll.
            # The radiodensity network and scanner retain native XYZ coordinates.
            inverse = np.asarray(ctx["meta"]["display_rotation"]).T
            eye_world = np.array([math.cos(phi) * math.cos(theta), math.sin(phi), -math.cos(phi) * math.sin(theta)])
            right_world = np.array([-math.sin(theta), 0., -math.cos(theta)])
            up_world = np.cross(right_world, -eye_world)
            eye_direction, right, up = (inverse @ v for v in (eye_world, right_world, up_world))
            forward = -eye_direction
            geo = ctx["geo"]
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
            options = dict(ctx["cfg"]["render"])
            options.update(perturb=False, raw_noise_std=0.)
            values = []
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.no_grad():
                for start in range(0, len(rays), 1024):
                    values.append(render(rays[start:start + 1024], ctx["model"], ctx["fine"], **options)["acc"])
            if values:
                result[hits] = torch.cat(values)
            torch.cuda.synchronize()
            compute_ms = (time.perf_counter() - started) * 1000
            image = result.reshape(resolution, resolution).cpu().numpy()
            if not np.isfinite(image).all():
                raise ValueError("The rendered projection contains nonfinite values")
            np.save(cache / raw_path, image)
            sidecar.write_text(json.dumps(dict(compute_ms=compute_ms)))
            cached = False
        display_key = hashlib.sha256(f"{signature}:{exposure:.4f}:{invert}".encode()).hexdigest()[:16]
        display_path = f"images/{ctx['scene']}-{display_key}-projection.png"
        if not (cache / display_path).is_file():
            positive = np.maximum(image, 0)
            pixels = np.clip(positive / max(float(positive.max()), 1e-8) * exposure, 0, 1)
            if invert:
                pixels = 1 - pixels
            Image.fromarray(np.round(pixels * 255).astype(np.uint8)).save(cache / display_path)
        return dict(scene=ctx["scene"], image=display_path, download=raw_path, resolution=resolution,
                    azimuth=azimuth, elevation=elevation, compute_ms=compute_ms, cached=cached,
                    backend="SAX-NeRF neural ray rendering")

    respond(dict(ok=True, ready=True, torch=torch.__version__, cuda=torch.version.cuda,
                 device=torch.cuda.get_device_name(0)))
    for line in sys.stdin:
        try:
            request = json.loads(line)
            operation = request["operation"]
            if operation == "load":
                result = public_meta(context(request))
            elif operation == "slice":
                result = slice_image(request)
            elif operation == "render":
                result = projection(request)
            elif operation == "mesh":
                ctx = context(request)
                path = f"meshes/{ctx['scene']}-{ctx['key']}-native.glb"
                if not (cache / path).is_file():
                    export_ct_mesh(ctx["volume"], cache / path, ctx["meta"]["spacing"], ctx["meta"]["origin"],
                                   level=ctx["meta"]["density_max"] * catalog[ctx["scene"]]["mesh_threshold_fraction"])
                result = dict(file=path)
            else:
                raise ValueError("Unknown operation")
            respond(dict(ok=True, result=result))
        except Exception as error:
            traceback.print_exc(file=sys.stderr)
            respond(dict(ok=False, error=str(error)))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        traceback.print_exc(file=sys.stderr)
        respond(dict(ok=False, error=str(error)))
        raise SystemExit(1)
