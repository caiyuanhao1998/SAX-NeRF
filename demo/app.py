"""Three-view local SAX-NeRF demo: interactive mesh, neural X-ray, and CT slice."""
from __future__ import annotations

import argparse
import atexit
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
CATALOG = json.loads((ROOT / "catalog.json").read_text())
SCENES = {item["id"]: item for item in CATALOG}
MODEL_REVISION = "ee2140bc088ae569a051442dcf0062f25d64d160"
DATA_REVISION = "31f7c039e9b9b420fbc82c24170e83bc94410652"


def local_config():
    path = ROOT / "local_config.json"
    return json.loads(path.read_text()) if path.is_file() else {}


class Assets:
    def __init__(self, args):
        self.cache = args.cache_dir.resolve()
        self.data_dir = args.data_dir
        self.weights_dir = args.weights_dir
        self.offline = args.offline
        self.lock = threading.Lock()
        self.resolved = {}

    def get(self, scene):
        if scene not in SCENES:
            raise HTTPException(404, "Unknown scene")
        with self.lock:
            if scene in self.resolved:
                return self.resolved[scene]
            from huggingface_hub import hf_hub_download

            def resolve(dataset):
                filename = scene + ("_50.pickle" if dataset else ".tar")
                directory = self.data_dir if dataset else self.weights_dir
                path = directory / filename
                if path.is_file():
                    return str(path.resolve())
                return hf_hub_download(
                    "CaiYuanhao/SAX-NeRF", filename,
                    repo_type="dataset" if dataset else "model",
                    revision=DATA_REVISION if dataset else MODEL_REVISION,
                    cache_dir=str(self.cache / "hub"), local_files_only=self.offline,
                )

            with ThreadPoolExecutor(max_workers=2) as pool:
                dataset = pool.submit(resolve, True)
                weights = pool.submit(resolve, False)
                self.resolved[scene] = dict(dataset=dataset.result(), weights=weights.result())
            return self.resolved[scene]


class Worker:
    def __init__(self, args, assets):
        self.args, self.assets = args, assets
        self.lock = threading.Lock()
        self.process = None
        self.log = None
        self.info = None
        atexit.register(self.close)

    def start(self):
        if self.process is not None and self.process.poll() is None:
            return
        if self.log is not None:
            self.log.close()
        self.log = (self.args.cache_dir / "worker.log").open("a", buffering=1)
        env = os.environ.copy()
        env.update(PYTHONNOUSERSITE="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
                   OPENBLAS_NUM_THREADS="2", MPLBACKEND="Agg",
                   TORCH_EXTENSIONS_DIR=str(self.args.cache_dir / "torch-extensions"),
                   PYTHONPYCACHEPREFIX=str(self.args.cache_dir / "python-cache"),
                   TMPDIR=str(self.args.cache_dir / "tmp"))
        env.pop("PYTHONPATH", None)
        # Launching the UI from a separate environment must still find the
        # model environment's Ninja and other extension build tools.
        env["PATH"] = str(self.args.model_python.parent) + ":" + env.get("PATH", "")
        if self.args.cuda_home:
            cuda = str(self.args.cuda_home)
            env.update(CUDA_HOME=cuda, CUDA_PATH=cuda)
            env["PATH"] = cuda + "/bin:" + env.get("PATH", "")
            env["LD_LIBRARY_PATH"] = cuda + "/lib64:" + env.get("LD_LIBRARY_PATH", "")
        if self.args.cxx:
            env["CXX"] = str(self.args.cxx)
            directory = str(self.args.cxx.parent)
            env["PATH"] = directory + ":" + env["PATH"]
            candidate = self.args.cxx.with_name("x86_64-conda-linux-gnu-cc")
            if candidate.is_file():
                env["CC"] = str(candidate)
            library = self.args.cxx.parent.parent / "lib"
            env["LD_LIBRARY_PATH"] = str(library) + ":" + env.get("LD_LIBRARY_PATH", "")
        command = [str(self.args.model_python), "-u", str(ROOT / "worker.py"),
                   "--cache-dir", str(self.args.cache_dir), "--gpu-id", self.args.gpu_id]
        self.process = subprocess.Popen(command, cwd=REPO, env=env, stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, stderr=self.log, text=True, bufsize=1)
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("The model worker did not start. Check the local worker.log and model environment.")
        self.info = json.loads(line)
        if not self.info.get("ok"):
            raise RuntimeError(self.info.get("error", "Unable to initialize SAX-NeRF"))

    def call(self, operation, scene, **params):
        files = self.assets.get(scene)
        with self.lock:
            self.start()
            request = dict(operation=operation, scene=scene, **files, **params)
            self.process.stdin.write(json.dumps(request) + "\n")
            self.process.stdin.flush()
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError("The model worker stopped. Check worker.log, then retry.")
            response = json.loads(line)
            if not response.get("ok"):
                raise RuntimeError(response.get("error", "SAX-NeRF inference failed"))
            return response["result"]

    def close(self):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        if self.log is not None:
            self.log.close()
            self.log = None


class RenderRequest(BaseModel):
    scene: str
    azimuth: float = 35
    elevation: float = 15
    resolution: int = 192
    exposure: float = Field(default=1, ge=.1, le=4)
    invert: bool = False
    count_use: bool = False


def create_app(args, *, worker_override=None, usage=None, hosted=False):
    args.cache_dir = args.cache_dir.resolve()
    for folder in ("tmp", "hub", "torch-extensions", "python-cache"):
        (args.cache_dir / folder).mkdir(parents=True, exist_ok=True)
    assets = Assets(args)
    worker = worker_override if worker_override is not None else Worker(args, assets)

    @asynccontextmanager
    async def lifespan(app):
        yield
        worker.close()

    app = FastAPI(title="SAX-NeRF local demo", lifespan=lifespan)
    app.state.worker = worker
    app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")

    def operation(name, scene, **params):
        try:
            return worker.call(name, scene, **params)
        except HTTPException:
            raise
        except Exception as error:
            raise HTTPException(503, str(error)) from error

    def file_url(path):
        return "/files/" + path

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(ROOT / "static/index.html")

    @app.get("/health")
    def health():
        return dict(ok=True, scenes=len(CATALOG), worker_ready=worker.info is not None,
                    backend="SAX-NeRF", model=worker.info)

    @app.get("/api/scenes")
    def scenes():
        return dict(scenes=[dict(**item, thumbnail=f"/api/thumbnail/{item['id']}") for item in CATALOG],
                    default_scene=args.scene, hosted=hosted)

    @app.get("/api/usage")
    def usage_summary():
        data = usage.summary() if usage is not None else None
        return dict(available=True, **data) if data else dict(available=False)

    @app.get("/api/thumbnail/{scene}")
    def thumbnail(scene: str):
        if scene not in SCENES:
            raise HTTPException(404)
        for suffix in ("-zoom85.png", ".png"):
            path = REPO / "fig/meshes" / (scene + suffix)
            if path.is_file():
                return FileResponse(path)
        raise HTTPException(404, "Scene preview not found")

    @app.post("/api/scene/{scene}")
    def load(scene: str):
        meta = operation("load", scene)
        return dict(**meta, mesh_url=file_url(meta["mesh"]), volume_url=file_url(meta["volume"]),
                    default_azimuth=35, default_elevation=15)

    @app.get("/api/slice/{scene}")
    def slice_data(scene: str, axis: int = 1, index: int = 0):
        result = operation("slice", scene, axis=axis, index=index)
        return dict(**result, image_url=file_url(result["image"]))

    @app.post("/api/render")
    def render(payload: RenderRequest, request: Request):
        params = dict(azimuth=payload.azimuth, elevation=payload.elevation,
                      resolution=payload.resolution, exposure=payload.exposure, invert=payload.invert)
        if hosted:
            params["http_request"] = request
        result = operation("render", payload.scene, **params)
        if usage is not None and payload.count_use:
            usage.record()
        return dict(**result, image_url=file_url(result["image"]), download_url=file_url(result["download"]))

    @app.get("/api/download/mesh/{scene}")
    def download_mesh(scene: str):
        result = operation("mesh", scene)
        if usage is not None:
            usage.record()
        return FileResponse(args.cache_dir / result["file"], media_type="model/gltf-binary",
                            filename=scene + "-sax-nerf.glb")

    @app.get("/files/{kind}/{filename}")
    def generated_file(kind: str, filename: str):
        if kind not in {"images", "meshes", "volumes", "projections"} or Path(filename).name != filename:
            raise HTTPException(404)
        path = args.cache_dir / kind / filename
        if not path.is_file():
            raise HTTPException(404)
        return FileResponse(path)

    return app


def parse_args():
    config = local_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7864)
    parser.add_argument("--scene", choices=list(SCENES), default="head")
    parser.add_argument("--gpu-id", "--gpu_id", default=str(config.get("gpu_id", "0")))
    parser.add_argument("--model-python", type=Path,
                        default=os.environ.get("SAX_NERF_PYTHON", config.get("model_python", sys.executable)))
    parser.add_argument("--cache-dir", type=Path,
                        default=os.environ.get("SAX_NERF_DEMO_CACHE", config.get("cache_dir", ROOT / "cache")))
    parser.add_argument("--data-dir", type=Path, default=config.get("data_dir", REPO / "data"))
    parser.add_argument("--weights-dir", type=Path, default=config.get("weights_dir", REPO / "pretrained"))
    parser.add_argument("--cuda-home", type=Path, default=config.get("cuda_home"))
    parser.add_argument("--cxx", type=Path, default=config.get("cxx"))
    parser.add_argument("--offline", action="store_true", help="Use only already-cached or local data and weights")
    parser.add_argument("--prepare-all", action="store_true", help="Cache every scene's model, CT, mesh and default projection, then exit")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    app = create_app(args)
    if args.prepare_all:
        try:
            for item in CATALOG:
                scene = item["id"]
                app.state.worker.call("load", scene)
                app.state.worker.call("render", scene, azimuth=35, elevation=15, resolution=192)
                print("Prepared " + scene, flush=True)
        finally:
            app.state.worker.close()
    else:
        print(f"Local SAX-NeRF demo: http://{args.host}:{args.port}", flush=True)
        uvicorn.run(app, host=args.host, port=args.port, access_log=False)
