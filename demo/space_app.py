"""The approved three-view UI on Gradio/ZeroGPU, with CPU native CT browsing."""
import os
from pathlib import Path
import sys
import threading

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(variable, "2")
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

# ZeroGPU's CUDA emulation must precede PyTorch and the original CUDA encoder.
import spaces
import torch
import gradio as gr
from gradio.context import LocalContext

from app import create_app, parse_args
from space_engine import SpaceEngine
from usage import UsageCounter

torch.set_num_threads(2)


def duration(request):
    return 60 if request.get("resolution", 192) >= 384 else 30


class SpaceWorker:
    info = {"device": "ZeroGPU", "torch": torch.__version__}

    def __init__(self, engine):
        self.engine = engine
        self.lock = threading.Lock()
        self.blocks = None

    def close(self):
        pass

    def call(self, operation, scene, **params):
        if operation == "load":
            return self.engine.load(scene)
        if operation == "slice":
            return self.engine.slice(scene, **params)
        if operation == "mesh":
            return self.engine.mesh(scene)
        if operation != "render":
            raise ValueError("Unknown operation")
        http_request = params.pop("http_request", None)
        cached = self.engine.cached_render(scene, **params)
        if cached is not None:
            return cached
        with self.lock:
            # Download/deserialize the trusted checkpoint before reserving GPU.
            self.engine.prepare_model(scene)
            request_token = LocalContext.request.set(gr.Request(request=http_request))
            blocks_token = LocalContext.blocks.set(self.blocks)
            try:
                return render_on_gpu(dict(scene=scene, **params))
            finally:
                LocalContext.blocks.reset(blocks_token)
                LocalContext.request.reset(request_token)


args = parse_args()
args.cache_dir = Path(os.environ.get("SAX_NERF_DEMO_CACHE", "/tmp/sax-nerf-demo"))
engine = SpaceEngine(args.cache_dir)
worker = SpaceWorker(engine)


@spaces.GPU(duration=duration)
def render_on_gpu(request):
    values = dict(request)
    return engine.render(values.pop("scene"), **values)


if __name__ == "__main__":
    usage = UsageCounter.from_env()
    usage.start()
    resources = create_app(args, worker_override=worker, usage=usage, hosted=True)
    # Gradio launches and registers ZeroGPU. Its GPU endpoint remains available,
    # while the public root serves the exact same HTML/Three.js local interface.
    with gr.Blocks(analytics_enabled=False, title="SAX-NeRF · CVPR 2024") as demo:
        request = gr.JSON(visible=False)
        result = gr.JSON(visible=False)
        trigger = gr.Button(visible=False)
        trigger.click(render_on_gpu, inputs=request, outputs=result, api_name="render_xray")
    worker.blocks = demo
    demo.queue(default_concurrency_limit=1)
    live_app, _, _ = demo.launch(
        server_name="0.0.0.0", server_port=int(os.environ.get("GRADIO_SERVER_PORT", 7860)),
        prevent_thread_lock=True, ssr_mode=False, show_error=True, show_api=False,
    )
    live_app.router.routes[0:0] = resources.router.routes
    print("SAX-NeRF online demo is ready", flush=True)
    demo.block_thread()
