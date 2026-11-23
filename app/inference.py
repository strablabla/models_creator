"""Loading models (PyTorch: ours, a bare state_dict or TorchScript; TensorFlow:
Keras files) and predicting masks."""
import io
import json
import threading
import uuid
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from . import storage
from .training import apply_torch_threads, pick_device
from .unet import UNet

_cache = {}
_lock = threading.Lock()


def _unet_from_state_dict(sd):
    first = sd["downs.0.block.0.weight"]
    base, cin = first.shape[0], first.shape[1]
    depth = len({k.split(".")[1] for k in sd if k.startswith("downs.")}) - 1
    cout = sd["head.weight"].shape[0]
    model = UNet(in_channels=cin, out_channels=cout, base=base, depth=depth)
    model.load_state_dict(sd)
    return model


class TorchRunner:
    """Prediction with a PyTorch model: 512x512 RGB in [0, 1] -> (H, W) bool mask."""

    def __init__(self, model, device):
        self.model, self.device = model.eval().to(device), device

    def __call__(self, x_hwc, threshold):
        x = torch.from_numpy(x_hwc).permute(2, 0, 1).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.model(x)
        if out.shape[1] == 1:
            return (torch.sigmoid(out[0, 0]) > threshold).cpu().numpy()
        return out[0].argmax(0).cpu().numpy() > 0


def load_model(path: Path):
    """Returns (runner, info). PyTorch files are read without unpickling
    objects and Keras files in safe mode, so a dropped file cannot run code."""
    key = (str(path), path.stat().st_mtime)
    with _lock:
        if key in _cache:
            return _cache[key]
    info = {"name": path.stem}
    if path.suffix in (".keras", ".h5"):
        from . import tf_backend
        runner, cfg = tf_backend.load(path)
        info.update(kind="Keras model (TensorFlow)", framework="tensorflow", config=cfg)
    else:
        apply_torch_threads()
        device = pick_device()
        if zipfile.is_zipfile(path) and _is_torchscript(path):
            model = torch.jit.load(str(path), map_location=device)
            info["kind"] = "TorchScript"
        else:
            obj = torch.load(str(path), map_location="cpu", weights_only=True)
            if isinstance(obj, dict) and "state_dict" in obj:
                sd = obj["state_dict"]
                info["kind"] = "U-Net (PyTorch)"
            elif isinstance(obj, dict):
                sd = obj
                info["kind"] = "U-Net (PyTorch state_dict)"
            else:
                raise ValueError("Unknown model format")
            model = _unet_from_state_dict(sd)
            info["config"] = model.config
        info["framework"] = "pytorch"
        runner = TorchRunner(model, device)
    with _lock:
        _cache.clear()  # keep one model in memory
        _cache[key] = (runner, info)
    return runner, info


def _is_torchscript(path):
    with zipfile.ZipFile(path) as z:
        return any(n.endswith("/constants.pkl") for n in z.namelist())


def model_summary(path: Path):
    out = {"name": path.stem, "file": path.name, "size_mb": round(path.stat().st_size / 1e6, 1)}
    if path.suffix in (".keras", ".h5"):
        out["framework"] = "tensorflow"
        try:
            meta = json.loads(storage.model_meta_path(path.stem).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            meta = {}
        cfg = meta.get("config")
    else:
        out["framework"] = "pytorch"
        obj = torch.load(str(path), map_location="cpu", weights_only=True)
        meta = obj.get("meta", {}) if isinstance(obj, dict) else {}
        cfg = obj.get("config") if isinstance(obj, dict) else None
    out.update(dataset=meta.get("dataset"), date=meta.get("date"), epoch=meta.get("epoch"),
               best=meta.get("best"), config=cfg)
    return out


def predict(runner, img: Image.Image, threshold: float):
    """Image of any size -> 0/255 mask of the same size, predicted at 512x512."""
    rgb = storage.to_rgb(img)
    small = rgb.resize((storage.SIZE, storage.SIZE), Image.BILINEAR)
    mask = runner(np.asarray(small, dtype=np.float32) / 255.0, threshold)
    m = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
    if m.size != rgb.size:
        m = m.resize(rgb.size, Image.NEAREST)
    return rgb, m, float(mask.mean())


def new_session():
    sid = uuid.uuid4().hex[:12]
    d = storage.INFERENCE / sid
    (d / "imgs").mkdir(parents=True)
    (d / "masks").mkdir()
    return sid


def session_dir(sid: str) -> Path:
    if not sid.isalnum():
        raise ValueError("session")
    d = storage.INFERENCE / sid
    if not d.exists():
        raise FileNotFoundError(sid)
    return d


def run_on_files(sid: str, model_path: Path, threshold: float, files):
    runner, _ = load_model(model_path)
    d = session_dir(sid)
    results = []
    for filename, data in files:
        stem = storage.clean_stem(filename)
        base, n = stem, 1
        while (d / "imgs" / f"{stem}.png").exists():
            n += 1
            stem = f"{base}_{n}"
        img = storage.open_image(data)
        rgb, mask, cover = predict(runner, img, threshold)
        rgb.save(d / "imgs" / f"{stem}.png")
        mask.save(d / "masks" / f"{stem}.png")
        results.append({"name": stem, "width": rgb.width, "height": rgb.height, "coverage": cover})
    return results


def zip_session(sid: str) -> bytes:
    d = session_dir(sid)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for sub in ("imgs", "masks"):
            for p in sorted((d / sub).glob("*.png")):
                z.write(p, f"{sub}/{p.name}")
    return buf.getvalue()
