"""FastAPI app: datasets & annotation, training, inference.

Started by launch.py (the Desktop icon), or: uv run launch.py --console
"""
import io
import shutil
from pathlib import Path
from typing import List

from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field

from . import inference, settings, storage, sysinfo
from .training import pick_device, trainer

STATIC = Path(__file__).resolve().parent.parent / "static"
app = FastAPI(title="Models creator")
app.mount("/static", StaticFiles(directory=STATIC), name="static")

NO_CACHE = {"Cache-Control": "no-store"}


def _guard(fn, *a):
    try:
        return fn(*a)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except FileNotFoundError as e:
        raise HTTPException(404, f"Not found: {e}")
    except FileExistsError as e:
        raise HTTPException(409, f"Already exists: {e}")


def _png(path: Path, size: int | None = None):
    if not path.exists():
        raise HTTPException(404, "Not found")
    if not size:
        return FileResponse(path, media_type="image/png", headers=NO_CACHE)
    img = Image.open(path)
    img.thumbnail((size, size))
    buf = io.BytesIO()
    if img.mode == "L":
        img.save(buf, "PNG")
        return Response(buf.getvalue(), media_type="image/png", headers=NO_CACHE)
    img.convert("RGB").save(buf, "JPEG", quality=80)
    return Response(buf.getvalue(), media_type="image/jpeg", headers=NO_CACHE)


@app.get("/", response_class=HTMLResponse)
def index():
    return (STATIC / "index.html").read_text(encoding="utf-8")


@app.get("/api/info")
def info():
    """For the header: framework of new trainings and where it computes."""
    fw = settings.get()["framework"]
    if fw == "tensorflow":
        import os, sys
        if "app.tf_backend" in sys.modules:
            device = sys.modules["app.tf_backend"].info()["device_used"]
        else:  # no GPU for TensorFlow on native Windows; elsewhere, unknown until loaded
            device = "CPU" if os.name == "nt" else None
    else:
        device = "GPU" if pick_device().type == "cuda" else "CPU"
    return {"framework": fw, "device": device, "size": storage.SIZE}


@app.get("/api/system")
def system_info():
    return sysinfo.everything()


@app.get("/api/system/live")
def system_live():
    """Polled by the Info tab: GPU load, VRAM in use, free RAM."""
    return {"gpus": sysinfo.nvidia_gpus(), "system": sysinfo.system()}


# ---------------- datasets ----------------

@app.get("/api/datasets")
def datasets():
    return storage.list_datasets()


@app.post("/api/datasets/{name}/files")
async def upload_files(name: str, files: List[UploadFile] = File(...), kinds: List[str] = Form(...)):
    """One batch of a dropped folder. kinds[i] is 'img' or 'mask'."""
    _guard(storage.check_name, name)
    if len(kinds) != len(files):
        raise HTTPException(400, "files/kinds")
    saved, errors = [], []
    for f, kind in zip(files, kinds):
        data = await f.read()
        try:
            if kind == "mask":
                storage.save_uploaded_mask(name, f.filename, data)
            else:
                saved.append(storage.save_image(name, f.filename, data))
        except Exception as e:
            errors.append(f"{f.filename} : {e}")
    return {"saved": saved, "errors": errors}


@app.post("/api/datasets/{name}/finalize")
def finalize(name: str):
    return _guard(storage.finalize, name)


@app.delete("/api/datasets/{name}")
def delete_dataset(name: str):
    _guard(storage.delete_dataset, name)
    return {"ok": True}


@app.post("/api/datasets/{name}/rename")
def rename_dataset(name: str, new_name: str = Body(..., embed=True)):
    _guard(storage.rename_dataset, name, new_name)
    return {"ok": True}


@app.get("/api/datasets/{name}/items")
def items(name: str):
    return _guard(storage.list_items, name)


@app.get("/api/datasets/{name}/image/{item}")
def get_image(name: str, item: str, size: int | None = None):
    d = _guard(storage.dataset_dir, name)
    _guard(storage.check_name, item)
    return _png(d / "imgs" / f"{item}.png", size)


@app.get("/api/datasets/{name}/mask/{item}")
def get_mask(name: str, item: str, size: int | None = None):
    d = _guard(storage.dataset_dir, name)
    _guard(storage.check_name, item)
    p = d / "masks" / f"{item}.png"
    if not p.exists() and (d / "imgs" / f"{item}.png").exists():
        Image.new("L", (storage.SIZE, storage.SIZE), 0).save(p)
    return _png(p, size)


@app.put("/api/datasets/{name}/mask/{item}")
async def put_mask(name: str, item: str, file: UploadFile = File(...)):
    _guard(storage.write_mask, name, item, await file.read())
    return {"ok": True}


@app.delete("/api/datasets/{name}/items/{item}")
def delete_item(name: str, item: str):
    _guard(storage.delete_item, name, item)
    return {"ok": True}


# ---------------- training ----------------

class TrainConfig(BaseModel):
    dataset: str
    model_name: str
    epochs: int = Field(30, ge=1, le=5000)
    batch_size: int = Field(4, ge=1, le=64)
    lr: float = Field(1e-3, gt=0, le=1)
    val_split: float = Field(0.2, ge=0, le=0.9)
    base: int = Field(16, ge=4, le=128)
    depth: int = Field(4, ge=1, le=6)
    augment: bool = True
    only_annotated: bool = True
    seed: int = 0
    framework: str | None = None  # default: the one chosen in Settings


@app.post("/api/train")
def train(cfg: TrainConfig):
    try:
        trainer.start(cfg.model_dump())
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"ok": True}


@app.get("/api/train/status")
def train_status():
    return trainer.snapshot()


@app.post("/api/train/stop")
def train_stop():
    trainer.stop()
    return {"ok": True}


# ---------------- models ----------------

@app.get("/api/models")
def models():
    out = []
    for p in sorted(storage.list_model_files(), key=lambda p: -p.stat().st_mtime):
        try:
            out.append(inference.model_summary(p))
        except Exception:
            out.append({"name": p.stem, "file": p.name, "size_mb": round(p.stat().st_size / 1e6, 1),
                        "framework": "tensorflow" if p.suffix in (".keras", ".h5") else "pytorch"})
    return out


@app.post("/api/models")
async def upload_model(file: UploadFile = File(...)):
    """A PyTorch (.pt, .pth) or Keras (.keras, .h5) model, checked by loading it."""
    stem = storage.clean_stem(file.filename)
    ext = Path(file.filename).suffix.lower()
    if ext not in storage.MODEL_EXTS:
        raise HTTPException(400, "A .pt, .pth, .keras or .h5 file is needed")
    if ext == ".pth":
        ext = ".pt"
    tmp = storage.MODELS / f"{stem}.upload{ext}"
    with tmp.open("wb") as fh:
        shutil.copyfileobj(file.file, fh)
    try:
        _, info_ = inference.load_model(tmp)
    except Exception as e:
        tmp.unlink(missing_ok=True)
        raise HTTPException(400, f"This file is not a readable model: {e}")
    dest = storage.MODELS / f"{stem}{ext}"
    tmp.replace(dest)
    storage.remove_other_model_files(stem, keep=dest)
    return {"name": stem, "kind": info_.get("kind"), "framework": info_.get("framework")}


@app.get("/api/models/{name}/download")
def download_model(name: str):
    p = _guard(storage.model_path, name)
    return FileResponse(p, filename=p.name, media_type="application/octet-stream")


@app.delete("/api/models/{name}")
def delete_model(name: str):
    _guard(storage.delete_model, name)
    return {"ok": True}


# ---------------- settings ----------------

@app.post("/api/shutdown")
def shutdown():
    """The Quit button: answer, then end the process (and with it uv and the
    hidden cmd started by the Desktop shortcut). A running training stops
    there; its best model is already saved after each epoch."""
    import os
    import threading
    trainer.stop()
    threading.Timer(0.5, os._exit, args=(0,)).start()
    return {"ok": True}


@app.get("/api/settings")
def get_settings():
    return settings.get()


@app.put("/api/settings")
def put_settings(changes: dict = Body(...)):
    try:
        new = settings.update(changes)
    except ValueError as e:
        raise HTTPException(400, str(e))
    if "threads" in changes:
        from .training import apply_torch_threads
        apply_torch_threads()
    return new


# ---------------- inference ----------------

@app.post("/api/inference/sessions")
def new_session():
    return {"id": inference.new_session()}


@app.post("/api/inference/sessions/{sid}/predict")
async def predict(sid: str, model: str = Form(...), threshold: float = Form(0.5),
                  files: List[UploadFile] = File(...)):
    try:
        path = storage.model_path(model)
    except (ValueError, FileNotFoundError):
        raise HTTPException(404, "Model not found")
    data = [(f.filename, await f.read()) for f in files]
    try:
        return inference.run_on_files(sid, path, threshold, data)
    except FileNotFoundError:
        raise HTTPException(404, "Session not found")
    except Exception as e:
        raise HTTPException(400, f"Prediction failed: {e}")


@app.get("/api/inference/sessions/{sid}/{kind}/{item}")
def session_file(sid: str, kind: str, item: str, size: int | None = None):
    if kind not in ("imgs", "masks"):
        raise HTTPException(404)
    d = _guard(inference.session_dir, sid)
    _guard(storage.check_name, item)
    return _png(d / kind / f"{item}.png", size)


@app.get("/api/inference/sessions/{sid}/zip")
def session_zip(sid: str):
    data = _guard(inference.zip_session, sid)
    return Response(data, media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="predictions_{sid}.zip"'})


@app.post("/api/inference/sessions/{sid}/to_dataset")
def session_to_dataset(sid: str, dataset: str = Body(..., embed=True)):
    """Predicted masks become annotations to correct (same 512x512 format)."""
    d = _guard(inference.session_dir, sid)
    ds = _guard(storage.dataset_dir, dataset)
    (ds / "imgs").mkdir(parents=True, exist_ok=True)
    (ds / "masks").mkdir(exist_ok=True)
    n = 0
    for p in (d / "imgs").glob("*.png"):
        img = Image.open(p).convert("RGB").resize((storage.SIZE, storage.SIZE), Image.BILINEAR)
        mask = Image.open(d / "masks" / p.name).convert("L").resize((storage.SIZE, storage.SIZE), Image.NEAREST)
        img.save(ds / "imgs" / p.name)
        mask.save(ds / "masks" / p.name)
        n += 1
    return {"added": n}
