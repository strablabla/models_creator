"""FastAPI app: datasets & annotation, training, inference.

Started by launch.py (the Desktop icon), or: uv run launch.py --console
"""
import io
import shutil
from pathlib import Path
from typing import List

from fastapi import Body, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field

from . import inference, nnunet_backend, settings, storage, sysinfo, volumes
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
    if volumes.dataset_dir(name).exists():
        raise HTTPException(409, f"A 3D dataset is already named {name!r}")
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


# ---------------- 3D datasets (nnU-Net) ----------------

def _volume_response(arr):
    return Response(volumes.packed(arr), media_type="application/octet-stream",
                    headers={**NO_CACHE, "Content-Encoding": "gzip", "X-Shape": ",".join(map(str, arr.shape))})


@app.get("/api/volumes")
def volume_datasets():
    return volumes.list_datasets()


@app.post("/api/volumes/{name}/files")
async def upload_volume_files(name: str, files: List[UploadFile] = File(...), kinds: List[str] = Form(...),
                              paths: List[str] = Form(...)):
    """One batch of a dropped folder. kinds[i] is 'img' or 'mask', paths[i]
    the path under imgs/ (or masks/): vol1.tif, or vol1/slice_001.png..."""
    _guard(storage.check_name, name)
    if storage.dataset_dir(name).exists():
        raise HTTPException(409, f"A 2D dataset is already named {name!r}")
    if not (len(kinds) == len(files) == len(paths)):
        raise HTTPException(400, "files/kinds/paths")
    errors = []
    for f, kind, rel in zip(files, kinds, paths):
        try:
            volumes.save_upload(name, kind, rel, await f.read())
        except Exception as e:
            errors.append(f"{f.filename}: {e}")
    return {"errors": errors}


@app.post("/api/volumes/{name}/finalize")
def finalize_volumes(name: str):
    return _guard(volumes.finalize, name)


@app.delete("/api/volumes/{name}")
def delete_volume_dataset(name: str):
    _guard(volumes.delete_dataset, name)
    return {"ok": True}


@app.post("/api/volumes/{name}/rename")
def rename_volume_dataset(name: str, new_name: str = Body(..., embed=True)):
    if storage.dataset_dir(new_name).exists():
        raise HTTPException(409, f"A 2D dataset is already named {new_name!r}")
    _guard(volumes.rename_dataset, name, new_name)
    return {"ok": True}


@app.get("/api/volumes/{name}/items")
def volume_items(name: str):
    d = _guard(volumes.dataset_dir, name)
    return {"items": _guard(volumes.list_items, name), "classes": volumes.read_meta(d)["classes"]}


@app.put("/api/volumes/{name}/classes")
def volume_classes(name: str, classes: list = Body(..., embed=True)):
    return _guard(volumes.set_classes, name, classes)


@app.get("/api/volumes/{name}/image/{item}")
def volume_image(name: str, item: str):
    d = _guard(volumes.dataset_dir, name)
    return _volume_response(_guard(volumes.display_volume, d, item))


@app.get("/api/volumes/{name}/labels/{item}")
def volume_labels(name: str, item: str):
    d = _guard(volumes.dataset_dir, name)
    return _volume_response(_guard(volumes.labels, d, item))


@app.get("/api/volumes/{name}/mesh/{item}")
def volume_mesh(name: str, item: str):
    """Surfaces of the saved labels, for the 3D view."""
    d = _guard(volumes.dataset_dir, name)
    return _guard(volumes.meshes, d, item)


@app.put("/api/volumes/{name}/labels/{item}")
async def put_volume_labels(name: str, item: str, request: Request):
    body = await request.body()
    _guard(volumes.write_labels, name, item, body, request.headers.get("content-encoding"))
    return {"ok": True}


@app.get("/api/volumes/{name}/thumb/{item}")
def volume_thumb(name: str, item: str, size: int = 128):
    d = _guard(volumes.dataset_dir, name)
    return Response(_guard(volumes.thumbnail, d, item, size), media_type="image/png", headers=NO_CACHE)


@app.delete("/api/volumes/{name}/items/{item}")
def delete_volume_item(name: str, item: str):
    _guard(volumes.delete_item, name, item)
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
    framework: str | None = None  # default: the one chosen in Settings; "nnunet" for a 3D dataset
    # nnU-Net only
    configuration: str = "3d_fullres"
    iterations: int = Field(250, ge=1, le=10000)
    val_iterations: int = Field(50, ge=1, le=10000)
    sparse: bool = False           # slices without any label are ignored, not background
    final_validation: bool = True  # Dice on the whole validation volumes at the end


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
        if p.suffix == storage.NNUNET:
            try:
                out.append(nnunet_backend.model_summary(p))
            except Exception:
                out.append({"name": p.stem, "file": p.name, "framework": "nnunet", "size_mb": None})
            continue
        try:
            out.append(inference.model_summary(p))
        except Exception:
            out.append({"name": p.stem, "file": p.name, "size_mb": round(p.stat().st_size / 1e6, 1),
                        "framework": "tensorflow" if p.suffix in (".keras", ".h5") else "pytorch"})
    return out


@app.post("/api/models")
async def upload_model(file: UploadFile = File(...)):
    """A PyTorch (.pt, .pth) or Keras (.keras, .h5) model, or a zipped nnU-Net
    model folder, checked by loading it."""
    stem = storage.clean_stem(file.filename)
    ext = Path(file.filename).suffix.lower()
    if ext == ".zip":
        tmp = storage.MODELS / f"{stem}.upload.zip"
        with tmp.open("wb") as fh:
            shutil.copyfileobj(file.file, fh)
        try:
            s = nnunet_backend.import_zip(stem, tmp)
        except Exception as e:
            raise HTTPException(400, f"This file is not a readable nnU-Net model: {e}")
        finally:
            tmp.unlink(missing_ok=True)
        return {"name": stem, "kind": "nnU-Net", "framework": "nnunet", "summary": s}
    if ext not in storage.MODEL_EXTS:
        raise HTTPException(400, "A .pt, .pth, .keras, .h5 file, or a .zip of an nnU-Net model, is needed")
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
    if p.is_dir():
        z = nnunet_backend.zip_model(p)
        return FileResponse(z, filename=z.name, media_type="application/zip")
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


# ---------------- 3D inference (nnU-Net) ----------------

@app.post("/api/inference3d/sessions")
def new_session3d():
    return {"id": nnunet_backend.new_session()}


@app.post("/api/inference3d/sessions/{sid}/files")
async def session3d_files(sid: str, files: List[UploadFile] = File(...), paths: List[str] = Form(...)):
    if len(paths) != len(files):
        raise HTTPException(400, "files/paths")
    for f, rel in zip(files, paths):
        _guard(nnunet_backend.stage_file, sid, rel, await f.read())
    return {"ok": True}


@app.post("/api/inference3d/sessions/{sid}/run")
def session3d_run(sid: str, model: str = Body(...), mirroring: bool = Body(False), again: bool = Body(False)):
    return _guard(nnunet_backend.run_session, sid, model, mirroring, again)


@app.get("/api/inference3d/sessions/{sid}")
def session3d_state(sid: str):
    return _guard(nnunet_backend.session_state, sid)


@app.get("/api/inference3d/sessions/{sid}/image/{item}")
def session3d_image(sid: str, item: str):
    d = _guard(nnunet_backend.session_dir, sid)
    return _volume_response(_guard(volumes.display_volume, d, item))


@app.get("/api/inference3d/sessions/{sid}/labels/{item}")
def session3d_labels(sid: str, item: str):
    d = _guard(nnunet_backend.session_dir, sid)
    return _volume_response(_guard(volumes.labels, d, item))


@app.get("/api/inference3d/sessions/{sid}/mesh/{item}")
def session3d_mesh(sid: str, item: str):
    d = _guard(nnunet_backend.session_dir, sid)
    return _guard(volumes.meshes, d, item)


@app.get("/api/inference3d/sessions/{sid}/thumb/{item}")
def session3d_thumb(sid: str, item: str, size: int = 128):
    d = _guard(nnunet_backend.session_dir, sid)
    return Response(_guard(volumes.thumbnail, d, item, size), media_type="image/png", headers=NO_CACHE)


@app.get("/api/inference3d/sessions/{sid}/zip")
def session3d_zip(sid: str):
    data = _guard(nnunet_backend.zip_session, sid)
    return Response(data, media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="predictions3d_{sid}.zip"'})


@app.post("/api/inference3d/sessions/{sid}/to_dataset")
def session3d_to_dataset(sid: str, dataset: str = Body(..., embed=True)):
    """Predicted labels become annotations to correct in Annotation 3D."""
    _guard(storage.check_name, dataset)
    if storage.dataset_dir(dataset).exists():
        raise HTTPException(409, f"{dataset!r} is a 2D dataset")
    return {"added": _guard(nnunet_backend.session_to_dataset, sid, dataset)}
