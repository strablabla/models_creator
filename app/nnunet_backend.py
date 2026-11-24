"""nnU-Net v2 in Models creator: 3D training (in a worker process) and prediction.

Training: the annotated volumes of a 3D dataset are written as an nnU-Net raw
dataset (NIfTI) in data/nnunet/, then app/nnunet_worker.py plans, preprocesses
and trains in its own process, whose progress lines fill the Training tab.
data/nnunet/ is the work folder of the current training: emptied at the start
of the next one.

Models: data/models/<name>.nnunet/, a standard nnU-Net trained-model folder
(plans.json, dataset.json, fold_0/checkpoint_best.pth) plus meta.json for this
application. Checkpoints are read with torch.load(weights_only=True).
"""
import io
import json
import os
import queue
import random
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np

from . import settings, storage, volumes

WORK = storage.ROOT / "nnunet"
INFERENCE3D = storage.ROOT / "inference3d"
INFERENCE3D.mkdir(parents=True, exist_ok=True)
PROJECT = Path(__file__).resolve().parent.parent
DATASET_ID = 500
SUFFIX = ".nnunet"
CONFIGURATIONS = ("3d_fullres", "2d", "3d_lowres")
_NO_WINDOW = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0

# nnU-Net reads these when first imported; set them for this process too
# (prediction), torch.compile off (no Triton on Windows, slow start elsewhere)
os.environ.setdefault("nnUNet_raw", str(WORK / "raw"))
os.environ.setdefault("nnUNet_preprocessed", str(WORK / "preprocessed"))
os.environ.setdefault("nnUNet_results", str(WORK / "results"))
os.environ.setdefault("nnUNet_compile", "false")


def installed() -> bool:
    from importlib.util import find_spec
    return find_spec("nnunetv2") is not None


def version():
    from importlib.metadata import PackageNotFoundError, version as v
    try:
        return v("nnunetv2")
    except PackageNotFoundError:
        return None


# ---------------- model folders ----------------

def model_dir(name: str) -> Path:
    return storage.MODELS / f"{storage.check_name(name)}{SUFFIX}"


def is_model_dir(p: Path) -> bool:
    return p.is_dir() and p.name.endswith(SUFFIX) and (p / "plans.json").exists()


def read_model_meta(d: Path) -> dict:
    try:
        return json.loads((d / "meta.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def write_model_meta(d: Path, meta: dict):
    tmp = d / "meta.tmp"
    tmp.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    tmp.replace(d / "meta.json")


def checkpoints(d: Path):
    """[(fold folder, checkpoint file)]: checkpoint_final.pth when there is one
    (a model trained to the end elsewhere), else checkpoint_best.pth."""
    out = []
    for f in sorted(d.glob("fold_*")):
        for name in ("checkpoint_final.pth", "checkpoint_best.pth", "checkpoint_latest.pth"):
            if (f / name).exists():
                out.append(f / name)
                break
    return out


def class_list(d: Path) -> list:
    """[{name, color}] for labels 1..N: from meta.json, else dataset.json."""
    meta = read_model_meta(d)
    if meta.get("classes"):
        return meta["classes"]
    try:
        labels = json.loads((d / "dataset.json").read_text(encoding="utf-8"))["labels"]
    except (OSError, ValueError, KeyError):
        return []
    pairs = sorted((v if isinstance(v, int) else max(v), k) for k, v in labels.items()
                   if k not in ("background", "ignore"))
    out = []
    for value, name in pairs:
        while len(out) < value - 1:
            out.append(volumes.default_class(len(out) + 1))
        out.append({"name": name, "color": volumes.default_class(value)["color"]})
    return out


def model_summary(d: Path) -> dict:
    meta = read_model_meta(d)
    size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
    plans = {}
    try:
        plans = json.loads((d / "plans.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    cfg_name = meta.get("configuration")
    if not cfg_name:
        ck = checkpoints(d)
        cfg_name = "?" if not ck else _load_checkpoint(ck[0]).get("init_args", {}).get("configuration", "?")
    conf = plans.get("configurations", {}).get(cfg_name, {})
    return {
        "name": d.name[: -len(SUFFIX)], "file": d.name, "framework": "nnunet",
        "size_mb": round(size / 1e6, 1), "dataset": meta.get("dataset"), "date": meta.get("date"),
        "epoch": meta.get("epoch"), "best": meta.get("best"), "validation": meta.get("validation"),
        "config": {"configuration": cfg_name, "patch_size": conf.get("patch_size"),
                   "spacing": conf.get("spacing"), "folds": len(checkpoints(d))},
        "classes": class_list(d), "kind": "nnU-Net",
    }


def zip_model(d: Path) -> Path:
    out = WORK / "download"
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.zip"):
        old.unlink(missing_ok=True)
    z = out / f"{d.name[: -len(SUFFIX)]}.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(d.rglob("*")):
            if f.is_file():
                zf.write(f, f.relative_to(d).as_posix())
    return z


def import_zip(stem: str, src: Path) -> dict:
    """A zipped nnU-Net trained-model folder (from this application, or
    nnUNetv2_export_model_to_zip: the first folder holding plans.json)."""
    tmp = Path(tempfile.mkdtemp(dir=storage.MODELS, prefix=".upload_"))
    unpacked = tmp / "zip"
    try:
        with zipfile.ZipFile(src) as zf:
            for m in zf.infolist():
                target = (unpacked / m.filename).resolve()
                if not str(target).startswith(str(unpacked.resolve())):
                    raise ValueError("unsafe path in the zip")
                if m.is_dir():
                    continue
                if not (m.filename.endswith((".json", ".pth", ".txt", ".png", ".pdf"))):
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(m) as fi, open(target, "wb") as fo:
                    shutil.copyfileobj(fi, fo)
        roots = sorted((p.parent for p in unpacked.rglob("plans.json")), key=lambda p: len(p.parts))
        roots = [r for r in roots if (r / "dataset.json").exists() and checkpoints(r)]
        if not roots:
            raise ValueError("no nnU-Net model in this zip (plans.json, dataset.json and fold_X/checkpoint_*.pth)")
        dest = model_dir(stem)
        staged = tmp / "_model"
        shutil.move(str(roots[0]), staged)
        load_predictor(staged, mirroring=False)  # refuses what is not a readable model
        shutil.rmtree(dest, ignore_errors=True)
        shutil.move(str(staged), dest)
        storage.remove_other_model_files(stem, keep=dest)
        return model_summary(dest)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------- training ----------------

_proc = {"p": None}


def kill_running():
    """End the worker and its data augmentation processes."""
    p = _proc["p"]
    if p is None or p.poll() is not None:
        return
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(p.pid), "/T", "/F"], capture_output=True,
                           creationflags=_NO_WINDOW, timeout=20)
        else:
            os.killpg(p.pid, signal.SIGTERM)
            try:
                p.wait(5)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
    except Exception:
        p.kill()


def _write_raw(cfg, items, ds_dir, classes):
    """imagesTr / labelsTr / dataset.json of Dataset500_<model>; returns the
    dataset name and {case: volume name}."""
    import SimpleITK as sitk
    name = f"Dataset{DATASET_ID:03d}_" + "".join(c if c.isalnum() else "_" for c in cfg["model_name"])[:60]
    raw = WORK / "raw" / name
    (raw / "imagesTr").mkdir(parents=True)
    (raw / "labelsTr").mkdir()
    n_classes = len(classes)
    for it in items:
        n_classes = max(n_classes, int(np.load(ds_dir / "labels" / f"{it['name']}.npy", mmap_mode="r").max()))
    ignore = n_classes + 1 if cfg.get("sparse") else None
    cases = {}
    for i, it in enumerate(items):
        case = f"case_{i:03d}"
        cases[case] = it["name"]
        img = np.load(ds_dir / "imgs" / f"{it['name']}.npy")
        lab = np.load(ds_dir / "labels" / f"{it['name']}.npy").copy()
        if ignore is not None:
            empty = lab.reshape(lab.shape[0], -1).max(axis=1) == 0
            lab[empty] = ignore
        spacing = [float(s) for s in it.get("spacing", [1, 1, 1])][::-1]  # SimpleITK: x, y, z
        for arr, path in ((img, raw / "imagesTr" / f"{case}_0000.nii.gz"), (lab, raw / "labelsTr" / f"{case}.nii.gz")):
            im = sitk.GetImageFromArray(arr)
            im.SetSpacing(spacing)
            sitk.WriteImage(im, str(path), useCompression=True)
    labels = {"background": 0}
    for k in range(1, n_classes + 1):
        cname = classes[k - 1]["name"] if k <= len(classes) else f"Class {k}"
        while cname in labels:
            cname += "_"
        labels[cname] = k
    if ignore is not None:
        labels["ignore"] = ignore
    (raw / "dataset.json").write_text(json.dumps({
        "name": cfg["model_name"], "channel_names": {"0": "image"}, "labels": labels,
        "numTraining": len(items), "file_ending": ".nii.gz",
    }, indent=2), encoding="utf-8")
    return name, cases


def _split(names, val_split, seed):
    names = list(names)
    random.Random(seed).shuffle(names)
    if len(names) < 2:
        return names, names
    n_val = min(max(int(round(len(names) * val_split)), 1), len(names) - 1)
    return names[n_val:], names[:n_val]


def _n_proc_da(device: str) -> int:
    cpus = os.cpu_count() or 2
    return max(1, min(12, cpus - 1)) if device == "cuda" else max(1, min(4, cpus // 3))


def train(trainer, cfg):
    """Runs in the Trainer's thread; fills trainer.status like the U-Net
    trainings, plus 'phase' and 'log'."""
    from .training import pick_device
    if not installed():
        raise RuntimeError("nnU-Net is not installed (run install.cmd again)")
    ds_dir = volumes.dataset_dir(cfg["dataset"])
    items = volumes.list_items(cfg["dataset"])
    if cfg.get("only_annotated", True):
        items = [i for i in items if i["annotated"]]
    if not items:
        raise RuntimeError("No annotated volume in this dataset")
    classes = volumes.read_meta(ds_dir)["classes"]

    trainer.set(message="Writing the volumes for nnU-Net…", phase="dataset")
    shutil.rmtree(WORK, ignore_errors=True)
    (WORK / "preprocessed").mkdir(parents=True)
    (WORK / "results").mkdir()
    name, cases = _write_raw(cfg, items, ds_dir, classes)
    by_item = {v: k for k, v in cases.items()}
    tr, va = _split([i["name"] for i in items], cfg["val_split"], cfg.get("seed", 0))
    device = pick_device().type
    export_dir = WORK / "export"
    job = {
        "dataset_id": DATASET_ID, "dataset_name": name, "configuration": cfg["configuration"],
        "epochs": cfg["epochs"], "iterations": cfg["iterations"], "val_iterations": cfg["val_iterations"],
        "device": device, "threads": settings.get()["threads"], "n_proc": _n_proc_da(device),
        "splits": [{"train": [by_item[n] for n in tr], "val": [by_item[n] for n in va]}],
        "export_dir": str(export_dir), "final_validation": cfg.get("final_validation", True),
        "parent_pid": os.getpid(),
    }
    job_path = WORK / "job.json"
    job_path.write_text(json.dumps(job, indent=2), encoding="utf-8")
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", nnUNet_n_proc_DA=str(job["n_proc"]),
               nnUNet_raw=str(WORK / "raw"), nnUNet_preprocessed=str(WORK / "preprocessed"),
               nnUNet_results=str(WORK / "results"), nnUNet_compile="false")
    import torch
    trainer.set(state="running", device=f"nnU-Net {version()} · PyTorch {torch.__version__} · {device}",
                n_train=len(tr), n_val=len(va), batches=cfg["iterations"] + cfg["val_iterations"],
                message=f"{len(tr)} training volumes, {len(va)} validation volumes", log=[])
    dest = model_dir(cfg["model_name"])
    meta = {"framework": "nnunet", "dataset": cfg["dataset"], "train_config": cfg,
            "configuration": cfg["configuration"], "classes": classes, "cases": cases,
            "split": {"train": tr, "val": va}, "ignore_label": bool(cfg.get("sparse"))}

    p = subprocess.Popen([sys.executable, "-m", "app.nnunet_worker", str(job_path)], cwd=PROJECT, env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                         errors="replace", bufsize=1, creationflags=_NO_WINDOW,
                         start_new_session=os.name != "nt")
    _proc["p"] = p
    trainer.on_stop = kill_running
    lines = queue.Queue()

    def read():
        for l in p.stdout:
            lines.put(l)
        lines.put(None)  # end of the output: the worker has ended

    threading.Thread(target=read, daemon=True).start()
    log_file = open(WORK / "train.log", "w", encoding="utf-8")
    error, tail = None, []
    try:
        while True:
            try:
                line = lines.get(timeout=0.5)
            except queue.Empty:
                if trainer.stopping:
                    kill_running()
                continue
            if line is None:
                break
            line = line.rstrip("\n")
            log_file.write(line + "\n")
            if not line.startswith("@@"):
                if line.strip():
                    tail = (tail + [line])[-300:]
                    trainer.set(log=list(tail))
                continue
            ev = json.loads(line[2:])
            if "phase" in ev:
                trainer.set(phase=ev["phase"], **({"message": ev["message"]} if ev.get("message") else {}))
                if ev.get("plan"):
                    meta["plan"] = ev["plan"]
                    trainer.set(plan=ev["plan"])
            if "progress" in ev:
                pr = ev["progress"]
                trainer.set(epoch=pr["epoch"], batch=pr["batch"], batches=pr["batches"],
                            **({"batch_loss": pr["loss"]} if pr.get("loss") is not None else {}))
            if "epoch" in ev:
                row = ev["epoch"]
                with trainer.lock:
                    trainer.status["history"].append(row)
                    history = list(trainer.status["history"])
                if ev.get("best"):
                    shutil.rmtree(dest, ignore_errors=True)
                    shutil.copytree(export_dir, dest)
                    storage.remove_other_model_files(cfg["model_name"], keep=dest)
                    meta.update(best=row, epoch=row["epoch"], history=history,
                                date=datetime.now().isoformat(timespec="seconds"))
                    write_model_meta(dest, meta)
                    trainer.set(best=row)
                trainer.set(message=f"Epoch {row['epoch']} in {row['seconds']:.0f} s · pseudo Dice "
                                    f"{row['val']['dice']:.3f} (EMA {row['val']['ema_dice']:.3f})")
            if "validation" in ev:
                meta["validation"] = ev["validation"]
                trainer.set(validation=ev["validation"])
            if "error" in ev:
                error = ev["error"]
        p.wait()
    finally:
        log_file.close()
        _proc["p"] = None
        trainer.on_stop = None
    if dest.exists() and meta.get("best"):
        with trainer.lock:
            meta["history"] = list(trainer.status["history"])
        for f in export_dir.glob("training_log*"):
            shutil.copy(f, dest)
        if (export_dir / "progress.png").exists():
            shutil.copy(export_dir / "progress.png", dest)
        write_model_meta(dest, meta)
    best = trainer.status.get("best")
    if trainer.stopping:
        trainer.set(state="stopped", message=(f"Stopped: best model (epoch {best['epoch']}) saved as {dest.name}"
                                              if best else "Stopped before the end of the first epoch: no model saved"))
        return
    if p.returncode != 0:
        raise RuntimeError(error or (tail[-1] if tail else f"nnU-Net stopped with code {p.returncode}"))
    v = meta.get("validation") or {}
    trainer.set(state="done", message=f"Done: best model (epoch {best['epoch']}) saved as {dest.name}"
                + (f" · validation Dice {v['dice']:.3f}" if v.get("dice") is not None else ""))


# ---------------- prediction ----------------

_cache = {}
_cache_lock = threading.Lock()


def _load_checkpoint(path: Path) -> dict:
    """Without running code: plain tensors and containers, plus the numpy
    scalars that checkpoints from nnU-Net itself hold in their logs."""
    import torch
    try:
        return torch.load(str(path), map_location="cpu", weights_only=True)
    except Exception:
        np_core = getattr(np, "_core", None) or np.core
        allowed = [np_core.multiarray.scalar, np.dtype] + [type(np.dtype(t)) for t in
                                                            ("float64", "float32", "int64", "int32", "bool")]
        with torch.serialization.safe_globals(allowed):
            return torch.load(str(path), map_location="cpu", weights_only=True)


def load_predictor(d: Path, mirroring: bool):
    from .training import apply_torch_threads, pick_device
    ck = checkpoints(d)
    if not ck:
        raise ValueError("no checkpoint in this model")
    key = (str(d), max(c.stat().st_mtime for c in ck), mirroring)
    with _cache_lock:
        if key in _cache:
            return _cache[key]
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    apply_torch_threads()
    dataset_json = json.loads((d / "dataset.json").read_text(encoding="utf-8"))
    pm = PlansManager(json.loads((d / "plans.json").read_text(encoding="utf-8")))
    params, first = [], None
    for c in ck:
        obj = _load_checkpoint(c)
        first = first or obj
        params.append(obj["network_weights"])
    conf_name = first["init_args"]["configuration"]
    cm = pm.get_configuration(conf_name)
    n_in = determine_num_input_channels(pm, cm, dataset_json)
    if n_in != 1:
        raise ValueError(f"this model takes {n_in} channels per voxel; Models creator gives it one (grayscale)")
    trainer_name = first.get("trainer_name", "nnUNetTrainer")
    try:
        from nnunetv2.utilities.find_objects import recursive_find_trainer_class_by_name
        trainer_class = recursive_find_trainer_class_by_name(trainer_name)
    except Exception:
        trainer_class = None
    if trainer_class is None:
        from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer as trainer_class
    lm = pm.get_label_manager(dataset_json)
    net = trainer_class.build_network_architecture(pm, cm, n_in, lm.num_segmentation_heads,
                                                   enable_deep_supervision=False)
    net.load_state_dict(params[0])
    device = pick_device()
    predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=mirroring,
                                perform_everything_on_device=device.type == "cuda", device=device,
                                verbose=False, allow_tqdm=False)
    axes = first.get("inference_allowed_mirroring_axes")
    predictor.manual_initialization(net, pm, cm, params, dataset_json, trainer_name,
                                    tuple(axes) if axes is not None else None)
    info = {"configuration": conf_name, "folds": len(ck), "device": device.type, "classes": class_list(d)}
    with _cache_lock:
        _cache.clear()  # one model in memory
        _cache[key] = (predictor, info)
    return predictor, info


def predict_volume(predictor, arr: np.ndarray, spacing) -> np.ndarray:
    seg = predictor.predict_single_npy_array(arr[None].astype(np.float32), {"spacing": [float(s) for s in spacing]})
    return np.asarray(seg).astype(np.uint8).reshape(arr.shape)


# ---------------- 3D inference sessions ----------------
# data/inference3d/<id>/ has the layout of a 3D dataset (imgs/, labels/,
# meta.json), so the viewer reads both the same way; meta.json also holds the
# state of each volume and the model used.

_jobs = queue.Queue()
_meta_lock = threading.Lock()


def new_session() -> str:
    sid = uuid.uuid4().hex[:12]
    (INFERENCE3D / sid / "imgs").mkdir(parents=True)
    (INFERENCE3D / sid / "labels").mkdir()
    volumes.write_meta(INFERENCE3D / sid, {"classes": [], "items": {}})
    return sid


def session_dir(sid: str) -> Path:
    if not sid.isalnum():
        raise ValueError("session")
    d = INFERENCE3D / sid
    if not d.exists():
        raise FileNotFoundError(sid)
    return d


def _update_item(d: Path, item: str, **kw):
    with _meta_lock:
        meta = volumes.read_meta(d)
        meta["items"].setdefault(item, {}).update(kw)
        volumes.write_meta(d, meta)


def stage_file(sid: str, rel: str, data: bytes):
    volumes.stage_file(session_dir(sid) / "_incoming", rel, data)


def run_session(sid: str, model: str, mirroring: bool, again: bool):
    """Volumes staged since the last call (and, with again, those already
    there) are queued for prediction with this model."""
    d = session_dir(sid)
    md = model_dir(model)
    if not is_model_dir(md):
        raise FileNotFoundError(model)
    inc = d / "_incoming"
    try:
        vols, errors = volumes.collect(inc, "volume")
    finally:
        shutil.rmtree(inc, ignore_errors=True)
    queued = []
    with _meta_lock:
        meta = volumes.read_meta(d)
        meta["classes"] = class_list(md)
        meta["model"] = model
        for n, (arr, spacing) in vols.items():
            stem, k = storage.clean_stem(n), 1
            base = stem
            while (d / "imgs" / f"{stem}.npy").exists():
                k += 1
                stem = f"{base}_{k}"
            if arr.dtype.kind not in "uif" or arr.dtype.itemsize > 4:
                arr = arr.astype(np.float32)
            np.save(d / "imgs" / f"{stem}.npy", arr)
            meta["items"][stem] = {"spacing": spacing}
            queued.append(stem)
        if again:
            queued += [n for n in meta["items"] if n not in queued]
        for n in queued:
            meta["items"][n].update(status="queued", model=model, error=None)
            (d / "labels" / f"{n}.npy").unlink(missing_ok=True)
        volumes.write_meta(d, meta)
    for n in queued:
        _jobs.put((sid, n, model, mirroring))
    _ensure_worker()
    return {"queued": queued, "errors": errors}


_worker = {"t": None}


def _ensure_worker():
    if _worker["t"] is None or not _worker["t"].is_alive():
        _worker["t"] = threading.Thread(target=_work, daemon=True)
        _worker["t"].start()


def _work():
    while True:
        sid, item, model, mirroring = _jobs.get()
        try:
            d = session_dir(sid)
        except (ValueError, FileNotFoundError):
            continue  # session cleared
        meta = volumes.read_meta(d)
        if meta["items"].get(item, {}).get("model") != model:
            continue  # queued again with another model since
        _update_item(d, item, status="running", started=time.time())
        try:
            predictor, _ = load_predictor(model_dir(model), mirroring)
            arr = np.load(d / "imgs" / f"{item}.npy")
            t0 = time.time()
            seg = predict_volume(predictor, arr, meta["items"][item].get("spacing", [1, 1, 1]))
            if not (d / "imgs" / f"{item}.npy").exists():
                continue
            np.save(d / "labels" / f"{item}.npy", seg)
            counts = np.bincount(seg.ravel(), minlength=int(seg.max()) + 1)
            _update_item(d, item, status="done", seconds=round(time.time() - t0, 1),
                         coverage=[float(c) / seg.size for c in counts[1:]])
        except Exception as e:
            try:
                _update_item(d, item, status="error", error=str(e))
            except Exception:
                pass


def session_state(sid: str) -> dict:
    d = session_dir(sid)
    meta = volumes.read_meta(d)
    items = []
    for it in volumes.list_items_in(d):
        m = meta["items"].get(it["name"], {})
        items.append({**it, "status": m.get("status"), "error": m.get("error"), "seconds": m.get("seconds"),
                      "coverage": m.get("coverage"), "started": m.get("started")})
    return {"items": items, "classes": meta["classes"], "model": meta.get("model")}


def zip_session(sid: str) -> bytes:
    """labels/<volume>.tif (uint8, one value per class) for every predicted volume."""
    import tifffile
    d = session_dir(sid)
    meta = volumes.read_meta(d)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted((d / "labels").glob("*.npy")):
            sp = meta["items"].get(p.stem, {}).get("spacing", [1, 1, 1])
            b = io.BytesIO()
            tifffile.imwrite(b, np.load(p), imagej=True, resolution=(1 / sp[2], 1 / sp[1]),
                             metadata={"spacing": sp[0], "axes": "ZYX"})
            z.writestr(f"labels/{p.stem}.tif", b.getvalue())
        names = ", ".join(f"{i} = {c['name']}" for i, c in enumerate(meta["classes"], 1))
        z.writestr("classes.txt", f"0 = background, {names}\n")
    return buf.getvalue()


def session_to_dataset(sid: str, dataset: str) -> int:
    d = session_dir(sid)
    meta = volumes.read_meta(d)
    ds = volumes.dataset_dir(dataset)
    n = 0
    for p in sorted((d / "labels").glob("*.npy")):
        img = np.load(d / "imgs" / p.name)
        volumes.add_volume(ds.name, p.stem, img, np.load(p), meta["items"].get(p.stem, {}).get("spacing", [1, 1, 1]),
                           meta["classes"])
        n += 1
    return n
