"""Datasets on disk: data/datasets/<name>/imgs/<item>.png and masks/<item>.png.

Every image is stored as a 512x512 RGB PNG, every mask as a 512x512 PNG
holding 0 (background) or 255 (object).
"""
import io
import re
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

SIZE = 512
ROOT = Path(__file__).resolve().parent.parent / "data"
DATASETS = ROOT / "datasets"
MODELS = ROOT / "models"
INFERENCE = ROOT / "inference"
for d in (DATASETS, MODELS, INFERENCE):
    d.mkdir(parents=True, exist_ok=True)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".gif"}
MASK_SUFFIXES = ("_mask", "_masks", "-mask", "_label", "_labels", "_seg", "_gt", "_annotation")
_SAFE = re.compile(r"^[^/\\:*?\"<>|]+$")


def check_name(name: str) -> str:
    """Refuse names that could leave their folder (.., slashes...)."""
    if not name or name in (".", "..") or not _SAFE.match(name) or name.strip() != name:
        raise ValueError(f"Invalid name: {name!r}")
    return name


def clean_stem(filename: str) -> str:
    stem = Path(filename).stem.strip()
    stem = re.sub(r"[/\\:*?\"<>|]", "_", stem)
    return stem or "image"


def to_rgb(img: Image.Image) -> Image.Image:
    """Any PIL image (16-bit, float, palette, alpha...) to 8-bit RGB."""
    if img.mode in ("I;16", "I;16B", "I;16L", "I", "F"):
        arr = np.asarray(img, dtype=np.float32)
        lo, hi = np.percentile(arr, 0.5), np.percentile(arr, 99.5)
        arr = np.clip((arr - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (0, 0, 0, 255))
        img = Image.alpha_composite(bg, img)
    return img.convert("RGB")


def to_mask(img: Image.Image) -> Image.Image:
    """Any mask image to 0/255 grayscale: every non-black pixel is the object."""
    if img.mode in ("RGBA", "LA"):
        img = img.convert("RGBA")
        arr = np.asarray(img)
        on = (arr[..., :3].max(axis=-1) > 0) & (arr[..., 3] > 0)
    else:
        arr = np.asarray(img.convert("L") if img.mode not in ("I;16", "I", "F") else img)
        on = arr > 0
    return Image.fromarray((on * 255).astype(np.uint8), mode="L")


def open_image(data: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(data))
    img.load()
    return img


def dataset_dir(name: str) -> Path:
    return DATASETS / check_name(name)


def list_datasets():
    out = []
    for d in sorted(DATASETS.iterdir()):
        if not d.is_dir():
            continue
        items = list_items(d.name)
        out.append({
            "name": d.name,
            "count": len(items),
            "annotated": sum(1 for i in items if i["annotated"]),
        })
    return out


def list_items(name: str):
    d = dataset_dir(name)
    imgs = d / "imgs"
    if not imgs.exists():
        return []
    items = []
    for p in sorted(imgs.glob("*.png"), key=lambda p: natural_key(p.stem)):
        m = d / "masks" / p.name
        annotated = False
        if m.exists():
            annotated = bool(np.asarray(Image.open(m)).any())
        items.append({"name": p.stem, "annotated": annotated})
    return items


def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def save_image(name: str, filename: str, data: bytes) -> str:
    d = dataset_dir(name)
    (d / "imgs").mkdir(parents=True, exist_ok=True)
    stem = clean_stem(filename)
    img = to_rgb(open_image(data))
    if img.size != (SIZE, SIZE):
        img = img.resize((SIZE, SIZE), Image.BILINEAR)
    img.save(d / "imgs" / f"{stem}.png")
    return stem


def save_uploaded_mask(name: str, filename: str, data: bytes) -> str:
    """Kept apart until `finalize` pairs it with its image."""
    d = dataset_dir(name)
    (d / "_masks_in").mkdir(parents=True, exist_ok=True)
    stem = clean_stem(filename)
    mask = to_mask(open_image(data))
    if mask.size != (SIZE, SIZE):
        mask = mask.resize((SIZE, SIZE), Image.NEAREST)
    mask.save(d / "_masks_in" / f"{stem}.png")
    return stem


def finalize(name: str):
    """Pair uploaded masks with images (same name, or name + _mask...), and
    give a black mask to every image that has none."""
    d = dataset_dir(name)
    imgs = d / "imgs"
    masks = d / "masks"
    masks.mkdir(parents=True, exist_ok=True)
    incoming = d / "_masks_in"
    stems = {p.stem for p in imgs.glob("*.png")} if imgs.exists() else set()
    paired = orphans = 0
    if incoming.exists():
        lower = {s.lower(): s for s in stems}
        for m in incoming.glob("*.png"):
            target = None
            cand = m.stem
            if cand in stems:
                target = cand
            else:
                low = cand.lower()
                for suf in MASK_SUFFIXES:
                    for c in (low, low.removeprefix("mask_"), low.removeprefix("mask")):
                        if c.endswith(suf) and c[: -len(suf)] in lower:
                            target = lower[c[: -len(suf)]]
                            break
                    if target:
                        break
                if not target and low in lower:
                    target = lower[low]
            if target:
                shutil.move(str(m), masks / f"{target}.png")
                paired += 1
            else:
                orphans += 1
        shutil.rmtree(incoming, ignore_errors=True)
    created = 0
    black = Image.new("L", (SIZE, SIZE), 0)
    for s in stems:
        if not (masks / f"{s}.png").exists():
            black.save(masks / f"{s}.png")
            created += 1
    return {"paired": paired, "orphan_masks": orphans, "black_masks": created, "images": len(stems)}


def write_mask(name: str, item: str, data: bytes):
    d = dataset_dir(name)
    check_name(item)
    if not (d / "imgs" / f"{item}.png").exists():
        raise FileNotFoundError(item)
    mask = to_mask(open_image(data))
    if mask.size != (SIZE, SIZE):
        mask = mask.resize((SIZE, SIZE), Image.NEAREST)
    (d / "masks").mkdir(exist_ok=True)
    mask.save(d / "masks" / f"{item}.png")


def delete_item(name: str, item: str):
    d = dataset_dir(name)
    check_name(item)
    for sub in ("imgs", "masks"):
        (d / sub / f"{item}.png").unlink(missing_ok=True)


def delete_dataset(name: str):
    shutil.rmtree(dataset_dir(name))


def rename_dataset(name: str, new: str):
    src, dst = dataset_dir(name), dataset_dir(new)
    if dst.exists():
        raise FileExistsError(new)
    src.rename(dst)


# ---------------- model files ----------------
# PyTorch models: <name>.pt (or .pth); TensorFlow / Keras: <name>.keras (or .h5)
# with their training information beside them in <name>.json.
MODEL_EXTS = (".pt", ".pth", ".keras", ".h5")


def model_path(name: str) -> Path:
    check_name(name)
    for ext in MODEL_EXTS:
        p = MODELS / f"{name}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(name)


def model_meta_path(name: str) -> Path:
    return MODELS / f"{check_name(name)}.json"


def list_model_files():
    return [p for p in MODELS.iterdir()
            if p.suffix in MODEL_EXTS and ".tmp." not in p.name and ".upload." not in p.name and p.is_file()]


def remove_other_model_files(name: str, keep: Path):
    """A model name refers to one file: drop the one of the other framework."""
    for ext in MODEL_EXTS:
        p = MODELS / f"{name}{ext}"
        if p != keep:
            p.unlink(missing_ok=True)
    if keep.suffix in (".pt", ".pth"):
        model_meta_path(name).unlink(missing_ok=True)


def delete_model(name: str):
    for ext in MODEL_EXTS + (".json",):
        (MODELS / f"{check_name(name)}{ext}").unlink(missing_ok=True)
