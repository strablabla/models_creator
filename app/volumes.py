"""3D datasets on disk, for nnU-Net: data/volumes/<name>/

    imgs/<item>.npy     the volume, (Z, Y, X), in its own type (uint8, uint16, float32...)
    labels/<item>.npy   uint8 labels of the same shape: 0 = background, 1..N = classes
    view/<item>.npy     uint8 copy of the image for display (made when first needed)
    meta.json           {"classes": [{"name", "color"}, ...], "items": {item: {"spacing": [z, y, x]}}}

Volumes come from multi-page TIFF files or from folders of 2D slices. The same
layout serves the 3D inference sessions (data/inference3d/<id>/).
"""
import gzip
import io
import json
import re
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from . import storage
from .storage import check_name, clean_stem, natural_key

VOLUMES = storage.ROOT / "volumes"
VOLUMES.mkdir(parents=True, exist_ok=True)

TIFF_EXTS = {".tif", ".tiff"}
MAX_VOXELS = 1_000_000_000  # what a browser tab can still hold and draw
MAX_CLASSES = 254           # labels are uint8; nnU-Net may add an "ignore" label after them
PALETTE = ["#ff2d55", "#00e0ff", "#ffd23f", "#3ccf91", "#b36bff", "#ff8a00", "#4c8dff",
           "#ff5cf0", "#a8e10c", "#00b3a4", "#ff6b6b", "#c9a227"]


def default_class(i: int) -> dict:
    """Class i (1-based) before the user names it."""
    return {"name": f"Class {i}", "color": PALETTE[(i - 1) % len(PALETTE)]}


def dataset_dir(name: str) -> Path:
    return VOLUMES / check_name(name)


# ---------------- meta ----------------

def read_meta(d: Path) -> dict:
    try:
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        meta = {}
    meta.setdefault("classes", [])
    meta.setdefault("items", {})
    return meta


def write_meta(d: Path, meta: dict):
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / "meta.tmp"
    tmp.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    tmp.replace(d / "meta.json")


def set_classes(name: str, classes: list):
    """Names and colors of classes 1..N, in that order."""
    d = dataset_dir(name)
    if not d.exists():
        raise FileNotFoundError(name)
    if not 1 <= len(classes) <= MAX_CLASSES:
        raise ValueError(f"Between 1 and {MAX_CLASSES} classes")
    out, seen = [], set()
    for i, c in enumerate(classes, 1):
        n = str(c.get("name", "")).strip() or f"Class {i}"
        if n.lower() in seen or n.lower() in ("background", "ignore"):
            raise ValueError(f"Class names must be different, and not 'background' or 'ignore': {n!r}")
        seen.add(n.lower())
        color = str(c.get("color", ""))
        if not (len(color) == 7 and color.startswith("#")):
            color = default_class(i)["color"]
        out.append({"name": n, "color": color})
    meta = read_meta(d)
    meta["classes"] = out
    write_meta(d, meta)
    return out


# ---------------- reading volumes ----------------

def _gray(arr: np.ndarray, axes: str) -> tuple[np.ndarray, str]:
    """Drop the color / channel / time axes of an image: mean of the colors
    or channels, first time point."""
    for ax in ("T",):
        if ax in axes:
            arr = np.take(arr, 0, axis=axes.index(ax))
            axes = axes.replace(ax, "")
    for ax in ("S", "C"):
        if ax in axes:
            i = axes.index(ax)
            dtype = arr.dtype
            arr = arr.mean(axis=i)
            if np.issubdtype(dtype, np.integer):
                arr = np.round(arr).astype(dtype)
            axes = axes.replace(ax, "")
    return arr, axes


def _labels_axes(arr: np.ndarray, axes: str) -> tuple[np.ndarray, str]:
    """Same as _gray for a mask, but a color becomes one integer per color
    (so that red and green stay two classes)."""
    if "T" in axes:
        arr = np.take(arr, 0, axis=axes.index("T"))
        axes = axes.replace("T", "")
    for ax in ("S", "C"):
        if ax in axes:
            i = axes.index(ax)
            arr = np.moveaxis(arr, i, -1).astype(np.int64)
            packed = np.zeros(arr.shape[:-1], dtype=np.int64)
            for c in range(min(arr.shape[-1], 3)):
                packed = packed * 256 + arr[..., c]
            arr = packed
            axes = axes.replace(ax, "")
    return arr, axes


def read_tiff(path: Path, as_labels=False) -> tuple[np.ndarray, list]:
    """A TIFF file as (Z, Y, X) with its spacing [z, y, x] (ImageJ tags, else 1)."""
    import tifffile
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        arr = series.asarray()
        axes = series.axes.upper()
        spacing = [1.0, 1.0, 1.0]
        try:
            ij = tif.imagej_metadata or {}
            page = tif.pages[0]
            xr = page.tags.get("XResolution")
            yr = page.tags.get("YResolution")
            if xr and xr.value[0]:
                spacing[2] = xr.value[1] / xr.value[0]
            if yr and yr.value[0]:
                spacing[1] = yr.value[1] / yr.value[0]
            if ij.get("spacing"):
                spacing[0] = float(ij["spacing"])
            if not ij:  # resolution tags of a non-ImageJ file are often dpi: not a spacing
                spacing = [1.0, 1.0, 1.0]
        except Exception:
            spacing = [1.0, 1.0, 1.0]
    arr, axes = (_labels_axes if as_labels else _gray)(arr, axes)
    # anything left besides Y and X is the depth (Z, Q, I...)
    while arr.ndim > 3:
        arr = arr.reshape((-1,) + arr.shape[-2:])
    if arr.ndim == 2:
        arr = arr[None]
    if arr.ndim != 3:
        raise ValueError(f"unexpected TIFF shape {arr.shape} ({axes})")
    return arr, spacing


def read_slice(path: Path, as_labels=False) -> np.ndarray:
    """One 2D slice as (Y, X)."""
    if path.suffix.lower() in TIFF_EXTS:
        arr, _ = read_tiff(path, as_labels)
        if arr.shape[0] != 1:
            raise ValueError(f"{path.name} holds {arr.shape[0]} slices, not one")
        return arr[0]
    img = Image.open(path)
    img.load()
    if as_labels:
        if img.mode == "P":
            return np.asarray(img).astype(np.int64)  # palette index = class
        if img.mode in ("RGB", "RGBA"):
            a = np.asarray(img.convert("RGB")).astype(np.int64)
            return (a[..., 0] * 256 + a[..., 1]) * 256 + a[..., 2]
        return np.asarray(img.convert("L") if img.mode not in ("I;16", "I;16B", "I;16L", "I", "F") else img)
    if img.mode in ("I;16", "I;16B", "I;16L"):
        return np.asarray(img).astype(np.uint16)
    if img.mode in ("I", "F"):
        return np.asarray(img)
    return np.asarray(img.convert("L"))


def _image_files(folder: Path):
    return sorted((p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in storage.IMAGE_EXTS),
                  key=lambda p: natural_key(str(p.relative_to(folder))))


def read_stack(folder: Path, as_labels=False) -> np.ndarray:
    files = _image_files(folder)
    if not files:
        raise ValueError("no image in this folder")
    slices = [read_slice(p, as_labels) for p in files]
    shapes = {s.shape for s in slices}
    if len(shapes) > 1:
        raise ValueError(f"slices of different sizes: {sorted(shapes)[:3]}")
    return np.stack(slices)


def _is_stack(folder: Path) -> bool:
    """A folder of 2D slices: no sub-folder, no multi-page TIFF."""
    import tifffile
    for p in folder.iterdir():
        if p.is_dir():
            return False
        if p.suffix.lower() in TIFF_EXTS:
            try:
                with tifffile.TiffFile(p) as t:
                    if len(t.pages) > 1:
                        return False
            except Exception:
                pass
    return True


def collect(root: Path, fallback_name: str, as_labels=False):
    """Volumes found in a folder: {name: (array, spacing)}, and errors.

    A multi-page TIFF is a volume; a folder of 2D slices is a volume; loose 2D
    images (PNG, JPEG, single-page TIFF) together are one volume of slices,
    named after their folder (fallback_name at the top). Other folders are
    looked into."""
    vols, errors = {}, []
    if root.exists():
        _collect_into(root, fallback_name, as_labels, vols, errors)
    for n, (a, _) in list(vols.items()):
        if a.size > MAX_VOXELS:
            errors.append(f"{n}: {a.shape} is too large (more than {MAX_VOXELS // 1_000_000} million voxels)")
            del vols[n]
    return vols, errors


def _collect_into(folder: Path, loose_name: str, as_labels, vols, errors):
    loose = []
    for p in sorted(folder.iterdir(), key=lambda p: natural_key(p.name)):
        try:
            if p.is_dir():
                if _is_stack(p):
                    vols[p.name] = (read_stack(p, as_labels), [1.0, 1.0, 1.0])
                else:
                    _collect_into(p, p.name, as_labels, vols, errors)
            elif p.suffix.lower() in TIFF_EXTS:
                arr, spacing = read_tiff(p, as_labels)
                if arr.shape[0] == 1:
                    loose.append((p, arr[0]))
                else:
                    vols[p.stem] = (arr, spacing)
            elif p.suffix.lower() in storage.IMAGE_EXTS:
                loose.append((p, read_slice(p, as_labels)))
        except Exception as e:
            errors.append(f"{p.name}: {e}")
    if len(loose) == 1:
        p, a = loose[0]
        vols[p.stem] = (a[None], [1.0, 1.0, 1.0])
    elif loose:
        if len({a.shape for _, a in loose}) > 1:
            errors.append(f"{loose_name}: slices of different sizes")
        else:
            vols[loose_name] = (np.stack([a for _, a in loose]), [1.0, 1.0, 1.0])


def to_class_values(masks: dict, existing_max: int = 0):
    """Raw mask values -> classes 0..N. Small integers (up to 20) are taken as
    class numbers; otherwise (0/255 masks, colors...) the distinct non-zero
    values become 1, 2, 3... in increasing order, the same for every volume."""
    values = set()
    for a in masks.values():
        values.update(np.unique(a).tolist())
    values.discard(0)
    if not values:
        return {n: np.zeros(a.shape, np.uint8) for n, a in masks.items()}, 0
    if max(values) <= 20 and all(float(v).is_integer() and v > 0 for v in values):
        return {n: a.astype(np.uint8) for n, a in masks.items()}, int(max(values))
    order = sorted(values)
    if len(order) > MAX_CLASSES:
        raise ValueError(f"{len(order)} different values in the masks: at most {MAX_CLASSES} classes")
    lut = {v: i + 1 for i, v in enumerate(order)}
    out = {}
    for n, a in masks.items():
        u, inv = np.unique(a, return_inverse=True)
        mapped = np.array([lut.get(v, 0) for v in u.tolist()], dtype=np.uint8)
        out[n] = mapped[inv].reshape(a.shape)
    return out, len(order)


# ---------------- import ----------------

def stage_file(root: Path, rel: str, data: bytes):
    """Write a dropped file under root at its relative path (cleaned: it
    cannot leave root)."""
    parts = [re.sub(r"[/\\:*?\"<>|]", "_", p).strip().rstrip(".") for p in rel.replace("\\", "/").split("/")]
    parts = [p for p in parts if p not in ("", ".", "..")]
    if not parts:
        raise ValueError("path")
    dest = root.joinpath(*parts)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)


def save_upload(name: str, kind: str, rel: str, data: bytes):
    """One dropped file, kept with its path until `finalize`."""
    if kind not in ("img", "mask"):
        raise ValueError("kind")
    stage_file(dataset_dir(name) / "_incoming" / kind, rel, data)


def finalize(name: str):
    """Build the volumes from the dropped files: images, then masks matched by
    name (vol1 <-> vol1, vol1_mask, vol1_label...). A volume without a mask
    gets empty labels."""
    d = dataset_dir(name)
    inc = d / "_incoming"
    try:
        imgs, errors = collect(inc / "img", name)
        masks, merr = collect(inc / "mask", name, as_labels=True)
        errors += [f"mask {e}" for e in merr]
        (d / "imgs").mkdir(parents=True, exist_ok=True)
        (d / "labels").mkdir(exist_ok=True)
        meta = read_meta(d)
        names = {}
        for n, (arr, spacing) in imgs.items():
            stem = clean_stem(n)
            if arr.dtype == np.bool_:
                arr = arr.astype(np.uint8)
            if arr.dtype.kind not in "uif" or arr.dtype.itemsize > 4:
                arr = arr.astype(np.float32)
            np.save(d / "imgs" / f"{stem}.npy", np.ascontiguousarray(arr))
            (d / "view" / f"{stem}.npy").unlink(missing_ok=True)
            meta["items"][stem] = {"spacing": [float(s) for s in spacing]}
            names[stem] = arr.shape
        matched, orphans = {}, 0
        for mn, (arr, _) in masks.items():
            target = storage.match_mask(clean_stem(mn), set(names))
            if not target:
                orphans += 1
            elif arr.shape != names[target]:
                errors.append(f"mask {mn}: shape {arr.shape} but the image is {names[target]}")
            else:
                matched[target] = arr
        matched, n_classes = to_class_values(matched)
        for stem, shape in names.items():
            lab = matched.get(stem)
            if lab is None:
                lab = np.zeros(shape, np.uint8)
            np.save(d / "labels" / f"{stem}.npy", lab)
        n_classes = max(n_classes, len(meta["classes"]), 1)
        meta["classes"] += [default_class(i) for i in range(len(meta["classes"]) + 1, n_classes + 1)]
        write_meta(d, meta)
    finally:
        shutil.rmtree(inc, ignore_errors=True)
    return {"volumes": len(names), "paired": len(matched), "orphan_masks": orphans,
            "classes": len(meta["classes"]), "errors": errors}


# ---------------- listing ----------------

def list_datasets():
    out = []
    for d in sorted(VOLUMES.iterdir(), key=lambda p: natural_key(p.name)):
        if not d.is_dir():
            continue
        items = list_items_in(d)
        out.append({"name": d.name, "count": len(items), "annotated": sum(i["annotated"] for i in items),
                    "classes": read_meta(d)["classes"]})
    return out


def list_items(name: str):
    d = dataset_dir(name)
    if not d.exists():
        raise FileNotFoundError(name)
    return list_items_in(d)


def list_items_in(d: Path):
    meta = read_meta(d)
    items = []
    for p in sorted((d / "imgs").glob("*.npy"), key=lambda p: natural_key(p.stem)):
        shape = np.load(p, mmap_mode="r").shape
        lp = d / "labels" / p.name
        annotated, slices = False, 0
        if lp.exists():
            lab = np.load(lp, mmap_mode="r")
            per_z = lab.reshape(lab.shape[0], -1).max(axis=1)
            slices = int((per_z > 0).sum())
            annotated = slices > 0
        items.append({"name": p.stem, "shape": list(shape), "annotated": annotated, "labeled_slices": slices,
                      "spacing": meta["items"].get(p.stem, {}).get("spacing", [1.0, 1.0, 1.0])})
    return items


# ---------------- volumes for the viewer ----------------

def display_volume(d: Path, item: str) -> np.ndarray:
    """uint8 copy of the image, contrast from the 0.5 and 99.5 percentiles."""
    check_name(item)
    src = d / "imgs" / f"{item}.npy"
    if not src.exists():
        raise FileNotFoundError(item)
    cache = d / "view" / f"{item}.npy"
    if cache.exists() and cache.stat().st_mtime >= src.stat().st_mtime:
        return np.load(cache)
    arr = np.load(src)
    if arr.dtype == np.uint8:
        out = arr
    else:
        sample = arr.ravel()[:: max(1, arr.size // 2_000_000)].astype(np.float32)
        lo, hi = np.percentile(sample, 0.5), np.percentile(sample, 99.5)
        out = np.clip((arr.astype(np.float32) - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
    cache.parent.mkdir(exist_ok=True)
    np.save(cache, out)
    return out


def labels(d: Path, item: str) -> np.ndarray:
    check_name(item)
    p = d / "labels" / f"{item}.npy"
    if p.exists():
        return np.load(p)
    shape = np.load(d / "imgs" / f"{item}.npy", mmap_mode="r").shape
    return np.zeros(shape, np.uint8)


def packed(arr: np.ndarray) -> bytes:
    """Raw bytes, gzip-compressed (sent with Content-Encoding: gzip)."""
    return gzip.compress(np.ascontiguousarray(arr, dtype=np.uint8).tobytes(), compresslevel=1)


def write_labels(name: str, item: str, body: bytes, encoding: str | None):
    """Labels sent by the annotation page: raw uint8 (Z, Y, X), maybe gzipped."""
    d = dataset_dir(name)
    check_name(item)
    src = d / "imgs" / f"{item}.npy"
    if not src.exists():
        raise FileNotFoundError(item)
    shape = np.load(src, mmap_mode="r").shape
    data = gzip.decompress(body) if encoding == "gzip" or body[:2] == b"\x1f\x8b" else body
    if len(data) != int(np.prod(shape)):
        raise ValueError(f"{len(data)} bytes for a volume of {shape}")
    lab = np.frombuffer(data, dtype=np.uint8).reshape(shape)
    meta = read_meta(d)
    top = int(lab.max())
    if top > len(meta["classes"]):
        meta["classes"] += [default_class(i) for i in range(len(meta["classes"]) + 1, top + 1)]
        write_meta(d, meta)
    tmp = d / "labels" / f"{item}.tmp.npy"
    np.save(tmp, lab)
    tmp.replace(d / "labels" / f"{item}.npy")


def meshes(d: Path, item: str, max_dim: int = 192) -> dict:
    """The surface of each class (marching cubes), for the 3D view.

    Vertices are (x, y, z) in the units of the spacing (µm...), float32, and
    faces triangles of vertex indices, uint32, both base64. A large volume is
    subsampled to about max_dim voxels per side first."""
    import base64
    import math
    from scipy.ndimage import gaussian_filter
    from skimage.measure import marching_cubes
    lab = labels(d, item)
    spacing = np.array(read_meta(d)["items"].get(item, {}).get("spacing", [1.0, 1.0, 1.0]), dtype=np.float32)
    step = max(1, math.ceil(max(lab.shape) / max_dim))
    out = []
    present = np.flatnonzero(np.bincount(lab[::step, ::step, ::step].ravel(), minlength=2)[1:]) + 1
    for k in present.tolist():
        m = (lab == k)
        if step > 1:  # a voxel of the coarse grid is on when any of its block is
            z, y, x = (math.ceil(s / step) * step for s in m.shape)
            m = np.pad(m, [(0, z - m.shape[0]), (0, y - m.shape[1]), (0, x - m.shape[2])])
            m = m.reshape(z // step, step, y // step, step, x // step, step).any(axis=(1, 3, 5))
        m = np.pad(m.astype(np.float32), 1)  # closed surfaces at the borders
        m = gaussian_filter(m, 0.7)
        if m.max() < 0.5:
            continue
        verts, faces, _, _ = marching_cubes(m, 0.5, spacing=tuple(spacing * step))
        verts = (verts - spacing * step)[:, ::-1]  # (z, y, x) -> (x, y, z), back from the padding
        out.append({
            "label": int(k), "triangles": int(len(faces)),
            "vertices": base64.b64encode(np.ascontiguousarray(verts, dtype=np.float32).tobytes()).decode(),
            "faces": base64.b64encode(np.ascontiguousarray(faces[:, ::-1], dtype=np.uint32).tobytes()).decode(),
        })
    return {"shape": list(lab.shape), "spacing": spacing.tolist(), "step": step, "meshes": out}


def thumbnail(d: Path, item: str, size: int = 128) -> bytes:
    vol = display_volume(d, item)
    img = Image.fromarray(vol[vol.shape[0] // 2])
    img.thumbnail((size, size))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


# ---------------- changes ----------------

def delete_item(name: str, item: str):
    d = dataset_dir(name)
    check_name(item)
    for sub in ("imgs", "labels", "view"):
        (d / sub / f"{item}.npy").unlink(missing_ok=True)
    meta = read_meta(d)
    meta["items"].pop(item, None)
    write_meta(d, meta)


def delete_dataset(name: str):
    shutil.rmtree(dataset_dir(name))


def rename_dataset(name: str, new: str):
    src, dst = dataset_dir(name), dataset_dir(new)
    if dst.exists():
        raise FileExistsError(new)
    src.rename(dst)


def add_volume(name: str, item: str, image: np.ndarray, lab: np.ndarray, spacing, classes=None):
    """Used by '-> Dataset' in Inference: a predicted volume to correct."""
    d = dataset_dir(name)
    (d / "imgs").mkdir(parents=True, exist_ok=True)
    (d / "labels").mkdir(exist_ok=True)
    meta = read_meta(d)
    stem, n = item, 1
    while (d / "imgs" / f"{stem}.npy").exists():
        n += 1
        stem = f"{item}_{n}"
    np.save(d / "imgs" / f"{stem}.npy", image)
    np.save(d / "labels" / f"{stem}.npy", lab.astype(np.uint8))
    meta["items"][stem] = {"spacing": [float(s) for s in spacing]}
    if classes and not meta["classes"]:
        meta["classes"] = [dict(c) for c in classes]
    top = int(lab.max())
    meta["classes"] += [default_class(i) for i in range(len(meta["classes"]) + 1, top + 1)]
    if not meta["classes"]:
        meta["classes"] = [default_class(1)]
    write_meta(d, meta)
    return stem
