"""Training in a background thread, with metrics the page polls.

The data handling, the metrics and the status are shared; the model and the
training steps come from PyTorch (here) or TensorFlow (tf_backend.py),
whichever the settings name.
"""
import random
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from . import settings, storage
from .unet import UNet, count_parameters

TORCH_DEFAULT_THREADS = torch.get_num_threads()


def apply_torch_threads():
    n = settings.get()["threads"]
    torch.set_num_threads(n or TORCH_DEFAULT_THREADS)


def pick_device():
    """PyTorch device: the GPU when one is usable and the settings allow it."""
    if settings.get()["device"] != "cpu" and torch.cuda.is_available():
        try:
            torch.zeros(1).cuda()
            return torch.device("cuda")
        except Exception:
            pass
    return torch.device("cpu")


def load_pair(ds_dir: Path, item: str):
    img = np.asarray(Image.open(ds_dir / "imgs" / f"{item}.png").convert("RGB"), dtype=np.uint8)
    mpath = ds_dir / "masks" / f"{item}.png"
    if mpath.exists():
        mask = np.asarray(Image.open(mpath).convert("L")) > 127
    else:
        mask = np.zeros(img.shape[:2], dtype=bool)
    return img, mask


def augment(img, mask):
    k = random.randint(0, 3)
    img, mask = np.rot90(img, k), np.rot90(mask, k)
    if random.random() < 0.5:
        img, mask = img[:, ::-1], mask[:, ::-1]
    if random.random() < 0.5:
        # brightness / contrast jitter
        a = random.uniform(0.8, 1.2)
        b = random.uniform(-20, 20)
        img = np.clip(img.astype(np.float32) * a + b, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(img), np.ascontiguousarray(mask)


def to_tensors(pairs):
    x = torch.from_numpy(np.stack([p[0] for p in pairs])).permute(0, 3, 1, 2).float() / 255.0
    y = torch.from_numpy(np.stack([p[1] for p in pairs])).unsqueeze(1).float()
    return x, y


def dice_loss(logits, target, eps=1.0):
    p = torch.sigmoid(logits)
    inter = (p * target).sum(dim=(1, 2, 3))
    union = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return (1 - (2 * inter + eps) / (union + eps)).mean()


class Counts:
    """Pixel confusion counts, summed over an epoch."""

    def __init__(self):
        self.tp = self.fp = self.tn = self.fn = 0
        self.loss = 0.0
        self.n = 0

    def add(self, tp, fp, tn, fn, loss, n):
        self.tp += int(tp)
        self.fp += int(fp)
        self.tn += int(tn)
        self.fn += int(fn)
        self.loss += float(loss) * n
        self.n += n

    def add_torch(self, logits, target, loss, threshold=0.5):
        pred = torch.sigmoid(logits) > threshold
        t = target > 0.5
        self.add((pred & t).sum(), (pred & ~t).sum(), (~pred & ~t).sum(), (~pred & t).sum(),
                 loss, target.shape[0])

    def result(self):
        total = self.tp + self.fp + self.tn + self.fn
        return {
            "loss": self.loss / max(self.n, 1),
            "accuracy": (self.tp + self.tn) / max(total, 1),
            # no positive prediction at all: precision undefined, shown as 0
            "precision": self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0,
            "recall": self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0,
            "iou": self.tp / (self.tp + self.fp + self.fn) if (self.tp + self.fp + self.fn) else 0.0,
        }


class Trainer:
    def __init__(self):
        self.lock = threading.Lock()
        self.thread = None
        self.stop_flag = threading.Event()
        self.status = {"state": "idle", "message": "", "history": [], "config": None}

    def snapshot(self):
        with self.lock:
            s = dict(self.status)
            s["history"] = list(self.status.get("history", []))
            return s

    def set(self, **kw):
        with self.lock:
            self.status.update(kw)

    def start(self, cfg: dict):
        if self.thread and self.thread.is_alive():
            raise RuntimeError("A training is already running")
        storage.check_name(cfg["dataset"])
        storage.check_name(cfg["model_name"])
        cfg = dict(cfg, framework=cfg.get("framework") or settings.get()["framework"])
        if cfg["framework"] == "tensorflow" and not tensorflow_installed():
            raise ValueError("TensorFlow is not installed")
        self.stop_flag.clear()
        self.status = {
            "state": "starting",
            "message": "Loading TensorFlow…" if cfg["framework"] == "tensorflow" else "Preparing…",
            "history": [], "config": cfg,
            "epoch": 0, "epochs": cfg["epochs"], "batch": 0, "batches": 0,
            "batch_loss": None, "started": time.time(), "best": None,
        }
        self.thread = threading.Thread(target=self._run, args=(cfg,), daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_flag.set()

    @property
    def stopping(self):
        return self.stop_flag.is_set()

    def _run(self, cfg):
        try:
            self._train(cfg)
        except Exception as e:  # shown in the page
            self.set(state="error", message=f"Error: {e}")

    def _train(self, cfg):
        ds_dir = storage.dataset_dir(cfg["dataset"])
        items = storage.list_items(cfg["dataset"])
        if cfg.get("only_annotated", True):
            items = [i for i in items if i["annotated"]]
        names = [i["name"] for i in items]
        if not names:
            raise RuntimeError("No annotated image in this dataset")
        rng = random.Random(cfg.get("seed", 0))
        rng.shuffle(names)
        n_val = int(round(len(names) * cfg["val_split"]))
        if len(names) >= 2:
            n_val = min(max(n_val, 1), len(names) - 1)
            val, train = names[:n_val], names[n_val:]
        else:
            val, train = names, names
        cache = {}

        def get(name):
            if name not in cache:
                cache[name] = load_pair(ds_dir, name)
            return cache[name]

        def batch(names_, aug):
            pairs = [get(n) for n in names_]
            if aug:
                pairs = [augment(*p) for p in pairs]
            return pairs

        run = _train_torch
        if cfg["framework"] == "tensorflow":
            from .tf_backend import train as run
        out_path = run(self, cfg, train, val, batch, rng)
        best = self.status.get("best")
        if best is None:
            self.set(state="stopped", message="Stopped before the end of the first epoch: no model saved")
            return
        self.set(state="stopped" if self.stopping else "done",
                 message=(f"{'Stopped' if self.stopping else 'Done'}: best model (epoch {best['epoch']}) "
                          f"saved as {out_path.name}"))

    def end_epoch(self, epoch, tr, va, lr, save):
        """Record an epoch; save(best_row, history) is called when it is the best so far."""
        row = {"epoch": epoch, "train": tr.result(), "val": va.result(), "lr": lr, "time": time.time()}
        with self.lock:
            self.status["history"].append(row)
            history = list(self.status["history"])
            best = self.status.get("best")
        if best is None or row["val"]["loss"] < best["val"]["loss"]:
            save(row, history)
            self.set(best=row)

    def meta(self, cfg, best, history):
        return {
            "framework": cfg["framework"], "dataset": cfg["dataset"], "epoch": best["epoch"],
            "train_config": cfg, "best": best, "history": history, "size": storage.SIZE,
            "date": datetime.now().isoformat(timespec="seconds"),
        }


def _train_torch(trainer, cfg, train, val, batch, rng):
    apply_torch_threads()
    device = pick_device()
    torch.manual_seed(cfg.get("seed", 0))
    model = UNet(base=cfg["base"], depth=cfg["depth"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(cfg["epochs"], 1))
    bs = cfg["batch_size"]
    batches = (len(train) + bs - 1) // bs
    trainer.set(state="running", device=f"PyTorch {torch.__version__} · {device}", n_train=len(train),
                n_val=len(val), params=count_parameters(model), batches=batches,
                message=f"{len(train)} training images, {len(val)} validation images")
    out_path = storage.MODELS / f"{cfg['model_name']}.pt"

    def save(best, history):
        ckpt = {
            "format": "models_creator.unet.v1",
            "config": model.config,
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "meta": trainer.meta(cfg, best, history),
        }
        tmp = out_path.with_suffix(".tmp")
        torch.save(ckpt, tmp)
        tmp.replace(out_path)
        storage.remove_other_model_files(cfg["model_name"], keep=out_path)

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        rng.shuffle(train)
        tr = Counts()
        for b in range(batches):
            if trainer.stopping:
                break
            x, y = to_tensors(batch(train[b * bs:(b + 1) * bs], cfg.get("augment", True)))
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.binary_cross_entropy_with_logits(logits, y) + dice_loss(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr.add_torch(logits.detach(), y, loss.item())
            trainer.set(epoch=epoch, batch=b + 1, batch_loss=loss.item())
        if trainer.stopping:
            break
        sched.step()
        model.eval()
        va = Counts()
        with torch.no_grad():
            for b in range(0, len(val), bs):
                x, y = to_tensors(batch(val[b:b + bs], False))
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = F.binary_cross_entropy_with_logits(logits, y) + dice_loss(logits, y)
                va.add_torch(logits, y, loss.item())
        trainer.end_epoch(epoch, tr, va, opt.param_groups[0]["lr"], save)
    return out_path


def tensorflow_installed():
    from importlib.util import find_spec
    return find_spec("tensorflow") is not None


trainer = Trainer()
