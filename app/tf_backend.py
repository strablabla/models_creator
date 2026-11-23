"""TensorFlow / Keras side: the same U-Net, training loop and prediction as
the PyTorch side. Imported only when TensorFlow is used, since importing it
takes several seconds and a few hundred MB."""
import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # no oneDNN / CPU feature notices

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from . import settings, storage


def _configure():
    """Threads and visible devices can only be set before TensorFlow starts
    its runtime, so they follow the settings as they were at first use."""
    s = settings.get()
    try:
        if s["threads"]:
            tf.config.threading.set_intra_op_parallelism_threads(s["threads"])
            tf.config.threading.set_inter_op_parallelism_threads(min(s["threads"], 4))
        if s["device"] == "cpu":
            tf.config.set_visible_devices([], "GPU")
    except RuntimeError:
        pass


_configure()


def gpus():
    return [d.name for d in tf.config.list_physical_devices("GPU")]


def info():
    return {
        "version": tf.__version__,
        "keras": keras.__version__,
        "built_with_cuda": tf.test.is_built_with_cuda(),
        "gpus": gpus(),
        "device_used": "GPU" if gpus() else "CPU",
    }


def _double_conv(x, ch):
    for _ in range(2):
        x = layers.Conv2D(ch, 3, padding="same", use_bias=False)(x)
        # momentum 0.9 = PyTorch's default; Keras' 0.99 leaves the running statistics
        # far from the data after a short training, and validation then fails
        x = layers.BatchNormalization(momentum=0.9)(x)
        x = layers.ReLU()(x)
    return x


def build_unet(base=16, depth=4, in_channels=3, out_channels=1):
    inp = keras.Input((None, None, in_channels))
    x, skips = inp, []
    for i in range(depth + 1):
        x = _double_conv(x, base * 2 ** i)
        if i < depth:
            skips.append(x)
            x = layers.MaxPooling2D(2)(x)
    for i in reversed(range(depth)):
        x = layers.Conv2DTranspose(base * 2 ** i, 2, strides=2)(x)
        x = layers.Concatenate()([skips[i], x])
        x = _double_conv(x, base * 2 ** i)
    out = layers.Conv2D(out_channels, 1)(x)  # logits, like the PyTorch U-Net
    return keras.Model(inp, out, name="unet")


def _loss(logits, y):
    bce = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits))
    p = tf.sigmoid(logits)
    inter = tf.reduce_sum(p * y, axis=[1, 2, 3])
    union = tf.reduce_sum(p, axis=[1, 2, 3]) + tf.reduce_sum(y, axis=[1, 2, 3])
    dice = tf.reduce_mean(1 - (2 * inter + 1) / (union + 1))
    return bce + dice


def _counts(logits, y):
    pred = logits > 0  # sigmoid(logits) > 0.5
    t = y > 0.5
    f = lambda m: tf.reduce_sum(tf.cast(m, tf.int64))
    return f(pred & t), f(pred & ~t), f(~pred & ~t), f(~pred & t)


def _arrays(pairs):
    x = np.stack([p[0] for p in pairs]).astype(np.float32) / 255.0
    y = np.stack([p[1] for p in pairs]).astype(np.float32)[..., None]
    return x, y


def train(trainer, cfg, train_names, val_names, batch, rng):
    from .training import Counts

    tf.keras.utils.set_random_seed(cfg.get("seed", 0))
    model = build_unet(cfg["base"], cfg["depth"])
    bs = cfg["batch_size"]
    batches = (len(train_names) + bs - 1) // bs
    # cosine decay over the whole training, as the PyTorch side does per epoch
    sched = keras.optimizers.schedules.CosineDecay(cfg["lr"], decay_steps=max(cfg["epochs"] * batches, 1))
    opt = keras.optimizers.Adam(sched)

    @tf.function(reduce_retracing=True)
    def train_step(x, y):
        with tf.GradientTape() as tape:
            logits = model(x, training=True)
            loss = _loss(logits, y)
        opt.apply_gradients(zip(tape.gradient(loss, model.trainable_variables), model.trainable_variables))
        return loss, _counts(logits, y)

    @tf.function(reduce_retracing=True)
    def eval_step(x, y):
        logits = model(x, training=False)
        return _loss(logits, y), _counts(logits, y)

    dev = info()["device_used"]
    trainer.set(state="running", device=f"TensorFlow {tf.__version__} · {dev}", n_train=len(train_names),
                n_val=len(val_names), params=int(model.count_params()), batches=batches,
                message=f"{len(train_names)} training images, {len(val_names)} validation images")
    out_path = storage.MODELS / f"{cfg['model_name']}.keras"

    def save(best, history):
        tmp = storage.MODELS / f"{cfg['model_name']}.tmp.keras"
        model.save(tmp)
        tmp.replace(out_path)
        meta = trainer.meta(cfg, best, history)
        meta["config"] = {"in_channels": 3, "out_channels": 1, "base": cfg["base"], "depth": cfg["depth"]}
        storage.model_meta_path(cfg["model_name"]).write_text(json.dumps(meta), encoding="utf-8")
        storage.remove_other_model_files(cfg["model_name"], keep=out_path)

    for epoch in range(1, cfg["epochs"] + 1):
        rng.shuffle(train_names)
        tr = Counts()
        for b in range(batches):
            if trainer.stopping:
                break
            names = train_names[b * bs:(b + 1) * bs]
            x, y = _arrays(batch(names, cfg.get("augment", True)))
            loss, c = train_step(x, y)
            loss = float(loss)
            tr.add(*(int(v) for v in c), loss, len(names))
            trainer.set(epoch=epoch, batch=b + 1, batch_loss=loss)
        if trainer.stopping:
            break
        va = Counts()
        for b in range(0, len(val_names), bs):
            names = val_names[b:b + bs]
            x, y = _arrays(batch(names, False))
            loss, c = eval_step(x, y)
            va.add(*(int(v) for v in c), float(loss), len(names))
        lr = float(sched(opt.iterations))
        trainer.end_epoch(epoch, tr, va, lr, save)
    return out_path


class KerasRunner:
    """Prediction with a Keras model: 512x512 RGB in [0, 1] -> (H, W) bool mask."""

    def __init__(self, path):
        # safe_mode (the default) refuses Lambda layers: a dropped file cannot run code
        self.model = keras.models.load_model(path, compile=False)

    def __call__(self, x_hwc, threshold):
        out = np.asarray(self.model(x_hwc[None], training=False))[0]
        if out.shape[-1] > 1:
            return out.argmax(-1) > 0
        out = out[..., 0]
        if out.min() < 0 or out.max() > 1:  # logits (ours), else already probabilities
            out = 1 / (1 + np.exp(-out))
        return out > threshold


def load(path):
    r = KerasRunner(path)
    m = r.model
    first = next((l for l in m.layers if isinstance(l, layers.Conv2D)), None)
    cfg = {"base": first.filters if first else None, "in_channels": m.input_shape[-1],
           "out_channels": m.output_shape[-1]}
    return r, cfg
