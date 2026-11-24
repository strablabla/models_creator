"""nnU-Net training, run in its own process by nnunet_backend.py:

    python -m app.nnunet_worker <job.json>

Plans and preprocesses the dataset written in nnUNet_raw, then trains the
standard nnUNetTrainer with the number of epochs and iterations of the job,
on the train / validation split of the job (fold 0). It prints its progress as
lines "@@{json}" that the server reads; everything else is nnU-Net's own log.
After each epoch that improves the EMA pseudo Dice, the model (plans,
dataset.json and a checkpoint holding only what inference needs) is exported
to the job's export folder.

Its own process because nnU-Net reads its folders from environment variables
when first imported, starts data augmentation processes, and can only be
stopped cleanly by ending the whole process tree.
"""
import json
import os
import shutil
import sys
import time


def emit(**kw):
    print("@@" + json.dumps(kw), flush=True)


def export_model(trainer, dest: str):
    """The trained-model folder nnU-Net expects (nnUNetv2_predict_from_modelfolder
    reads it too), with a small checkpoint that torch.load(weights_only=True) opens."""
    import torch
    src = trainer.output_folder_base
    tmp = dest + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(os.path.join(tmp, "fold_0"))
    for f in ("plans.json", "dataset.json", "dataset_fingerprint.json"):
        if os.path.exists(os.path.join(src, f)):
            shutil.copy(os.path.join(src, f), tmp)
    net = trainer.network
    net = getattr(net, "_orig_mod", net)
    axes = trainer.inference_allowed_mirroring_axes
    torch.save({
        "network_weights": {k: v.detach().cpu() for k, v in net.state_dict().items()},
        "trainer_name": trainer.__class__.__name__,
        "init_args": {"configuration": trainer.configuration_name, "fold": 0},
        "inference_allowed_mirroring_axes": tuple(int(a) for a in axes) if axes is not None else None,
        "current_epoch": int(trainer.current_epoch),
        "_best_ema": float(trainer._best_ema) if trainer._best_ema is not None else None,
    }, os.path.join(tmp, "fold_0", "checkpoint_best.pth"))
    shutil.rmtree(dest, ignore_errors=True)
    os.replace(tmp, dest)


def hook(trainer, job):
    """Report batches and epochs, and export the model when it improves."""
    iters, val_iters = trainer.num_iterations_per_epoch, trainer.num_val_iterations_per_epoch
    state = {"batch": 0, "t": 0.0}

    def progress(loss, force=False):
        now = time.time()
        if force or now - state["t"] > 0.5:
            state["t"] = now
            emit(progress={"epoch": trainer.current_epoch + 1, "batch": state["batch"],
                           "batches": iters + val_iters, "loss": loss})

    o_epoch_start, o_train, o_val, o_end = (trainer.on_train_epoch_start, trainer.train_step,
                                            trainer.validation_step, trainer.on_epoch_end)

    def on_train_epoch_start():
        state["batch"] = 0
        o_epoch_start()
        progress(None, force=True)

    def train_step(batch):
        out = o_train(batch)
        state["batch"] += 1
        progress(float(out["loss"]))
        return out

    def validation_step(batch):
        out = o_val(batch)
        state["batch"] += 1
        progress(None)
        return out

    def on_epoch_end():
        before = trainer._best_ema
        o_end()  # logs, saves checkpoint_best.pth when better, counts the epoch
        lg = trainer.logger
        better = bool(trainer._best_ema is not None and trainer._best_ema != before)  # numpy bool otherwise
        row = {
            "epoch": trainer.current_epoch,
            "train": {"loss": float(lg.get_value("train_losses", -1))},
            "val": {"loss": float(lg.get_value("val_losses", -1)),
                    "dice": float(lg.get_value("mean_fg_dice", -1)),
                    "ema_dice": float(lg.get_value("ema_fg_dice", -1)),
                    "per_class": [float(v) for v in lg.get_value("dice_per_class_or_region", -1)]},
            "lr": float(lg.get_value("lrs", -1)),
            "seconds": float(lg.get_value("epoch_end_timestamps", -1) - lg.get_value("epoch_start_timestamps", -1)),
            "time": time.time(),
        }
        if better:
            export_model(trainer, job["export_dir"])
        emit(epoch=row, best=better)

    trainer.on_train_epoch_start = on_train_epoch_start
    trainer.train_step = train_step
    trainer.validation_step = validation_step
    trainer.on_epoch_end = on_epoch_end


def watch_parent(pid: int):
    """End this training when the server is gone (closed without Quit)."""
    if os.name == "nt":
        import ctypes
        h = ctypes.windll.kernel32.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE
        if not h:
            return
        ctypes.windll.kernel32.WaitForSingleObject(h, 0xFFFFFFFF)
    else:
        while os.getppid() == pid:
            time.sleep(5)
    os._exit(3)


def main(job_path):
    with open(job_path, encoding="utf-8") as f:
        job = json.load(f)
    if job.get("parent_pid"):
        import threading
        threading.Thread(target=watch_parent, args=(job["parent_pid"],), daemon=True).start()
    import torch
    if job.get("threads"):
        torch.set_num_threads(job["threads"])
    from batchgenerators.utilities.file_and_folder_operations import load_json, save_json
    from nnunetv2.experiment_planning.plan_and_preprocess_api import (extract_fingerprints, plan_experiments,
                                                                      preprocess)
    from nnunetv2.paths import nnUNet_preprocessed
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

    did, cfg, nproc = job["dataset_id"], job["configuration"], job["n_proc"]
    pre = os.path.join(str(nnUNet_preprocessed), job["dataset_name"])

    emit(phase="fingerprint", message="nnU-Net: checking the dataset and analysing the volumes…")
    extract_fingerprints([did], num_processes=nproc, check_dataset_integrity=True, clean=True,
                         verbose=False, show_progress_bar=False)
    emit(phase="planning", message="nnU-Net: planning the network (patch size, spacing, batch size)…")
    plan_experiments([did])
    plans = load_json(os.path.join(pre, "nnUNetPlans.json"))
    available = list(plans["configurations"])
    if cfg not in available:
        raise RuntimeError(f"nnU-Net did not plan '{cfg}' for this dataset (planned: {', '.join(available)}). "
                           "3d_lowres is planned only for large volumes.")
    c = plans["configurations"][cfg]
    emit(phase="preprocessing", plan={"patch_size": c.get("patch_size"), "batch_size": c.get("batch_size"),
                                      "spacing": c.get("spacing"), "configurations": available},
         message=f"nnU-Net: preprocessing for {cfg} (patch {c.get('patch_size')}, batch {c.get('batch_size')})…")
    preprocess([did], configurations=[cfg], num_processes=[nproc], verbose=False, show_progress_bar=False)
    save_json(job["splits"], os.path.join(pre, "splits_final.json"))

    plans["continue_training"] = False
    trainer = nnUNetTrainer(plans=plans, configuration=cfg, fold=0,
                            dataset_json=load_json(os.path.join(pre, "dataset.json")),
                            device=torch.device(job["device"]))
    trainer.num_epochs = job["epochs"]
    trainer.num_iterations_per_epoch = job["iterations"]
    trainer.num_val_iterations_per_epoch = job["val_iterations"]
    trainer.save_every = 10 ** 9  # the best model is exported after each improvement
    hook(trainer, job)
    emit(phase="training", message="nnU-Net: starting the data augmentation processes…")
    trainer.run_training()

    if job.get("final_validation", True):
        emit(phase="validation", message="nnU-Net: validating the best model on the whole validation volumes…")
        trainer.load_checkpoint(os.path.join(trainer.output_folder, "checkpoint_best.pth"))
        trainer.perform_actual_validation(False)
        summary = load_json(os.path.join(trainer.output_folder, "validation", "summary.json"))
        per_class = {k: v.get("Dice") for k, v in summary.get("mean", {}).items()}
        emit(validation={"dice": summary.get("foreground_mean", {}).get("Dice"), "per_class": per_class})
    for f in os.listdir(trainer.output_folder):
        if f.startswith("training_log") or f == "progress.png":
            shutil.copy(os.path.join(trainer.output_folder, f), job["export_dir"])
    emit(phase="done")


if __name__ == "__main__":
    try:
        main(sys.argv[1])
    except Exception as e:
        import traceback
        traceback.print_exc()
        emit(error=f"{type(e).__name__}: {e}")
        sys.exit(1)
