# Models creator

A FastAPI application to annotate images, train U-Net segmentation models (black and white
masks) on 512×512 images, and test them, with PyTorch or TensorFlow. It does the same for 3D
volumes (multi-class labels) with [nnU-Net v2](https://github.com/MIC-DKFZ/nnUNet).

## Install

One file, the same for Windows and Linux: **`install.cmd`**.

- **Windows**: double-click `install.cmd`.
- **Linux**: in a terminal, `sh install.cmd`.

It installs [uv](https://docs.astral.sh/uv/) if it is missing. uv then installs Python 3.12
and the libraries into `.venv`. This takes a few GB, mostly PyTorch, TensorFlow and nnU-Net, and needs
an Internet connection. The installer then puts a "Models creator" icon on the Desktop and in
the applications menu (the Start menu on Windows), and offers to start the application.
Nothing is installed outside this folder, except uv (in `~/.local/bin`) and Python (in uv's
cache).

It is safe to run it again, for example after updating the folder or moving it: it brings
the libraries and the icons up to date.

## Start and stop

The icon runs `launch.py`. It starts the server with no window and opens the page in the
browser as soon as the server answers. If the application is already running, it only opens
the page. The server's messages go to `logs/server.log`.

The **Quit** button, at the top right, stops the server and closes the window when the
browser allows it.

To see the server's messages live, for example when it does not start:

```
uv run launch.py --console
```

then open http://127.0.0.1:8765.

## Tabs

**Data**: choose **2D images · U-Net** or **3D volumes · nnU-Net** above the drop zone.

*2D*: drop a folder containing `imgs/` and `masks/`. Masks are matched to images by
name (`img1.png` ↔ `img1.png`, `img1_mask.png`, `img1_label.png`…). A folder that holds only
images is accepted too: each image gets a black mask. Images are resized to 512×512, and
every non-black pixel of a mask counts as object. `data_examples/` holds a folder of each
kind to try it.

*3D*: a volume is a multi-page TIFF file, or a folder of 2D slices (PNG, TIFF…, stacked in
name order). Put the volumes in `imgs/` and their labels in `masks/` (same names, or
`vol1_mask.tif`…), or drop volumes only: their labels start empty. A dropped folder that
holds only loose slices is one volume. Volumes keep their size and their values (8 or 16
bits); colors and channels are averaged to gray. The spacing (z, y, x) is read from ImageJ
TIFF tags, 1 otherwise. Labels: 0 is the background and every other value a class; small
integers (up to 20) are kept as class numbers, other values (0/255 masks, colors) become
classes 1, 2, 3… in increasing order. In 2D mode, a folder whose `imgs/` holds sub-folders
of slices is offered as 3D.

`data_examples/noyaux_3d/` is a small synthetic 3D example to try it: five 16-bit volumes
of 32×64×64 voxels of 1 µm with nuclei (class 1) and their nucleoli (class 2). Four are
multi-page TIFF files, the fifth a folder of PNG slices, with their labels in `masks/`
(`noyaux_01_mask.tif`…, values 0 / 127 / 254). On a CPU, 20 epochs of 10 iterations in
`3d_fullres` (about 7 minutes) reach a validation Dice of 0.97 for the nuclei and 0.81 for
the nucleoli; the nucleoli, small, only appear after about ten epochs.

**Annotation**: the image with its mask overlaid.

| Tool | Key | Use |
|---|---|---|
| Brush | B | Shift = erase |
| Eraser | E | |
| Polygon | P | click = add a point, double-click or Enter = close, Backspace = remove the last point, Esc = cancel, Shift = erase |
| Rectangle | R | Shift = erase |
| Fill | F | fills the clicked area, Shift = empties it |
| Pan | hold Space, or right / middle button drag | |

Mouse wheel: zoom. `[` `]`: brush size. H: hide the mask. O: outline only.
Ctrl+Z / Ctrl+Y: undo / redo. Ctrl+S: save (also done automatically when changing image).
← → or A D: previous / next image. 0: fit to the window.

**Annotation 3D**: a volume slice by slice, in the XY, XZ or YZ plane (V), with its labels,
one color per class. Classes are named and colored in the side panel (double-click a name to
rename it, + Class to add one); 1 to 9 pick the class to paint. "keep other classes" makes the
brush paint only the background. Each thumbnail shows how many Z slices hold labels; the
strip under the slice slider shows them along the current axis.

| Tool | Key | Use |
|---|---|---|
| Brush | B | Shift = erase |
| Eraser | E | |
| Fill | F | fills the clicked area of the slice, Shift = empties it |
| Copy slice | C | copies the labels of the previous slice onto this one, Shift = of the next one |
| Pan | hold Space, or right / middle button drag | |

Mouse wheel, ↑ ↓ or W S: change slice (Shift = 10 at a time). Ctrl + wheel: zoom. ← → or A D:
previous / next volume. `[` `]`, H, O, 0, Ctrl+Z / Ctrl+Y and Ctrl+S as in Annotation.

**3D** (next to XY, XZ, YZ, or T): the surfaces of the labels, one color per class, drawn
with [Three.js](https://threejs.org/) in the box of the volume, the current slice as a
translucent plane (↑ ↓ and the slider move it). Drag: rotate, right drag: pan, wheel: zoom,
double-click: fit. The Opacity slider makes the surfaces translucent, to see the classes
inside others. The surfaces are those of the saved labels (unsaved changes are saved
first), made by the server (marching cubes on each class, slightly smoothed; a volume larger
than 192 voxels on a side is first reduced). It shows; it does not edit.

**Training**: choose the dataset, epochs, batch size, learning rate, validation split and
U-Net size. Loss, Accuracy and Recall, as well as IoU, are followed live for training and
validation. Precision is computed too and kept in the model file. The best model (lowest
validation loss) is saved after each epoch that improves it, so stopping a training keeps
the best model so far.

With a 3D dataset the training is nnU-Net's: it plans the network, the patch size, the
spacing and the augmentation from the volumes, then trains its standard `nnUNetTrainer`.
Configuration: `3d_fullres` (default), `2d` (slice by slice, faster) or `3d_lowres` (planned
only for large volumes). nnU-Net's defaults are 1000 epochs of 250 iterations plus 50
validation iterations; the form starts at 100 epochs and lets you lower the iterations.
The validation split works as in 2D (at least one volume; with a single volume it is also
the validation volume). "Slices without labels are not annotated" uses nnU-Net's ignore
label: the Z slices holding no label are not learnt as background, for volumes annotated on
a few slices only. The curves are the losses, the pseudo Dice of the validation batches and
its moving average (EMA), and the pseudo Dice per class; the best model (highest EMA) is
saved after each improvement. At the end, nnU-Net predicts the whole validation volumes
with the best model and gives their Dice. Its log can be unfolded under the curves.

**Inference**: choose a model, or drop a `.pt`, `.keras`, `.h5` file or the `.zip` of an
nnU-Net model, then drop images.
They can be of any size: the prediction is made at 512×512, then brought back to the image's
size. Results show as a mosaic, or stacked with a before/after slider. Image size (slider or
Ctrl + wheel), opacity, color and threshold can be adjusted. Results can be downloaded as a
zip, or copied into a dataset to be corrected in Annotation.

With an nnU-Net model the tab takes volumes (TIFF files or folders of slices). They are
predicted one after the other at their own size and shown slice by slice in a viewer like
Annotation 3D, with the share of each class. "Test-time mirroring" also predicts the
mirrored volumes and averages them: a little more accurate, up to 8 times slower. Results
download as a zip of label TIFFs (one value per class, listed in `classes.txt`), or go into
a 3D dataset to be corrected in Annotation 3D. The 3D button shows the predicted surfaces.

**Info**: the framework in use and the other one, their versions and builds (CPU or CUDA),
the compute device and, when no GPU is used, why. For each GPU (through `nvidia-smi`): name,
VRAM used and total, load, temperature, power, clocks and driver, refreshed every 2 s. Also
the system (OS, processor, RAM, Python) and the versions of the libraries (TensorFlow, JAX,
ONNX Runtime… when installed).

**Settings**: dark, light or system theme; the framework for new trainings, PyTorch (`.pt`
models) or TensorFlow / Keras (`.keras` models), while Inference opens both whatever this
setting; the device (auto or CPU only) and the number of CPU threads. Settings are kept in
`data/settings.json`. TensorFlow reads the device and threads only when it is first loaded:
after changing them, restart the application.

## Where the models are

In `data/models/`, named after the model name given in Training (by default
`unet_<dataset>_<date>_<time>`):

- PyTorch: one `.pt` file holding the weights, the architecture and the training history.
- TensorFlow: a `.keras` file, with a `.json` file of the same name for the history and
  scores.
- nnU-Net: a folder `<name>.nnunet/` in nnU-Net's own trained-model layout (`plans.json`,
  `dataset.json`, `fold_0/checkpoint_best.pth`), plus `meta.json` for the history and
  scores. Its checkpoint holds only what prediction needs. The folder also works with
  nnU-Net's command `nnUNetv2_predict_from_modelfolder -m <folder> -f 0 -chk checkpoint_best.pth`.

The **Download** button of each model card in Training saves a copy elsewhere (a zip for
nnU-Net). 3D datasets are in `data/volumes/` (`.npy` volumes and labels). `data/nnunet/` is
the work folder of the last nnU-Net training (its dataset in nnU-Net's format, the
preprocessed data, `train.log`): it is emptied when the next one starts. `data/` (datasets,
models, results, settings) is not versioned.

## Files

```
install.cmd        the installer (Windows and Linux)
launch.py          what the icon runs
app/main.py        FastAPI routes
app/installer.py   creates the icons (second half of install.cmd)
app/storage.py     datasets and model files on disk
app/unet.py        the U-Net model (PyTorch)
app/training.py    background training and metrics
app/tf_backend.py  the same U-Net, training and prediction in TensorFlow / Keras
app/inference.py   loading models and predicting
app/volumes.py     3D datasets: reading TIFF and slice folders, labels, classes
app/nnunet_backend.py  nnU-Net: training (dataset, worker process), models, 3D prediction
app/nnunet_worker.py   the nnU-Net training process (plan, preprocess, train, export)
app/settings.py    settings (data/settings.json)
app/sysinfo.py     what the Info tab shows
static/            the interface (HTML, CSS, JavaScript; no dependency but Three.js)
static/vendor/three/   Three.js 0.170.0 and its OrbitControls (MIT), for the 3D view
data_examples/     example folders to drop in the Data tab
data/              datasets, models, results (not versioned)
```

Dropped models are read without running code: `torch.load(weights_only=True)` for PyTorch
(models from this application, the `state_dict` of a U-Net with the same architecture, or
TorchScript) and for nnU-Net checkpoints, and `keras.models.load_model` in safe mode for
`.keras` / `.h5`. An nnU-Net zip may come from this application or from
`nnUNetv2_export_model_to_zip`; its model must take one channel (grayscale).

## Computation

On Windows, the PyTorch that gets installed is the CPU version. On a CPU, keep a small
network (8 or 16 base filters). Measured on a Xeon W-2223, per image and per epoch: 0.3 s
with 8 filters, 0.6 s with 16, 1.5 s with 32, so about 1 minute per epoch for 100 images
with 16 filters. To use a recent NVIDIA card, install the CUDA version of PyTorch: the
application then uses it automatically. On Linux, the default PyTorch is the CUDA version.

nnU-Net trains in its own process, with a few data augmentation processes beside it;
Stop and Quit end them all. On a CPU, a 3D network is slow: with the small synthetic volumes
of the tests (32×64×64), an epoch of 4 iterations took 10 s; real volumes and 250 iterations
take far longer, so check a dataset with few epochs or iterations, or train on a machine
with an NVIDIA GPU and bring the model back as a zip. `torch.compile` is off (no Triton on
Windows).

TensorFlow does not use GPUs on native Windows since version 2.11 (only under WSL2), so it
always computes on the CPU there. Its first loading takes up to half a minute; it is loaded
only when it is selected in Settings or a `.keras` model is used.
