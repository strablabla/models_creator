# Models creator

A FastAPI application to annotate images, train U-Net segmentation models (black and white
masks) on 512×512 images, and test them, with PyTorch or TensorFlow.

## Install

One file, the same for Windows and Linux: **`install.cmd`**.

- **Windows**: double-click `install.cmd`.
- **Linux**: in a terminal, `sh install.cmd`.

It installs [uv](https://docs.astral.sh/uv/) if it is missing. uv then installs Python 3.12
and the libraries into `.venv`. This takes a few GB, mostly PyTorch and TensorFlow, and needs
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

**Data**: drop a folder containing `imgs/` and `masks/`. Masks are matched to images by
name (`img1.png` ↔ `img1.png`, `img1_mask.png`, `img1_label.png`…). A folder that holds only
images is accepted too: each image gets a black mask. Images are resized to 512×512, and
every non-black pixel of a mask counts as object. `data_examples/` holds a folder of each
kind to try it.

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

**Training**: choose the dataset, epochs, batch size, learning rate, validation split and
U-Net size. Loss, Accuracy and Recall, as well as IoU, are followed live for training and
validation. Precision is computed too and kept in the model file. The best model (lowest
validation loss) is saved after each epoch that improves it, so stopping a training keeps
the best model so far.

**Inference**: choose a model, or drop a `.pt`, `.keras` or `.h5` file, then drop images.
They can be of any size: the prediction is made at 512×512, then brought back to the image's
size. Results show as a mosaic, or stacked with a before/after slider. Image size (slider or
Ctrl + wheel), opacity, color and threshold can be adjusted. Results can be downloaded as a
zip, or copied into a dataset to be corrected in Annotation.

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

The **Download** button of each model card in Training saves a copy elsewhere. `data/`
(datasets, models, results, settings) is not versioned.

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
app/settings.py    settings (data/settings.json)
app/sysinfo.py     what the Info tab shows
static/            the interface (HTML, CSS, JavaScript with no dependency)
data_examples/     example folders to drop in the Data tab
data/              datasets, models, results (not versioned)
```

Dropped models are read without running code: `torch.load(weights_only=True)` for PyTorch
(models from this application, the `state_dict` of a U-Net with the same architecture, or
TorchScript), and `keras.models.load_model` in safe mode for `.keras` / `.h5`.

## Computation

On Windows, the PyTorch that gets installed is the CPU version. On a CPU, keep a small
network (8 or 16 base filters). Measured on a Xeon W-2223, per image and per epoch: 0.3 s
with 8 filters, 0.6 s with 16, 1.5 s with 32, so about 1 minute per epoch for 100 images
with 16 filters. To use a recent NVIDIA card, install the CUDA version of PyTorch: the
application then uses it automatically. On Linux, the default PyTorch is the CUDA version.

TensorFlow does not use GPUs on native Windows since version 2.11 (only under WSL2), so it
always computes on the CPU there. Its first loading takes up to half a minute; it is loaded
only when it is selected in Settings or a `.keras` model is used.
