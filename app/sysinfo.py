"""What the Info tab shows: framework and library versions, GPUs, system."""
import ctypes
import os
import platform
import re
import subprocess
import sys
from functools import lru_cache
from importlib import metadata

import torch

from . import settings

# Without CREATE_NO_WINDOW, a server started with no console (the Desktop
# shortcut) would flash a terminal window at every nvidia-smi call.
_NO_WINDOW = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}

LIBRARIES = [
    ("PyTorch", "torch"), ("torchvision", "torchvision"), ("TensorFlow", "tensorflow"),
    ("Keras", "keras"), ("JAX", "jax"), ("ONNX Runtime", "onnxruntime"),
    ("NumPy", "numpy"), ("Pillow", "pillow"), ("FastAPI", "fastapi"), ("Uvicorn", "uvicorn"),
]

SMI_FIELDS = ["index", "name", "memory.total", "memory.used", "memory.free", "driver_version",
              "temperature.gpu", "utilization.gpu", "utilization.memory", "power.draw", "power.limit",
              "pci.bus_id", "pstate", "fan.speed", "clocks.gr", "clocks.max.gr"]


def _run(cmd, timeout=5):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, **_NO_WINDOW)
        return r.stdout if r.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None  # "[N/A]", "[Not Supported]"


def nvidia_gpus():
    """Live state of NVIDIA GPUs from nvidia-smi, or [] when there is none."""
    out = _run(["nvidia-smi", f"--query-gpu={','.join(SMI_FIELDS)}", "--format=csv,noheader,nounits"])
    if not out:
        return []
    gpus = []
    for line in out.strip().splitlines():
        v = dict(zip(SMI_FIELDS, [x.strip() for x in line.split(",")]))
        gpus.append({
            "index": int(v["index"]),
            "name": v["name"],
            "vendor": "NVIDIA",
            "memory_total_mb": _num(v["memory.total"]),
            "memory_used_mb": _num(v["memory.used"]),
            "memory_free_mb": _num(v["memory.free"]),
            "driver": v["driver_version"],
            "temperature_c": _num(v["temperature.gpu"]),
            "utilization_pct": _num(v["utilization.gpu"]),
            "memory_utilization_pct": _num(v["utilization.memory"]),
            "power_w": _num(v["power.draw"]),
            "power_limit_w": _num(v["power.limit"]),
            "pci_bus": v["pci.bus_id"],
            "pstate": v["pstate"],
            "fan_pct": _num(v["fan.speed"]),
            "clock_mhz": _num(v["clocks.gr"]),
            "clock_max_mhz": _num(v["clocks.max.gr"]),
        })
    return gpus


@lru_cache(maxsize=1)
def driver_cuda_version():
    """Highest CUDA version the installed NVIDIA driver supports."""
    out = _run(["nvidia-smi"])
    m = re.search(r"CUDA Version:\s*([\d.]+)", out or "")
    return m.group(1) if m else None


@lru_cache(maxsize=1)
def other_display_adapters():
    """Non-NVIDIA adapters (Intel, AMD...), Windows only. AdapterRAM is a
    32-bit field: it cannot report more than 4 GB."""
    if os.name != "nt":
        return []
    out = _run(["powershell", "-NoProfile", "-Command",
                "Get-CimInstance Win32_VideoController | ForEach-Object { $_.Name + '|' + $_.DriverVersion + '|' + $_.AdapterRAM }"],
               timeout=10)
    adapters = []
    for line in (out or "").strip().splitlines():
        parts = line.split("|")
        if len(parts) != 3 or "NVIDIA" in parts[0].upper():
            continue
        ram = _num(parts[2])
        adapters.append({"name": parts[0].strip(), "driver": parts[1].strip(),
                         "memory_total_mb": ram / 2**20 if ram else None, "vendor": "other"})
    return adapters


def _cpu_name():
    if os.name == "nt":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
                return winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or None


def _ram_mb():
    if os.name == "nt":
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        s = MEMORYSTATUSEX()
        s.dwLength = ctypes.sizeof(s)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s)):
            return s.ullTotalPhys / 2**20, s.ullAvailPhys / 2**20
    try:
        info = {}
        with open("/proc/meminfo") as f:
            for line in f:
                k, v = line.split(":", 1)
                info[k] = int(v.split()[0]) / 1024
        return info["MemTotal"], info.get("MemAvailable")
    except (OSError, KeyError, ValueError):
        return None, None


def _version(dist):
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return None


def torch_info():
    cuda_ok = torch.cuda.is_available()
    built_cuda = torch.version.cuda
    info = {
        "id": "pytorch",
        "name": "PyTorch",
        "installed": True,
        "version": torch.__version__,
        "build": f"CUDA {built_cuda}" if built_cuda else ("ROCm " + torch.version.hip if getattr(torch.version, "hip", None) else "CPU only"),
        "cudnn": torch.backends.cudnn.version() if cuda_ok else None,
        "cuda_available": cuda_ok,
        "threads": torch.get_num_threads(),
        "arch_list": torch.cuda.get_arch_list() if built_cuda else [],
        "devices": [],
    }
    if cuda_ok:
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            free, total = torch.cuda.mem_get_info(i)
            info["devices"].append({
                "index": i, "name": p.name, "capability": f"{p.major}.{p.minor}",
                "memory_total_mb": total / 2**20, "memory_free_mb": free / 2**20,
                "multiprocessors": p.multi_processor_count,
            })
    from .training import pick_device
    info["device_used"] = str(pick_device())
    if cuda_ok:
        info["why"] = None
    elif not built_cuda:
        info["why"] = "This PyTorch build has no CUDA support, so training and inference run on the CPU."
    else:
        info["why"] = "PyTorch has CUDA support but sees no usable GPU (driver too old or GPU not supported by this build)."
    return info


@lru_cache(maxsize=1)
def _static_system():
    total, _ = _ram_mb()
    return {
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "machine": platform.machine(),
        "cpu": _cpu_name(),
        "cpu_cores": os.cpu_count(),
        "python": sys.version.split()[0],
        "python_path": sys.executable,
        "ram_total_mb": total,
    }


def system():
    s = dict(_static_system())
    s["ram_available_mb"] = _ram_mb()[1]
    return s


def tf_info():
    """TensorFlow details need importing it (seconds, hundreds of MB): done
    only when it is the selected framework or already in use."""
    version = _version("tensorflow")
    out = {"id": "tensorflow", "name": "TensorFlow", "installed": version is not None, "version": version,
           "loaded": False}
    if not version:
        out["why"] = "TensorFlow is not installed in this environment."
        return out
    if "app.tf_backend" in sys.modules or settings.get()["framework"] == "tensorflow":
        from . import tf_backend
        i = tf_backend.info()
        out.update(loaded=True, version=i["version"], keras=i["keras"], gpus=i["gpus"],
                   build="CUDA" if i["built_with_cuda"] else "CPU only", device_used=i["device_used"],
                   cuda_available=bool(i["gpus"]))
        if not i["gpus"] and os.name == "nt":
            out["why"] = ("TensorFlow does not use GPUs on native Windows since version 2.11 "
                          "(only under WSL2), so it runs on the CPU.")
        elif not i["gpus"]:
            out["why"] = "TensorFlow sees no usable GPU, so it runs on the CPU."
    return out


def frameworks():
    return {"selected": settings.get()["framework"], "list": [torch_info(), tf_info()]}


def everything():
    return {
        "frameworks": frameworks(),
        "libraries": [{"name": n, "package": p, "version": _version(p)} for n, p in LIBRARIES],
        "gpus": nvidia_gpus() + other_display_adapters(),
        "driver_cuda": driver_cuda_version(),
        "system": system(),
    }
