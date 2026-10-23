"""Sample memory without counting shared CPU/GPU DRAM as separate pools."""


def configure_device(cfg):
    import torch
    from camera_rc.training import _device
    device = _device(cfg.get("device", "auto"))
    if device.type == "cuda":
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        total = torch.cuda.get_device_properties(device).total_memory
        cap = cfg["runtime"]["torch_allocator_cap_gib"] * 2**30
        torch.cuda.set_per_process_memory_fraction(min(cap / total, 0.9), device)
        torch.cuda.reset_peak_memory_stats(device)
    return str(device)


class ResourceMonitor:
    def __init__(self, device, memory_architecture="discrete"):
        import psutil
        if memory_architecture not in ("discrete", "shared"):
            raise ValueError("Unknown memory architecture")
        self.process = psutil.Process()
        self.device = device
        self.memory_architecture = memory_architecture
        self.peak_rss = 0
        self.system_available_min = None
        self.system_available = None
        self.peak_gpu_process = None
        self.peak_gpu_device = None
        self.initial_gpu_device = None
        self.nvml = self.handle = None
        if device.startswith("cuda"):
            try:
                import pynvml
                pynvml.nvmlInit()
                self.nvml = pynvml
                # Match CUDA by PCI bus ID where possible; do not assume remapped indices.
                import torch
                properties = torch.cuda.get_device_properties(device)
                uuid = getattr(properties, "uuid", None)
                if uuid is not None:
                    self.handle = pynvml.nvmlDeviceGetHandleByUUID(str(uuid))
            except Exception:
                self.nvml = self.handle = None
        self.sample()
        self.initial_gpu_device = self.peak_gpu_device

    def sample(self):
        import psutil
        rss = self.process.memory_info().rss
        self.peak_rss = max(self.peak_rss, rss)
        self.system_available = psutil.virtual_memory().available
        self.system_available_min = (self.system_available if self.system_available_min is None
                                     else min(self.system_available_min, self.system_available))
        if self.device.startswith("cuda"):
            try:
                import torch
                free, total = torch.cuda.mem_get_info(self.device)
                self.peak_gpu_device = max(self.peak_gpu_device or 0, total - free)
            except Exception:
                pass
        if self.nvml is not None and self.handle is not None:
            try:
                processes = self.nvml.nvmlDeviceGetComputeRunningProcesses(self.handle)
                values = [p.usedGpuMemory for p in processes if p.pid == self.process.pid and isinstance(p.usedGpuMemory, int) and 0 <= p.usedGpuMemory < 2**50]
                if values:
                    self.peak_gpu_process = max(self.peak_gpu_process or 0, max(values))
            except Exception:
                pass  # Windows WDDM may not expose per-process GPU usage.
        return rss / 2**30

    def check_limits(self, runtime):
        """Fail before publishing a new plan; a sampled guard cannot prevent every OOM."""
        rss_gib = self.sample()
        if rss_gib > runtime["process_ram_limit_gib"]:
            raise RuntimeError("process_ram_budget_exceeded")
        if self.system_available / 2**30 < runtime["system_available_min_gib"]:
            raise RuntimeError("system_available_ram_below_floor")

    def report(self):
        import torch
        self.sample()
        cuda = self.device.startswith("cuda")
        total_cuda = torch.cuda.get_device_properties(self.device).total_memory / 2**30 if cuda else None
        shared = self.memory_architecture == "shared"
        return {"device": self.device,
                "memory_architecture": self.memory_architecture,
                "memory_architecture_source": "configuration; not hardware detection",
                "gpu_name": torch.cuda.get_device_name(self.device) if cuda else None,
                "cuda_addressable_memory_gib": total_cuda,
                "physical_vram_gib": None if shared else total_cuda,
                "process_rss_peak_gib_sampled": self.peak_rss / 2**30,
                "system_available_min_gib_sampled": self.system_available_min / 2**30,
                "process_gpu_peak_gib_sampled": self.peak_gpu_process / 2**30 if self.peak_gpu_process is not None else None,
                "whole_device_used_peak_gib_sampled": self.peak_gpu_device / 2**30 if self.peak_gpu_device is not None else None,
                "whole_device_used_at_monitor_start_gib": self.initial_gpu_device / 2**30 if self.initial_gpu_device is not None else None,
                "torch_peak_allocated_gib": torch.cuda.max_memory_allocated(self.device) / 2**30 if cuda else None,
                "torch_peak_reserved_gib": torch.cuda.max_memory_reserved(self.device) / 2**30 if cuda else None,
                "system_ram_gib": __import__("psutil").virtual_memory().total / 2**30,
                "measurement_note": "Memory architecture comes from the profile. On shared-memory devices CPU and CUDA use the same DRAM: do not add RSS and GPU usage or interpret CUDA total as dedicated VRAM. PyTorch allocator cap excludes CUDA context and other libraries; null process GPU memory is unavailable, not zero. Whole-device usage includes the OS/other processes and is not attributable to this model. Samples may miss transient peaks; available-RAM checks cannot guarantee prevention of OOM."}


def latency_report(values):
    import numpy as np
    if not values:
        return {"frames": 0}
    return {"frames": len(values), "mean_ms": float(np.mean(values)),
            "p50_ms": float(np.percentile(values, 50)), "p95_ms": float(np.percentile(values, 95)),
            "p99_ms": float(np.percentile(values, 99)), "max_ms": float(max(values))}
