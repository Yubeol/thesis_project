"""Device diagnostics include a real CUDA forward/backward kernel test."""
import argparse
import importlib.metadata
import json


def package_versions():
    return {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft", "accelerate")}


def check_device(requested="auto", fp16=False):
    import torch
    available = torch.cuda.is_available()
    if requested == "cuda" and not available:
        raise RuntimeError("CUDA required but unavailable. Check NVIDIA driver, Python version and CUDA-enabled torch wheel; CPU fallback is disabled.")
    device = "cuda" if requested != "cpu" and available else "cpu"
    info = {"device": device, "cuda_available": available, "torch_version": torch.__version__, "torch_cuda": torch.version.cuda}
    if device == "cuda":
        capability = torch.cuda.get_device_capability(0)
        info.update(gpu_name=torch.cuda.get_device_name(0), compute_capability=list(capability),
                    compiled_architectures=torch.cuda.get_arch_list(),
                    vram_mib=round(torch.cuda.get_device_properties(0).total_memory / 1024**2))
        if fp16 and capability[0] < 7:
            raise ValueError("fp16 is disabled on Pascal in this configuration; use float32/LoRA on GTX 1050")
        try:
            x = torch.ones((32, 32), device=device, requires_grad=True)
            (x @ x).sum().backward()
            torch.cuda.synchronize()
            del x
            torch.cuda.empty_cache()
        except RuntimeError:
            raise RuntimeError("CUDA kernel test failed. Check compiled_architectures against this GPU; GTX 1050 needs a Pascal-compatible wheel (tested configuration: torch 2.8.0 cu126, Python 3.12).") from None
    elif fp16:
        raise ValueError("--fp16 requires a supported CUDA GPU")
    print(json.dumps(info, indent=2), flush=True)
    return info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    try:
        check_device(args.device)
    except (RuntimeError, ValueError) as exc:
        parser.exit(1, str(exc) + "\n")


if __name__ == "__main__":
    main()
