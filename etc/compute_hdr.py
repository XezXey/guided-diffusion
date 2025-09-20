# reserve_gpu.py
import argparse, math, sys, time
try:
    import torch
except Exception as e:
    print("[!] PyTorch not found. Install it with CUDA support.", e)
    sys.exit(1)

def pretty_gb(bytes_):
    return bytes_ / (1024**3)

def allocate_vram(device, target_gb, block_gb, safety_frac, use_uint8):
    torch.cuda.set_device(device)
    free_b, total_b = torch.cuda.mem_get_info(device)
    print(f"[i] Device {device} total={pretty_gb(total_b):.2f} GB, free={pretty_gb(free_b):.2f} GB")

    max_bytes = int(free_b * safety_frac)
    want_bytes = int(target_gb * (1024**3))
    to_alloc = min(max_bytes, want_bytes)

    if to_alloc <= 0:
        print("[!] No free memory to allocate.")
        return []

    block_bytes = int(block_gb * (1024**3))
    block_bytes = max(block_bytes, 32 * 1024**2)  # >=32MB to avoid too many small tensors

    tensors = []
    allocated = 0
    dtype = torch.uint8 if use_uint8 else torch.float32
    elsize = torch.tensor([], dtype=dtype).element_size()
    print(f"[i] Allocating ~{pretty_gb(to_alloc):.2f} GB in blocks of ~{pretty_gb(block_bytes):.2f} GB (dtype={dtype})")

    while allocated < to_alloc:
        this_block = min(block_bytes, to_alloc - allocated)
        numel = math.ceil(this_block / elsize)
        try:
            t = torch.empty(numel, dtype=dtype, device=device)
            # Touch memory so it’s really committed
            t.fill_(1)
            tensors.append(t)
            allocated += t.numel() * elsize
            if len(tensors) % 2 == 0:
                free_b, _ = torch.cuda.mem_get_info(device)
                print(f"[i] Allocated {pretty_gb(allocated):.2f} GB, free now ~{pretty_gb(free_b):.2f} GB")
        except RuntimeError as e:
            print(f"[!] Allocation hit limit/OOM at ~{pretty_gb(allocated):.2f} GB: {e}")
            break

    free_b, _ = torch.cuda.mem_get_info(device)
    print(f"[✓] Reserved ~{pretty_gb(allocated):.2f} GB on device {device}. Free remains ~{pretty_gb(free_b):.2f} GB")
    return tensors

def keep_gpu_busy(device, matdim, duration_s, iters_per_report, sleep_ms):
    torch.cuda.set_device(device)
    print(f"[i] Starting busy loop: matmul {matdim}x{matdim}, duration ~{duration_s}s")
    # Pre-allocate work tensors to avoid extra allocations each step
    A = torch.randn((matdim, matdim), device=device)
    B = torch.randn((matdim, matdim), device=device)
    start = time.time()
    it = 0
    while time.time() - start < duration_s:
        C = A @ B  # matmul
        # A tiny non-linear step to ensure some compute
        A = torch.tanh(C)
        it += 1
        if it % iters_per_report == 0:
            torch.cuda.synchronize()
            elapsed = time.time() - start
            print(f"[i] Busy iters={it}, elapsed={elapsed:.1f}s")
        if sleep_ms > 0:
            time.sleep(sleep_ms / 1000.0)
    torch.cuda.synchronize()
    print("[✓] Busy loop finished.")

def main():
    p = argparse.ArgumentParser(description="Reserve GPU VRAM and keep GPU busy.")
    p.add_argument("--device", type=int, default=0, help="CUDA device index")
    p.add_argument("--target_gb", type=float, default=10.0, help="Target VRAM to reserve (GB)")
    p.add_argument("--block_gb", type=float, default=0.5, help="Allocation block size (GB)")
    p.add_argument("--safety_frac", type=float, default=0.92, help="Only allocate up to this fraction of reported free VRAM")
    p.add_argument("--uint8", action="store_true", help="Allocate as uint8 for exact-bytes control (denser); default float32")
    p.add_argument("--busy", action="store_true", help="Run a tiny compute loop to keep GPU active")
    p.add_argument("--busy_secs", type=int, default=3600, help="How long to keep GPU busy (seconds)")
    p.add_argument("--matdim", type=int, default=4096, help="Matrix size for matmul busywork")
    p.add_argument("--iters_per_report", type=int, default=20, help="Status print frequency")
    p.add_argument("--sleep_ms", type=int, default=0, help="Sleep between busy iterations (ms)")
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("[!] CUDA is not available. Exiting.")
        sys.exit(1)

    try:
        tensors = allocate_vram(
            device=args.device,
            target_gb=args.target_gb,
            block_gb=args.block_gb,
            safety_frac=args.safety_frac,
            use_uint8=args.uint8,
        )
        if args.busy:
            keep_gpu_busy(
                device=args.device,
                matdim=args.matdim,
                duration_s=args.busy_secs,
                iters_per_report=args.iters_per_report,
                sleep_ms=args.sleep_ms,
            )
        else:
            print("[i] Sleeping to hold allocations. Press Ctrl+C to release.")
            while True:
                time.sleep(60)
    except KeyboardInterrupt:
        print("\n[i] Interrupted. Releasing allocations…")
        tensors = []  # drop references
        torch.cuda.empty_cache()
        time.sleep(1)
        free_b, total_b = torch.cuda.mem_get_info(args.device)
        print(f"[i] Freed. Now free={pretty_gb(free_b):.2f} GB / total={pretty_gb(total_b):.2f} GB")

if __name__ == "__main__":
    main()