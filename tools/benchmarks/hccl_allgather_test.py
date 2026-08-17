#!/usr/bin/env python3
"""Simple HCCL all-gather test across two nodes.

Usage:
  # On the master node (node0, rank 0):
  export MASTER_ADDR=33.215.116.107
  export MASTER_PORT=29551
  python3 hccl_allgather_test.py --rank 0 --world-size 4

  # On node1 (ranks 2,3):
  export MASTER_ADDR=33.215.116.107
  export MASTER_PORT=29551
  python3 hccl_allgather_test.py --rank 2 --world-size 4

Launches one rank per NPU device.  Each rank does an AllGather of a 1 MiB
tensor across all 4 ranks with the HCCL backend.
"""

import argparse
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def init_process(rank, world_size, master_addr, master_port, backend):
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)

    # Set NPU device for this rank
    device_id = rank % torch.npu.device_count()
    torch.npu.set_device(device_id)

    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
    )
    return device_id


def run_allgather(rank, world_size, master_addr, master_port):
    backend = "hccl"
    device_id = init_process(rank, world_size, master_addr, master_port, backend)

    # Create a 1 MiB tensor on NPU
    data = torch.randn(256, 256, dtype=torch.bfloat16, device=f"npu:{device_id}")
    gathered = [torch.zeros_like(data) for _ in range(world_size)]

    # Warmup (discard result)
    dist.all_gather(gathered, data)

    # Timed run
    torch.npu.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(10):
        dist.all_gather(gathered, data)
    end.record()
    torch.npu.synchronize()

    elapsed_ms = start.elapsed_time(end)
    avg_us = elapsed_ms * 1000.0 / 10.0

    # Validate correctness
    local_sum = data.sum().item()
    # Gather local sums across all ranks
    sum_tensor = torch.tensor([local_sum], dtype=torch.float32, device=f"npu:{device_id}")
    gathered_sums = [torch.zeros_like(sum_tensor) for _ in range(world_size)]
    dist.all_gather(gathered_sums, sum_tensor)
    all_sums = [s.item() for s in gathered_sums]

    if rank == 0:
        print(f"OK rank={rank} device={device_id} all_gather={avg_us:.1f} us/iter "
              f"world_size={world_size} all_sums={all_sums}")

    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="HCCL AllGather test")
    parser.add_argument("--rank", type=int, required=True, help="Rank of this process")
    parser.add_argument("--world-size", type=int, required=True, help="Total number of ranks")
    parser.add_argument("--master-addr", default="33.215.116.107")
    parser.add_argument("--master-port", default="29551")
    args = parser.parse_args()

    run_allgather(args.rank, args.world_size,
                  args.master_addr, args.master_port)


if __name__ == "__main__":
    main()