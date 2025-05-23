import torch
import torch.distributed as dist
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int)
    args = parser.parse_args()

    torch.cuda.set_device(args.local_rank)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    print(f"Hello from rank {rank} on device {args.local_rank}")

if __name__ == "__main__":
    main()
