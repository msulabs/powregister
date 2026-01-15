"""
CLI entry point for Bittensor PoW Registration Tool
"""

import argparse
import sys
import time

from powregister.core import (
    CUDA_AVAILABLE,
    SubtensorPowRegistration,
    check_cuda_available,
    create_seal_hash,
    get_cuda_device_count,
    hash_block_with_hotkey,
    seal_meets_difficulty,
    solve_pow_cuda,
    solve_pow_parallel,
)


def _parse_netuids(netuids_str: str) -> list[int]:
    """Parse netuids string like '1,2,3' or '0-20' or '1,5,10-15'"""
    result = []
    for part in netuids_str.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            result.extend(range(int(start), int(end) + 1))
        else:
            result.append(int(part))
    return sorted(set(result))


def cmd_info(args):
    """Show subnet info"""
    client = SubtensorPowRegistration(network=args.network)
    print(f"\nConnected to: {client.substrate_url}")

    if args.netuid is not None:
        client.print_subnet_params(args.netuid)
    else:
        netuids = _parse_netuids(args.netuids) if args.netuids else None
        client.print_registration_costs(
            netuids=netuids,
            min_burn_tao=args.min_burn_tao,
            pow_only=args.pow_only,
            max_difficulty=args.max_difficulty,
        )


def cmd_register(args):
    """Run PoW registration"""
    import bittensor as bt

    # Load wallet and unlock keys upfront (before PoW solving)
    wallet = bt.Wallet(name=args.wallet, hotkey=args.hotkey)

    print(f"\nWallet: {args.wallet}")
    print(f"Hotkey: {args.hotkey}")
    print(f"Coldkey: {wallet.coldkeypub.ss58_address}")
    print(f"Hotkey Address: {wallet.hotkey.ss58_address}")

    # Unlock coldkey now so it doesn't ask during/after PoW
    print("\nUnlocking coldkey...")
    wallet.unlock_coldkey()
    print("Coldkey unlocked.")

    # Create client
    client = SubtensorPowRegistration(network=args.network)
    print(f"Network: {client.network_name} ({client.substrate_url})")

    # Fund hotkey early — transfer before PoW so fees are ready
    try:
        client._ensure_hotkey_funded(wallet)
    except Exception as e:
        print(f"Error funding hotkey: {e}")
        return 1

    # Show subnet info
    client.print_subnet_params(args.netuid)

    # Confirm
    if not args.yes:
        confirm = input("\nProceed with registration? [y/N]: ")
        if confirm.lower() != "y":
            print("Cancelled.")
            return 1

    # Register
    success = client.solve_and_register(
        netuid=args.netuid,
        wallet=wallet,
        use_cuda=args.cuda,
        dev_id=args.dev_id,
        tpb=args.tpb,
        num_processes=args.processes,
        update_interval=args.update_interval,
        max_attempts=args.max_attempts,
        timeout=args.timeout,
    )

    return 0 if success else 1


def cmd_test(args):
    """Run test to verify setup"""
    print("=" * 60)
    print("Bittensor PoW Registration Test")
    print("=" * 60)

    # CUDA status
    print("\n=== CUDA Status ===")
    cuda_status = check_cuda_available()
    print(f"cubit available: {cuda_status['cubit_available']}")
    print(f"torch CUDA available: {cuda_status['torch_cuda_available']}")
    print(f"Recommended solver: {cuda_status['recommended']}")
    if cuda_status["torch_cuda_available"]:
        print(f"CUDA devices: {get_cuda_device_count()}")

    # Hash computation test
    print("\n=== Hash Computation Test ===")
    test_block_hash = bytes(32)  # 32 zero bytes
    test_hotkey = bytes(32)
    test_nonce = 12345
    test_difficulty = 10_000_000

    block_and_hotkey = hash_block_with_hotkey(test_block_hash, test_hotkey)
    print(f"Block+Hotkey Hash: {block_and_hotkey.hex()[:32]}...")

    seal = create_seal_hash(block_and_hotkey, test_nonce)
    print(f"Seal Hash: {seal.hex()[:32]}...")

    meets = seal_meets_difficulty(seal, test_difficulty)
    print(f"Meets difficulty {test_difficulty:,}: {meets}")

    # CPU PoW test
    print("\n=== CPU PoW Solve Test (low difficulty) ===")
    low_difficulty = 100
    start_time = time.time()

    from powregister.core import solve_pow_single_thread

    solution = solve_pow_single_thread(
        block_and_hotkey_hash=block_and_hotkey,
        difficulty=low_difficulty,
        block_number=1000,
        nonce_start=0,
        nonce_range=100_000,
    )

    elapsed = time.time() - start_time
    if solution:
        print(f"Solution found in {elapsed:.3f}s")
        print(f"Nonce: {solution.nonce}")
        print(f"Seal: {solution.seal.hex()[:32]}...")
    else:
        print(f"No solution found in {elapsed:.3f}s")

    # Parallel CPU test
    print("\n=== Parallel CPU PoW Test ===")
    start_time = time.time()

    solution = solve_pow_parallel(
        block_and_hotkey_hash=block_and_hotkey,
        difficulty=1000,
        block_number=1000,
        num_processes=4,
        update_interval=50_000,
    )

    elapsed = time.time() - start_time
    if solution:
        print(f"Solution found in {elapsed:.3f}s")
        print(f"Nonce: {solution.nonce}")
    else:
        print(f"No solution found in {elapsed:.3f}s")

    # CUDA test
    if CUDA_AVAILABLE:
        print("\n=== CUDA PoW Test ===")
        start_time = time.time()

        solution = solve_pow_cuda(
            block_and_hotkey_hash=block_and_hotkey,
            difficulty=10_000,
            block_number=1000,
            dev_id=0,
            tpb=256,
            update_interval=100_000,
            timeout=60,
        )

        elapsed = time.time() - start_time
        if solution:
            print(f"CUDA solution found in {elapsed:.3f}s")
            print(f"Nonce: {solution.nonce}")
        else:
            print(f"No CUDA solution found in {elapsed:.3f}s")

    # Network connectivity test
    if args.network:
        print(f"\n=== Network Connectivity Test ({args.network}) ===")
        try:
            client = SubtensorPowRegistration(network=args.network)
            block = client.get_current_block()
            print(f"Connected to: {client.substrate_url}")
            print(f"Current block: {block}")
            print("Network connectivity: OK")
        except Exception as e:
            print(f"Network connectivity: FAILED ({e})")

    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)

    return 0


def cmd_status(args):
    """Show CUDA/system status"""
    print("=== System Status ===\n")

    # CUDA
    cuda_status = check_cuda_available()
    print(f"cubit (CUDA solver): {'Available' if cuda_status['cubit_available'] else 'Not installed'}")
    print(f"PyTorch CUDA: {'Available' if cuda_status['torch_cuda_available'] else 'Not available'}")

    if cuda_status["torch_cuda_available"]:
        try:
            import torch

            print(f"CUDA devices: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  [{i}] {torch.cuda.get_device_name(i)}")
        except ImportError:
            print("CUDA devices: (torch not available for detailed info)")

    print(f"\nRecommended solver: {cuda_status['recommended']}")

    # CPU
    import multiprocessing as mp

    print(f"\nCPU cores: {mp.cpu_count()}")

    return 0


def main():
    parser = argparse.ArgumentParser(
        prog="powregister",
        description="Bittensor Subnet PoW Registration Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show all subnet registration costs on mainnet
  powregister info --network mainnet

  # Show specific subnet info on testnet
  powregister info --network testnet --netuid 1

  # Register on testnet with CPU
  powregister register --network testnet --netuid 1 --wallet default --hotkey default

  # Register on mainnet with GPU
  powregister register --network mainnet --netuid 1 --wallet mywallet --hotkey myhotkey --cuda

  # Run tests
  powregister test --network testnet

  # Check system status
  powregister status
        """,
    )

    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # info command
    info_parser = subparsers.add_parser("info", help="Show subnet registration info")
    info_parser.add_argument(
        "--network", "-n", default="mainnet", help="Network: mainnet, testnet, local, or custom URL (default: mainnet)"
    )
    info_parser.add_argument("--netuid", "-u", type=int, help="Show detailed info for a single subnet")
    info_parser.add_argument("--netuids", type=str, help="Subnet IDs to query: '1,2,3' or '0-20' or '1,5,10-15'")
    info_parser.add_argument(
        "--min-burn-tao",
        type=float,
        default=0.001,
        help="Filter to burn cost greater than this TAO amount (default: 0.001)",
    )
    info_parser.add_argument(
        "--max-difficulty",
        type=int,
        default=10_000_000,
        help="Filter to difficulty at or below this value (default: 10,000,000; use 0 to disable)",
    )
    info_parser.add_argument("--pow-only", action="store_true", help="Show only subnets with PoW registration enabled")
    info_parser.set_defaults(func=cmd_info)

    # register command
    reg_parser = subparsers.add_parser("register", help="Run PoW and register to subnet")
    reg_parser.add_argument(
        "--network", "-n", default="mainnet", help="Network: mainnet, testnet, local, or custom URL (default: mainnet)"
    )
    reg_parser.add_argument("--netuid", "-u", type=int, required=True, help="Subnet ID to register")
    reg_parser.add_argument("--wallet", "-w", default="default", help="Wallet name (default: default)")
    reg_parser.add_argument("--hotkey", "-k", default="default", help="Hotkey name (default: default)")
    reg_parser.add_argument("--cuda", action="store_true", help="Use CUDA (GPU) for solving")
    reg_parser.add_argument("--dev-id", type=int, nargs="+", default=[0], help="CUDA device ID(s) (default: 0)")
    reg_parser.add_argument("--tpb", type=int, default=256, help="Threads per block for CUDA (default: 256)")
    reg_parser.add_argument("--processes", "-p", type=int, help="Number of CPU processes (default: all cores)")
    reg_parser.add_argument(
        "--update-interval", type=int, default=50_000, help="Nonce update interval (default: 50000)"
    )
    reg_parser.add_argument("--max-attempts", type=int, default=3, help="Maximum registration attempts (default: 3)")
    reg_parser.add_argument("--timeout", type=int, default=3600, help="PoW solve timeout in seconds (default: 3600)")
    reg_parser.add_argument("-y", "--yes", action="store_true", help="Skip confirmation prompt")
    reg_parser.set_defaults(func=cmd_register)

    # test command
    test_parser = subparsers.add_parser("test", help="Run tests to verify setup")
    test_parser.add_argument("--network", "-n", help="Optional: test network connectivity (mainnet, testnet, local)")
    test_parser.set_defaults(func=cmd_test)

    # status command
    status_parser = subparsers.add_parser("status", help="Show system/CUDA status")
    status_parser.set_defaults(func=cmd_status)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
