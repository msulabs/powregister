<p align="center">
  <h1 align="center">powregister</h1>
  <p align="center">
    <strong>High-performance Proof-of-Work registration tool for Bittensor subnets</strong>
  </p>
  <p align="center">
    <a href="#features">Features</a> &middot;
    <a href="#quick-start">Quick Start</a> &middot;
    <a href="#usage">Usage</a> &middot;
    <a href="#python-api">Python API</a> &middot;
    <a href="#how-it-works">How It Works</a>
  </p>
</p>

---

**powregister** is an open-source CLI tool that lets you register neurons to [Bittensor](https://bittensor.com/) subnets using Proof-of-Work — no TAO burn required. It supports both CPU and CUDA (GPU) solving, parallel processing across all cores, multi-GPU setups, and automatic network resilience with RPC retry logic.

Built from the ground up by analyzing the `subtensor`, `btcli`, and `bittensor` source code.

## Features

- **CPU & CUDA (GPU) Solver** — Parallel CPU solving across all cores, or blazing-fast CUDA solving via [cubit](https://github.com/opentensor/cubit)
- **Multi-GPU Support** — Distribute workload across multiple CUDA devices
- **Subnet Discovery** — Query and filter all subnets by PoW availability, difficulty, burn cost, and slot capacity
- **Auto Hotkey Funding** — Automatically transfers minimum balance to hotkey before PoW solving
- **Network Resilience** — Exponential backoff retry on all RPC calls with automatic connection recovery
- **Mainnet / Testnet / Local** — Works with any Bittensor network endpoint
- **Python API** — Use as a library in your own scripts and automation

## Quick Start

### Prerequisites

You need a Bittensor wallet with a coldkey and hotkey. Create one using [btcli](https://github.com/opentensor/btcli):

```bash
# Install btcli
pip install bittensor-cli

# Create a new wallet (coldkey + hotkey)
btcli wallet create --wallet.name mywallet --wallet.hotkey myhotkey

# Or if you already have a coldkey, just create a new hotkey
btcli wallet new-hotkey --wallet.name mywallet --wallet.hotkey myhotkey
```

> **Note:** Your coldkey needs a small TAO balance (~0.01 TAO) for transaction fees. powregister automatically transfers the minimum required amount from your coldkey to your hotkey before solving.

### Install

```bash
# With uv (recommended)
uv sync

# Or with pip
pip install -e .

# For CUDA/GPU support
pip install -e ".[cuda]"
```

## Usage

### Discover Subnets

Find the best subnets to register on — filter by PoW availability, difficulty, and more:

```bash
# Show all PoW-enabled subnets on mainnet
powregister info --network mainnet --netuids "0-50" --pow-only

# Filter by difficulty (easier registration)
powregister info --network mainnet --netuids "0-50" --max-difficulty 100000000

# Detailed view for a single subnet
powregister info --network mainnet --netuid 18
```

**Output columns:**

| Column | Description |
|--------|-------------|
| NetUID | Subnet ID |
| Name | Subnet name |
| Difficulty | PoW difficulty (higher = harder) |
| Burn (TAO) | Cost to register via burn |
| Slots | Available / Maximum neuron capacity |
| PoW | PoW registration enabled |
| Reg | Registration open |

### Register to a Subnet

```bash
# Register with CPU (uses all cores)
powregister register --network mainnet --netuid 1 --wallet mywallet --hotkey myhotkey

# Register with GPU (CUDA)
powregister register --network mainnet --netuid 1 --wallet mywallet --hotkey myhotkey --cuda

# Multi-GPU registration
powregister register --network mainnet --netuid 1 --wallet mywallet --hotkey myhotkey \
  --cuda --dev-id 0 1

# Skip confirmation prompt
powregister register --network testnet --netuid 1 --wallet default --hotkey default -y
```

### System Status

```bash
# Check CUDA availability, GPU info, CPU cores
powregister status

# Full diagnostic test (hash computation, CPU solver, CUDA solver, network)
powregister test --network testnet
```

## Python API

Use powregister as a library for custom automation and scripting:

```python
from powregister.core import SubtensorPowRegistration

# Connect to mainnet
client = SubtensorPowRegistration(network="mainnet")

# Discover subnets
info = client.get_registration_info(netuid=18)
print(f"Difficulty: {info['difficulty']:,}")
print(f"Burn cost: {info['burn_cost_tao']} TAO")
print(f"Slots: {info['slots_available']}/{info['max_neurons']}")

# Get detailed parameters
params = client.get_subnet_params(netuid=18)
print(f"Max Validators: {params['max_validators']}")

# Scan all subnets
all_subnets = client.get_all_subnet_registration_costs(netuids=list(range(50)))
pow_subnets = [s for s in all_subnets if s.get("pow_allowed")]
for s in pow_subnets:
    print(f"  Subnet {s['netuid']}: difficulty={s['difficulty']:,}")
```

## How It Works

```
┌─────────────┐     ┌──────────────┐     ┌───────────────┐
│  Query Chain │────>│  Solve PoW   │────>│   Submit      │
│  (block hash,│     │  (CPU/CUDA)  │     │   Extrinsic   │
│   difficulty)│     │              │     │   (register)  │
└─────────────┘     └──────────────┘     └───────────────┘
       │                    │                     │
       ▼                    ▼                     ▼
  substrate-         keccak-256 hash       Signed extrinsic
  interface          + nonce search        to Bittensor chain
  RPC calls          parallel/CUDA
```

1. **Query** the chain for the current block hash, difficulty target, and subnet parameters
2. **Solve** the PoW puzzle by finding a nonce where `keccak256(block_hash + hotkey + nonce)` meets the difficulty
3. **Submit** the solution as a signed extrinsic to register the neuron on-chain

The solver automatically handles block updates, difficulty changes, and connection drops during long-running solves.

## Architecture

```
powregister/
├── core.py     # PoW solver engine, chain queries, registration logic
├── cli.py      # CLI interface (argparse)
└── pyproject.toml
```

| Component | Details |
|-----------|---------|
| Chain Interface | `substrateinterface` — direct RPC to Bittensor nodes |
| Metadata | `bittensor` SDK — subnet names, wallet management, transfers |
| Hashing | `pycryptodome` keccak-256 |
| CUDA Solver | `cubit` — GPU-accelerated nonce search |
| CPU Solver | `multiprocessing` — parallel nonce search across all cores |

## Networks

| Network | Endpoint |
|---------|----------|
| mainnet | `wss://entrypoint-finney.opentensor.ai:443` |
| testnet | `wss://test.finney.opentensor.ai:443` |
| local | `ws://127.0.0.1:9944` |

## Requirements

- Python 3.11+
- For GPU: NVIDIA GPU with CUDA support + [cubit](https://github.com/opentensor/cubit) + PyTorch

## License

MIT

## Contributing

Contributions are welcome! Feel free to open issues and pull requests.
