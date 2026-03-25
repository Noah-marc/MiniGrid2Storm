# Minigrid2Storm

A framework for training shielded reinforcement learning policies on MiniGrid environments, using [Stormvogel](https://github.com/moves-rwth/stormvogel) for probabilistic model construction and formal verification.

The core idea is to wrap MiniGrid environments in a probabilistic model, compute safety properties via model checking, and use the resulting shield to restrict the agent's actions during training.

## Project Structure

```
Minigrid2Storm/
│
├── core/                          # Core framework logic
│   ├── probabilistic_minigrids.py #   ProbabilisticEnvWrapper: wraps a MiniGrid env with
│   │                              #   probabilistic action outcomes and builds a Stormvogel MDP
│   └── shield.py                  #   Shield (base class) and DeltaShield: block actions
│                                  #   whose lava-reach probability exceeds a delta threshold
│
├── envs/                          # Environment registration and loading
│   ├── registry.py                #   Registers environments with Gymnasium from YAML configs
│   ├── loader.py                  #   Instantiates and wraps environments from config dicts
│   └── configs/                   #   YAML configuration files for each environment
│       ├── goal_state/            #     Envs with an explicit goal state (used in experiments)
│       │   ├── CrossingEnv.yaml
│       │   ├── DistShiftEnv.yaml
│       │   ├── FourRoomsEnv.yaml
│       │   ├── LavaGapEnv.yaml
│       │   ├── LockedRoomEnv.yaml
│       │   └── MultiRoomEnv.yaml
│       └── no_goal_state/         #     Envs without an explicit goal state
│
├── experiments/                   # Everything related to running experiments
│   ├── scripts/                   #   Training entry-point scripts (run via uv run or the shell script)
│   │   ├── run_all_experiments.sh #     Runs all three training scripts in sequence
│   │   ├── train_multiple_envs_no_shield.py
│   │   ├── train_multiple_envs_with_shield_instant_turn_off.py
│   │   └── train_multiple_envs_shield_gradual_reduction.py
│   ├── tests/
│   │   └── smoke_test.py          #   Quick sanity check (10 timesteps) for all three scripts
│   ├── callbacks.py               #   SB3 callbacks (e.g. ShieldHardCutoffCallback)
│   ├── extra_plotting.py          #   Additional plotting utilities for experiment results
│   ├── feature_extractor.py       #   Custom CNN feature extractor for PPO (MinigridFeaturesExtractor)
│   └── train_utils.py             #   Shared helpers: DummyVecEnvRenderSubset, video trigger, env image saving
│
├── utils/                         # General-purpose utilities
│   ├── action_utils.py            #   Conversion functions between MiniGrid, Gymnasium, and Stormvogel action formats
│   ├── logging_config.py          #   Centralized logging setup (console + rotating file handler)
│   ├── plotting.py                #   Plotting helpers
│   └── utils.py                   #   Miscellaneous utilities
│
├── logs/                          #   Log files and environment images (gitkeep'd, populated at runtime)
├── training_RL_policies.ipynb     # Interactive notebook: walkthrough of environment setup and training
├── pyproject.toml                 # Project metadata and dependencies (managed by uv)
└── README.md
```

## Installing the project

### Dependencies
If you want to use the scripts for training, then you need ffmpeg installed on your system.

Stormvogel uses the cairosvg library for visualization and uses it as a default dependency, even if the visualization tools are not used. At the time of writing, the imports are not defined lazily, so if you run into issues due to the cairo library make sure it is installed on your system.

### Uv
The project uses uv for managing the dependencies. You can see the installation guidelines on https://docs.astral.sh/uv/getting-started/installation/

Run `uv sync` in the project root to install all necessary dependencies.

## Running the experiments

Run all experiments in sequence from the project root:

```bash
bash experiments/scripts/run_all_experiments.sh --output_dir my_run
```

Or run individual scripts directly:

```bash
uv run experiments/scripts/train_multiple_envs_no_shield.py --output_dir my_run
uv run experiments/scripts/train_multiple_envs_with_shield_instant_turn_off.py --output_dir my_run
uv run experiments/scripts/train_multiple_envs_shield_gradual_reduction.py --mechanism delta --output_dir my_run
```

Results are saved under `experiments/scripts/output/<output_dir>/`.

## Running the smoke test

To quickly verify the setup still works after changes:

```bash
uv run experiments/tests/smoke_test.py
```

This runs 10 timesteps of each training script on a single environment.