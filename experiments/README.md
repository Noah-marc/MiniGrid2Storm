# Experiment 1.0: Baseline Training Without Shield

This experiment trains PPO policies for all goal_state environments **without any shielding**.

## Environments Trained:
- CrossingEnv
- DistShiftEnv
- FourRoomsEnv
- LockedRoomEnv
- MultiRoomEnv
- lavagap_small

## Configuration:
- Algorithm: PPO (Proximal Policy Optimization)
- Total timesteps: 200,000 per environment
- Feature extractor: Custom CNN for MiniGrid
- Features dimension: 128
- Fixed seed: 42 (via ReseedWrapper)
- No shield applied during training

## Running the Experiment:

```bash
cd /home/noah/Noah_tmp/thesis/Minigrid2Storm
uv run experiments/1.0/train_all_goal_state_envs.py
```

## Output Structure:

For each environment, the following files are saved:

```
experiments/1.0/
├── {EnvName}/
│   ├── PPO_{EnvName}.zip          # Trained policy
│   ├── {EnvName}_training_plot.png # Performance visualization
│   └── ppo_logs/                   # Training logs (CSV, stdout)
│       ├── progress.csv
│       └── ...
```

## Results:

The training plots show:
- Left plot: Episode rewards over time (raw + rolling average)
- Right plot: Episode lengths over time (raw + rolling average)

Training statistics are printed to console for each environment.
