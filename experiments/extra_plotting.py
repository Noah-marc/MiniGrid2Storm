""" This script generates extra plots for the experiments. These plots are based on the logs of different scripts"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Environments to look for in the output directory
KNOWN_ENVS = ["CrossingEnv", "DistShiftEnv", "LavaGapEnv"]


def plot_training_comparison(dir_name: str) -> None:
    """Generate per-environment training comparison plots for all script versions
    found under ``experiments/output/<dir_name>``.

    The function walks the given directory, auto-discovers every
    ``ppo_logs/progress.csv`` file and groups the data by environment name and
    script-version label.  For every environment it creates one plot with a
    separate line per script version (``ep_rew_mean`` vs ``total_timesteps``)
    and saves it as a PNG into ``experiments/output/<dir_name>/``.

    Directory layout handled
    ------------------------
    Flat versions (the env folder is a direct child of the version folder)::

        <dir_name>/<version>/<EnvName>/ppo_logs/progress.csv

    Nested versions (an extra sub-variant level exists)::

        <dir_name>/<top_level>/<variant>/<EnvName>/ppo_logs/progress.csv

    Both layouts can co-exist; each unique path prefix before the env name is
    used as the version label (e.g. ``shielded_gradual_reduction/delta``).

    Parameters
    ----------
    dir_name:
        Name of the directory inside ``experiments/output/`` to process
        (e.g. ``"1.0"``).

    Raises
    ------
    FileNotFoundError
        If ``experiments/output/<dir_name>`` does not exist.
    """
    base_dir, data = _collect_data(dir_name)

    if not data:
        print(f"No usable training data found in {base_dir}.")
        return

    # ------------------------------------------------------------------
    # Create one plot per environment
    # ------------------------------------------------------------------
    for env_name in KNOWN_ENVS:
        if env_name not in data:
            print(f"No data for {env_name}, skipping.")
            continue

        versions = data[env_name]
        fig, ax = plt.subplots(figsize=(10, 6))

        for version_label, (timesteps, ep_rew_mean) in sorted(versions.items()):
            ax.plot(timesteps, ep_rew_mean, label=version_label, linewidth=1.5)

        ax.set_xlabel("Total Timesteps")
        ax.set_ylabel("Mean Episode Reward (last 100 episodes)")
        ax.set_title(f"{env_name} — Training Comparison")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

        output_path = base_dir / f"{env_name}_training_comparison.png"
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {output_path}")

def _collect_data(dir_name: str) -> tuple[Path, dict[str, dict[str, tuple]]]:
    """Shared data-collection logic used by both plotting functions.

    Returns the resolved ``base_dir`` and a nested dict
    ``{env_name: {version_label: (timesteps, ep_rew_mean)}}``.
    """
    script_dir = Path(__file__).parent
    base_dir = script_dir / "output" / dir_name

    if not base_dir.exists():
        raise FileNotFoundError(
            f"Directory not found: {base_dir}\n"
            f"Make sure '{dir_name}' exists inside experiments/output/."
        )

    data: dict[str, dict[str, tuple]] = {}

    for progress_file in sorted(base_dir.rglob("ppo_logs/progress.csv")):
        rel_parts = progress_file.relative_to(base_dir).parts

        env_name: str | None = None
        env_idx: int | None = None
        for i, part in enumerate(rel_parts):
            match = next((env for env in KNOWN_ENVS if env in part), None)
            if match:
                env_name = match
                env_idx = i
                break

        if env_name is None or env_idx is None:
            print(f"Warning: could not identify environment in path {progress_file}, skipping.")
            continue

        version_label = os.path.join(*rel_parts[:env_idx]) if env_idx > 0 else "(root)"

        try:
            df = pd.read_csv(progress_file)
        except Exception as exc:
            print(f"Warning: could not read {progress_file}: {exc}")
            continue

        required_cols = {"time/total_timesteps", "rollout/ep_rew_mean"}
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            print(f"Warning: {progress_file} is missing columns {missing}, skipping.")
            continue

        timesteps = df["time/total_timesteps"].to_numpy()
        ep_rew_mean = df["rollout/ep_rew_mean"].to_numpy()

        data.setdefault(env_name, {})[version_label] = (timesteps, ep_rew_mean)

    return base_dir, data


def plot_training_comparison_subplots(dir_name: str) -> None:
    """Same as :func:`plot_training_comparison` but uses one subplot per script
    version instead of overlaying all versions on a single set of axes.

    For every environment a single figure is saved, where each column is a
    separate script version.  All subplots share the same y-axis scale to make
    visual comparisons straightforward.

    Parameters
    ----------
    dir_name:
        Name of the directory inside ``experiments/output/`` to process
        (e.g. ``"1.0"``).

    Raises
    ------
    FileNotFoundError
        If ``experiments/output/<dir_name>`` does not exist.
    """
    base_dir, data = _collect_data(dir_name)

    if not data:
        print(f"No usable training data found in {base_dir}.")
        return

    for env_name in KNOWN_ENVS:
        if env_name not in data:
            print(f"No data for {env_name}, skipping.")
            continue

        versions = dict(sorted(data[env_name].items()))
        n_versions = len(versions)

        fig, axes = plt.subplots(
            1,
            n_versions,
            figsize=(5 * n_versions, 5),
            sharey=True,
        )

        # Make axes always iterable even for a single subplot
        if n_versions == 1:
            axes = [axes]

        for ax, (version_label, (timesteps, ep_rew_mean)) in zip(axes, versions.items()):
            ax.plot(timesteps, ep_rew_mean, linewidth=1.5)
            ax.set_title(version_label, fontsize=9)
            ax.set_xlabel("Total Timesteps")
            ax.grid(True, alpha=0.3)

        axes[0].set_ylabel("Mean Episode Reward (last 100 episodes)")
        fig.suptitle(f"{env_name} — Training Comparison (subplots)", fontsize=12)
        fig.tight_layout()

        output_path = base_dir / f"{env_name}_training_comparison_subplots.png"
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {output_path}")


def main():
    # In case that this script is run directly, generate the training comparison plot for a directory.
    plot_training_comparison("1.0")
    plot_training_comparison_subplots("1.0")

if __name__ == "__main__":
    main()

