""" This script generates extra plots for the experiments. These plots are based on the logs of different scripts"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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

    envs_present = [env for env in KNOWN_ENVS if env in data]
    n_envs = len(envs_present)

    fig, axes = plt.subplots(1, n_envs, figsize=(6 * n_envs, 5), sharey=True)
    if n_envs == 1:
        axes = [axes]

    for ax, env_name in zip(axes, envs_present):
        versions = data[env_name]
        for version_label, (timesteps, ep_rew_mean, _, _) in sorted(versions.items()):
            ax.plot(timesteps, ep_rew_mean, label=version_label, linewidth=1.5)
        ax.set_title(env_name)
        ax.set_xlabel("Total Timesteps")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)

    axes[0].set_ylabel("Mean Episode Reward (last 100 episodes)")
    fig.suptitle("Training Comparison", fontsize=13)
    fig.tight_layout()

    output_path = base_dir / "training_comparison.png"
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

        # Extract shield events from optional columns
        events: list[tuple[int, str | None]] = []
        initial_annot: str | None = None
        if "shield_cutoff_timestep" in df.columns:
            cutoff = df["shield_cutoff_timestep"].dropna()
            if not cutoff.empty:
                events.append((int(cutoff.iloc[0]), None))
        elif "shield/transition_timestep" in df.columns:
            trans_rows = df[df["shield/transition_timestep"].notna()]
            for _, row in trans_rows.iterrows():
                ts = int(row["shield/transition_timestep"])
                if "shield/delta" in df.columns:
                    annotation = f"δ={row['shield/delta']}"
                elif "shield/ignore_prob" in df.columns:
                    annotation = f"p={row['shield/ignore_prob']}"
                else:
                    annotation = None
                events.append((ts, annotation))

        # Extract initial / constant delta annotation
        if "shield/initial_delta" in df.columns:
            v = df["shield/initial_delta"].dropna()
            if not v.empty:
                initial_annot = f"δ={v.iloc[0]} (init.)"
        elif "shield/constant_delta" in df.columns:
            v = df["shield/constant_delta"].dropna()
            if not v.empty:
                initial_annot = f"δ={v.iloc[0]}"

        data.setdefault(env_name, {})[version_label] = (timesteps, ep_rew_mean, events, initial_annot)

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

    # Fallback initial/constant delta annotations for data that predates CSV logging
    CONST_DELTA_FALLBACK = {
        "instant_turn_off": "δ=0.5",
        "ignore_prob":      "δ=0.9 (const.)",
        "delta":            "δ=0.9 (init.)",
    }

    # Map version-label substrings to (linestyle, legend_label)
    VLINE_STYLE = {
        "instant_turn_off": ((0, (1, 0)), "Shield disabled"),
        "delta":            ((0, (4, 2)), "δ changed"),
        "ignore_prob":      ((0, (4, 2)), "Ignore prob. changed"),
    }

    envs_present = [env for env in KNOWN_ENVS if env in data]
    n_envs = len(envs_present)

    # Determine the superset of version labels (columns) in sorted order
    all_versions = sorted({v for env in envs_present for v in data[env]})
    n_versions = len(all_versions)

    fig, axes = plt.subplots(
        n_envs,
        n_versions,
        figsize=(5 * n_versions, 4 * n_envs),
        sharey="row",
        squeeze=False,
    )

    for row, env_name in enumerate(envs_present):
        versions = data[env_name]
        for col, version_label in enumerate(all_versions):
            ax = axes[row][col]
            ts_arr, rew_arr = None, None
            events = []
            initial_annot = None
            if version_label in versions:
                ts_arr, rew_arr, events, initial_annot = versions[version_label]
                ax.plot(ts_arr, rew_arr, linewidth=1.5)

            # Fallback initial/constant annotation if not logged in CSV
            if initial_annot is None:
                for key, fallback in CONST_DELTA_FALLBACK.items():
                    if key in version_label:
                        initial_annot = fallback
                        break

            # Show initial/constant delta in the upper-left corner
            if initial_annot is not None:
                ax.text(0.03, 0.96, initial_annot, transform=ax.transAxes,
                        color="red", fontsize=7, ha="left", va="top",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                                  edgecolor="lightgrey", alpha=0.8))

            # Determine vline style from version label
            linestyle: tuple = (0, (4, 2))
            legend_label = "Event"
            for key, (ls, lbl) in VLINE_STYLE.items():
                if key in version_label:
                    linestyle, legend_label = ls, lbl
                    break

            # Draw vertical event lines and collect legend handles
            legend_handles = []
            handle = None
            for ts, annotation in events:
                line = ax.axvline(x=ts, color="red", linestyle=linestyle,
                                  alpha=0.8, linewidth=1.2)
                if handle is None:
                    handle = line
                if annotation is not None:
                    # Place label in whichever half (upper/lower) has less data near this vline.
                    # Use a window of ±5 % of the x-range for robustness.
                    if ts_arr is not None and len(ts_arr) > 0:
                        y_min, y_max = rew_arr.min(), rew_arr.max()
                        y_range = y_max - y_min if y_max > y_min else 1.0
                        x_window = (ts_arr[-1] - ts_arr[0]) * 0.05
                        mask = np.abs(ts_arr - ts) <= x_window
                        window_vals = rew_arr[mask] if mask.any() else rew_arr[np.argmin(np.abs(ts_arr - ts)):np.argmin(np.abs(ts_arr - ts))+1]
                        data_y_norm = float((window_vals.mean() - y_min) / y_range)
                        text_y = 0.20 if data_y_norm > 0.5 else 0.75
                    else:
                        text_y = 0.35
                    ax.text(ts, text_y, f" {annotation}", transform=ax.get_xaxis_transform(),
                            color="red", fontsize=7, ha="left", va="center")
            if handle is not None:
                handle.set_label(legend_label)
                legend_handles.append(handle)

            if legend_handles:
                ax.legend(handles=legend_handles, fontsize=7, loc="lower right")

            ax.grid(True, alpha=0.3)
            ax.set_xlabel("Total Timesteps")
            if col == 0:
                ax.set_ylabel(f"{env_name}\nMean Ep. Reward")
            if row == 0:
                ax.set_title(version_label, fontsize=9)

    fig.suptitle("Training Comparison (subplots)", fontsize=13)
    fig.tight_layout()

    output_path = base_dir / "training_comparison_subplots.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    # In case that this script is run directly, generate the training comparison plot for a directory.
    plot_training_comparison("10_March_16:47")
    plot_training_comparison_subplots("10_March_16:47")

if __name__ == "__main__":
    main()

