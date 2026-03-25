"""
Smoke test for all three training scripts.
Runs a single environment from each script with 10 timesteps to verify
that imports, environment construction, and the training loop still work
after the core/ refactoring.

Usage (from project root):
    uv run experiments/smoke_test.py
"""

import sys
from pathlib import Path

# Make project root importable
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

# Fake the --output_dir argument expected by each training module at import time
sys.argv = [sys.argv[0], "--output_dir", "smoke_test"]

TIMESTEPS = 10
TEST_ENV = "CrossingEnv"  # One env is enough to verify imports + wrappers

# ── 1. No-shield script ────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SMOKE TEST 1/3: unshielded")
print("="*60)

import experiments.train_multiple_envs_no_shield as no_shield

no_shield.TOTAL_TIMESTEPS = TIMESTEPS
no_shield.NUM_ENVS = 1
no_shield.RECORDING_TIMESTEPS = []  # Skip video recording
no_shield.train_environment(TEST_ENV)

print("✓ Unshielded: OK")

# ── 2. Shielded - instant turn-off ────────────────────────────────────────────
print("\n" + "="*60)
print("SMOKE TEST 2/3: shielded instant turn-off")
print("="*60)

import experiments.train_multiple_envs_with_shield_instant_turn_off as instant

instant.TOTAL_TIMESTEPS = TIMESTEPS
instant.NUM_ENVS = 1
instant.RECORDING_TIMESTEPS = []
instant.SHIELD_DISABLE_TIMESTEP = TIMESTEPS + 1  # Keep shield on throughout
instant.train_environment(TEST_ENV)

print("✓ Shielded (instant turn-off): OK")

# ── 3. Shielded - gradual reduction ───────────────────────────────────────────
print("\n" + "="*60)
print("SMOKE TEST 3/3: shielded gradual reduction")
print("="*60)

# Provide the extra --mechanism argument required by this module
sys.argv = [sys.argv[0], "--mechanism", "ignore_prob", "--output_dir", "smoke_test"]

import experiments.train_multiple_envs_shield_gradual_reduction as gradual

gradual.TOTAL_TIMESTEPS = TIMESTEPS
gradual.NUM_ENVS = 1
gradual.RECORDING_TIMESTEPS = []
gradual.train_environment(TEST_ENV)

print("✓ Shielded (gradual reduction): OK")

print("\n" + "="*60)
print("All smoke tests passed.")
print("="*60)
