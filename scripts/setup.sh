#!/bin/bash
# =============================================================================
# SafeSteer environment installer.
#
#   bash scripts/setup.sh [env_name]
#
# env_name defaults to `safesteer`. If a conda env with that name already
# exists, the script aborts (it will NOT touch an existing env) -- pick a
# different name or remove the old one first.
#
# Installs all Python dependencies from requirements.txt and patches TRL so
# its `print_prompt_completions_sample` accepts the extra `completions_teacher`
# column SafeSteer logs. PyTorch is assumed to be already installed.
# =============================================================================
set -e

ENV_NAME="${1:-safesteer}"

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_PROJ_ROOT="$(cd "${_SCRIPT_DIR}/.." && pwd)"

# ---------------- 1. conda env ----------------
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "[setup] ERROR: conda env '${ENV_NAME}' already exists. Please check it," >&2
    echo "        remove it (conda env remove -n ${ENV_NAME}), or pick another name:" >&2
    echo "          bash scripts/setup.sh <env_name>" >&2
    exit 1
fi
echo "[setup] creating conda env '${ENV_NAME}' (python 3.12)..."
conda create -y -n "${ENV_NAME}" python=3.12

# Resolve the env's python without needing `conda activate` in a script.
ENV_PREFIX="$(conda env list | awk -v n="${ENV_NAME}" '$1==n {print $NF}')"
PY="${ENV_PREFIX}/bin/python"
echo "[setup] using interpreter: ${PY}"

# ---------------- 2. dependencies ----------------
echo "[setup] installing dependencies from requirements.txt ..."
${PY} -m pip install -r "${_PROJ_ROOT}/requirements.txt"

# ---------------- 3. patch TRL ----------------
# SafeSteer passes an extra `completions_teacher` column into TRL's
# print_prompt_completions_sample; stock TRL doesn't accept it and would crash
# on the first logging step. This adds the column in place (idempotent).
echo "[setup] patching trl.trainer.utils.print_prompt_completions_sample ..."
${PY} - <<'PY'
import inspect, sys
import trl.trainer.utils as u

src_path = inspect.getsourcefile(u)
with open(src_path, "r") as f:
    src = f.read()

if "completions_teacher" in src:
    print("  already patched, skipping.")
    sys.exit(0)

src = src.replace(
    "def print_prompt_completions_sample(\n"
    "    prompts: list,\n"
    "    completions: list,\n"
    "    rewards: dict[str, list[float]],\n"
    "    advantages: list[float],\n"
    "    step: int,\n"
    "    num_samples: int = None,\n"
    ") -> None:",
    "def print_prompt_completions_sample(\n"
    "    prompts: list,\n"
    "    completions: list,\n"
    "    completions_teacher: list,\n"
    "    rewards: dict[str, list[float]],\n"
    "    advantages: list[float],\n"
    "    step: int,\n"
    "    num_samples: int = None,\n"
    ") -> None:",
)

src = src.replace(
    '    table.add_column("Completion", style="bright_green")\n'
    '    for reward_name in rewards.keys():',
    '    table.add_column("Completion", style="bright_green")\n'
    '    table.add_column("Completion_teacher", style="bright_green")\n'
    '    for reward_name in rewards.keys():',
)

src = src.replace(
    "        completions = [completions[i] for i in indices]\n"
    "        rewards = {key: [val[i] for i in indices] for key, val in rewards.items()}",
    "        completions = [completions[i] for i in indices]\n"
    "        completions_teacher = [completions_teacher[i] for i in indices]\n"
    "        rewards = {key: [val[i] for i in indices] for key, val in rewards.items()}",
)

src = src.replace(
    "        table.add_row(\n"
    "            format_entry(prompts[i]),\n"
    "            format_entry(completions[i]),\n"
    "            *reward_values,",
    "        table.add_row(\n"
    "            format_entry(prompts[i]),\n"
    "            format_entry(completions[i]),\n"
    "            format_entry(completions_teacher[i]),\n"
    "            *reward_values,",
)

with open(src_path, "w") as f:
    f.write(src)
print(f"  patched {src_path}")
PY

echo ""
echo "[setup] done. Activate the env with:  conda activate ${ENV_NAME}"
