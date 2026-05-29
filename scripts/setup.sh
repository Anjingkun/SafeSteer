#!/bin/bash
# One-shot installer: bash scripts/set.sh
# Creates / updates the `distillation` conda env and patches TRL.
set -e

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_PROJ_ROOT="$(cd "${_SCRIPT_DIR}/.." && pwd)"

# ---------------- 模型下载 ----------------
# Llama-Guard-4-12B 每次都下 (评测必需);
# 训练用 base model 由 BASE_MODEL 选一个:
#   Llama-3-8B-Instruct | Llama-3.1-8B-Instruct | Llama-3.2-3B-Instruct | Qwen3-8B | Qwen3-4B
# 不需要时 BASE_MODEL=none 跳过。
#   bash scripts/set.sh                                # 默认下 Meta-Llama-3-8B-Instruct
#   BASE_MODEL=Qwen3-8B bash scripts/set.sh
#   BASE_MODEL=none bash scripts/set.sh                # 只下 Llama Guard
#   DOWNLOAD_DEST=/some/dir bash scripts/set.sh        # 换目标位置
DOWNLOAD_DEST="${DOWNLOAD_DEST:-/opt/tiger/entry}"
BASE_MODEL="${BASE_MODEL:-Llama-3-8B-Instruct}"
HDFS_USER_BASE="hdfs://harunava/home/byte_malia_gcp_aiic/user/lihao.612"

# 已知 base model 在 HDFS_USER_BASE 下的子目录名 (相对 HDFS_USER_BASE)
declare -A BASE_MODEL_DIRS=(
    [Llama-3-8B-Instruct]="Meta-Llama-3-8B-Instruct"
    [Llama-3.1-8B-Instruct]="Llama-3.1-8B-Instruct"
    [Llama-3.2-3B-Instruct]="Llama-3.2-3B-Instruct"
    [Qwen3-8B]="Qwen3-8B"
    [Qwen3-4B]="Qwen3-4B"
)

if [ "${BASE_MODEL}" = "none" ] || [ -z "${BASE_MODEL}" ]; then
    BASE_SRC=""
elif [ -n "${BASE_MODEL_DIRS[${BASE_MODEL}]+x}" ]; then
    BASE_SRC="${HDFS_USER_BASE}/${BASE_MODEL_DIRS[${BASE_MODEL}]}"
else
    echo "[set.sh] ERROR: unknown BASE_MODEL=${BASE_MODEL}." \
         "可选: ${!BASE_MODEL_DIRS[*]} | none"
    exit 1
fi

# 待下载列表: 总是带 Llama-Guard-4-12B; 按需追加 base model
DOWNLOAD_LIST=(
    "Llama-Guard-4-12B|${HDFS_USER_BASE}/Llama-Guard-4-12B"
)
[ -n "${BASE_SRC}" ] && DOWNLOAD_LIST+=("${BASE_MODEL}|${BASE_SRC}")

mkdir -p "${DOWNLOAD_DEST}"
for entry in "${DOWNLOAD_LIST[@]}"; do
    name="${entry%%|*}"
    src="${entry#*|}"
    dst="${DOWNLOAD_DEST}/${name}"
    if [ -d "${dst}" ] && [ -n "$(ls -A "${dst}" 2>/dev/null)" ]; then
        echo "[set.sh] ${name} already at ${dst}, skipping."
        continue
    fi
    echo "[set.sh] downloading ${name} → ${dst}"
    hdfs dfs -get "${src}" "${dst}"
    echo "[set.sh] ${name} done."
done
echo "[set.sh] installing requirements.txt..."
pip install -r "${_PROJ_ROOT}/requirements.txt" --user

echo "[set.sh] installing extras (jaxtyping)..."
pip install jaxtyping --user

echo "[set.sh] patching trl.trainer.utils.print_prompt_completions_sample..."
python - <<'PY'
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

echo "[set.sh] done."
