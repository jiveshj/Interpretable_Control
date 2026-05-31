#!/bin/bash
# launch_probing.sh
# ─────────────────────────────────────────────────────────────────────────────
# Runs TransFuser's CARLA leaderboard evaluator with our ProbingAgent,
# collecting activations and ground-truth distances during naturalistic
# driving rollouts.
#
# Prereqs:
#   - CARLA running on PORT 2000 (or override CARLA_PORT below)
#   - tfuse conda env activated
#   - $WORK_DIR points to your transfuser checkout
#
# Usage:
#   bash launch_probing.sh
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Paths (override via env if your layout differs) ──────────────────────────
WORK_DIR="${WORK_DIR:-/ocean/projects/cis250201p/jjain2/transfuser}"
LEADERBOARD_ROOT="${LEADERBOARD_ROOT:-$WORK_DIR/leaderboard}"
SCENARIO_RUNNER_ROOT="${SCENARIO_RUNNER_ROOT:-$WORK_DIR/scenario_runner}"
TEAM_CODE_ROOT="${TEAM_CODE_ROOT:-$WORK_DIR/team_code_transfuser}"
CARLA_ROOT="${CARLA_ROOT:-$WORK_DIR/carla}"

# Location of the probing agent file (this directory)
PROBING_AGENT="${PROBING_AGENT:-$(pwd)/probing_agent.py}"

# Where to save collected activations
PROBING_OUTPUT="${PROBING_OUTPUT:-/ocean/projects/cis250201p/jjain2/data/latents_leaderboard.h5}"

# Model checkpoint folder (contains the .pth file)
MODEL_CKPT="${MODEL_CKPT:-$WORK_DIR/model_ckpt/models_2022/transfuser}"

# Route + scenario JSON files. Devtest is short; full routes are longer.
ROUTES="${ROUTES:-$LEADERBOARD_ROOT/data/longest6/longest6.xml}"
SCENARIOS="${SCENARIOS:-$LEADERBOARD_ROOT/data/longest6/eval_scenarios.json}"

CARLA_PORT="${CARLA_PORT:-2000}"
TM_PORT="${TM_PORT:-8000}"
CHECKPOINT_ENDPOINT="${CHECKPOINT_ENDPOINT:-$(pwd)/leaderboard_results.json}"

# ── Validate ─────────────────────────────────────────────────────────────────
for p in "$LEADERBOARD_ROOT" "$SCENARIO_RUNNER_ROOT" "$TEAM_CODE_ROOT" \
         "$CARLA_ROOT" "$PROBING_AGENT" "$MODEL_CKPT" "$ROUTES" "$SCENARIOS"; do
    if [ ! -e "$p" ]; then
        echo "MISSING: $p"
        exit 1
    fi
done

# ── PYTHONPATH (matches TransFuser's run.sh / leaderboard conventions) ───────
export CARLA_EGG="$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg"
export PYTHONPATH="$CARLA_EGG:$CARLA_ROOT/PythonAPI/carla:$LEADERBOARD_ROOT:$SCENARIO_RUNNER_ROOT:$TEAM_CODE_ROOT:${PYTHONPATH:-}"

# Tell our agent where to save
export PROBING_OUTPUT
export TEAM_CODE_ROOT

# ── Launch ───────────────────────────────────────────────────────────────────
echo "──────────────────────────────────────────────────────────"
echo "  Probing leaderboard run"
echo "──────────────────────────────────────────────────────────"
echo "  Agent       : $PROBING_AGENT"
echo "  Model ckpt  : $MODEL_CKPT"
echo "  Routes      : $ROUTES"
echo "  Scenarios   : $SCENARIOS"
echo "  CARLA port  : $CARLA_PORT"
echo "  TM port     : $TM_PORT"
echo "  Output      : $PROBING_OUTPUT"
echo "──────────────────────────────────────────────────────────"

python "$LEADERBOARD_ROOT/leaderboard/leaderboard_evaluator.py" \
    --agent="$PROBING_AGENT" \
    --agent-config="$MODEL_CKPT" \
    --routes="$ROUTES" \
    --scenarios="$SCENARIOS" \
    --port="$CARLA_PORT" \
    --trafficManagerPort="$TM_PORT" \
    --checkpoint="$CHECKPOINT_ENDPOINT" \
    --track=SENSORS \
    --debug=0

echo ""
echo "Run finished. Collected data → $PROBING_OUTPUT"
echo "Probe with:"
echo "  python per_bin_analysis.py --data $PROBING_OUTPUT"