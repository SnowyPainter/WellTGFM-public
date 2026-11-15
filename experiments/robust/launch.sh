#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p experiments/robust/outputs/v2_tgn
if tmux has-session -t tfra-robust 2>/dev/null; then
  current=$(tmux display-message -p -t tfra-robust:run '#{pane_current_command}')
  if [[ "$current" == bash || "$current" == sh ]]; then
    tmux send-keys -t tfra-robust:run '/opt/conda/bin/python -u -m experiments.robust.preflight >> experiments/robust/outputs/v2_tgn/run.log 2>&1' Enter
    echo 'Resumed: tmux attach -t tfra-robust'
  else
    echo 'tfra-robust is running; inspect it with: tmux attach -t tfra-robust'
  fi
  exit 0
fi
mkdir -p experiments/robust/outputs/v2_tgn
tmux new-session -d -s tfra-robust -n run -c "$PWD"
tmux set-option -t tfra-robust remain-on-exit on
tmux send-keys -t tfra-robust:run '/opt/conda/bin/python -u -m experiments.robust.preflight >> experiments/robust/outputs/v2_tgn/run.log 2>&1' Enter
tmux new-window -t tfra-robust -n log -c "$PWD" 'tail -n 30 -F experiments/robust/outputs/v2_tgn/run.log'
echo 'Started: tmux attach -t tfra-robust'
