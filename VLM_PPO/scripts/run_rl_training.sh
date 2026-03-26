#!/bin/bash
# RL Training Launch Script for NumberLine
# Runs inside tmux session with rl4vlm_clean conda environment
# All output goes to terminal AND log files

SESSION_NAME="rl_numberline"
CONDA_ENV="rl4vlm_clean"
SCRIPT_DIR="/mnt/raid/rl_gaming/RL4VLM/VLM_PPO/scripts"

# Kill existing session if it exists
tmux kill-session -t "$SESSION_NAME" 2>/dev/null

# Create new tmux session and run training
tmux new-session -d -s "$SESSION_NAME" -x 220 -y 50

# Send commands to the tmux session
tmux send-keys -t "$SESSION_NAME" "source activate $CONDA_ENV && cd $SCRIPT_DIR && echo '=== RL Training Starting ===' && bash run_nl.sh 2>&1 | tee /mnt/raid/rl_gaming/RL4VLM/VLM_PPO/rl_logs/terminal_output.log; echo '=== Training finished with exit code: \$? ==='" Enter

echo ""
echo "=========================================="
echo "  RL Training launched in tmux!"
echo "=========================================="
echo ""
echo "  Session name: $SESSION_NAME"
echo "  Conda env:    $CONDA_ENV"
echo "  GPUs:         1, 3 (NVIDIA H200)"
echo ""
echo "  To attach:    tmux attach -t $SESSION_NAME"
echo "  To detach:    Ctrl+B, then D"
echo "  To kill:      tmux kill-session -t $SESSION_NAME"
echo ""
echo "  Log files will be in:"
echo "    /mnt/raid/rl_gaming/RL4VLM/VLM_PPO/rl_logs/numberline_run/"
echo "      ├── training.log          (full log)"
echo "      ├── config.json           (training args)"
echo "      ├── step_details.jsonl    (per-step CoT outputs)"
echo "      ├── iteration_summary.csv (per-iteration metrics)"
echo "      └── checkpoints/          (model checkpoints)"
echo ""
