import time
import re
import sys
import os

def main():
    log_file = "sft_training.log"
    print(f"Monitoring logs at {log_file}...")
    
    total_steps = 3125  # Estimated: 50k / 16 batch size
    last_step = 0
    
    last_loss = "N/A"
    last_epoch = "0.0"

    print("Progress: [--------------------------------------------------] 0% | Loss: N/A | Epoch: 0.0", end="\r")

    while True:
        try:
            if not os.path.exists(log_file):
                time.sleep(2)
                continue
                
            with open(log_file, "r") as f:
                content = f.read()
                
            steps_matches = re.findall(r"'step':\s*(\d+)", content)
            loss_matches = re.findall(r"'loss':\s*([0-9.]+)", content)
            epoch_matches = re.findall(r"'epoch':\s*([0-9.]+)", content)
            
            if steps_matches:
                current_step = int(steps_matches[-1])
                if loss_matches:
                    last_loss = loss_matches[-1]
                if epoch_matches:
                    last_epoch = epoch_matches[-1]
                
                # We can dynamically adjust total_steps if it exceeds our estimate
                if current_step > total_steps:
                    total_steps = current_step
                
                percent = current_step / total_steps
                bar_len = int(50 * percent)
                bar = "=" * bar_len + "-" * (50 - bar_len)
                
                print(f"Progress: [{bar}] {int(percent*100)}% ({current_step}/{total_steps}) | Loss: {last_loss} | Epoch: {last_epoch}", end="\r")
                
            if "Training completed" in content or "TrainOutput" in content:
                print(f"\nTraining Completed successfully! Checkpoints saved at checkpoints/points24_sft_1epoch")
                break
                
        except Exception as e:
            pass
            
        time.sleep(5)

if __name__ == "__main__":
    main()
