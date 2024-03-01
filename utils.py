import os
import re


def find_latest_checkpoint(model_dir):
    """Find the latest model checkpoint in the given directory."""
    checkpoints = [f for f in os.listdir(model_dir) if f.startswith('model_epoch_') and f.endswith('.pth')]
    if not checkpoints:
        return None

    # Extracting the epoch number from the model filename using regex
    checkpoints.sort(key=lambda x: int(re.search(r'(\d+)', x).group()))
    return os.path.join(model_dir, checkpoints[-1])
