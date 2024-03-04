import torch
from torch.utils.data import Dataset
import random


class DiscreteSample:
    def __init__(
            self,
            state: torch.Tensor,
            action: torch.Tensor,
            log_prob: torch.Tensor,
            reward: torch.Tensor,
            state_value: torch.Tensor,
            advantage: torch.Tensor,
            done: bool,
            priority: float = 0.0,
    ):
        self.state: torch.Tensor = state
        self.action: torch.Tensor = action
        self.log_prob: torch.Tensor = log_prob
        self.reward: torch.Tensor = reward
        self.state_value: torch.Tensor = state_value
        self.advantage: torch.Tensor = advantage
        self.done: bool = done
        self.priority: float = priority


def weighted_sample_selection(
    samples: list[DiscreteSample],
    n_samples: int,
    invert_priority: bool = False
) -> tuple[list[DiscreteSample], list[DiscreteSample]]:
    if invert_priority:
        # Calculate inverse weights: subtract each sample's priority from the max priority and take the absolute,
        # making lower numerical priorities have higher weight
        max_priority = max(sample.priority for sample in samples)
        weights = [abs(max_priority - sample.priority) + 1 for sample in samples]  # Add 1 to avoid weight of 0
    else:
        # Use the original priorities as weights
        weights = [sample.priority for sample in samples]

    selected_samples = []
    for _ in range(n_samples):
        # Perform weighted sampling without replacement
        selection = random.choices(samples, weights=weights, k=1)[0]
        selected_samples.append(selection)
        # Remove the selected sample and its weight from the lists
        index = samples.index(selection)
        samples.pop(index)
        weights.pop(index)
        if len(samples) == 0:
            break  # Break if there are no more samples to select

    # The remaining samples are those not selected
    remaining_samples = samples

    return selected_samples, remaining_samples


class DiscretePrioritizedReplayBuffer(Dataset):
    def __init__(
            self,
            output_capacity: int,
            total_capacity: int,
            random_rotate: bool = True,
    ):
        pass
