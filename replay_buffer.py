import torch
from torch.utils.data import Dataset
import random
from torchvision import transforms


class DiscreteSample:
    def __init__(
            self,
            state: torch.Tensor,
            action: torch.Tensor,
            log_prob: torch.Tensor,
            reward: torch.Tensor,
            advantage: torch.Tensor,
            priority: float = 1e-7,
    ):
        self.state: torch.Tensor = state.to(torch.device('cpu'))
        self.action: torch.Tensor = action.to(torch.device('cpu'))
        self.log_prob: torch.Tensor = log_prob.to(torch.device('cpu'))
        self.reward: torch.Tensor = reward.to(torch.device('cpu'))
        self.advantage: torch.Tensor = advantage.to(torch.device('cpu'))
        self.priority: float = abs(priority) + 1e-7
        self.used: bool = False

    def use(self) -> None:
        self.used = True

    def is_used(self) -> bool:
        return self.used


def weighted_sample_selection(
    samples: list[DiscreteSample],
    n_samples: int,
    invert_priority: bool = False,
    used_only: bool = False,
) -> tuple[list[DiscreteSample], list[DiscreteSample]]:
    if invert_priority:
        # Calculate inverse weights: subtract each sample's priority from the max priority and take the absolute,
        # making lower numerical priorities have higher weight
        max_priority = max(sample.priority for sample in samples)
        weights = [abs(max_priority - sample.priority) for sample in samples]
    else:
        # Use the original priorities as weights
        weights = [sample.priority for sample in samples]

    selected_samples = []
    counter = 0
    counter_ = 0
    while counter < n_samples:
        if counter_ >= n_samples:  # avoid dead loop
            break
        # Perform weighted sampling without replacement
        selection: DiscreteSample = random.choices(samples, weights=weights, k=1)[0]
        if used_only and not selection.is_used():
            counter_ += 1
            continue
        selected_samples.append(selection)
        # Remove the selected sample and its weight from the lists
        index = samples.index(selection)
        samples.pop(index)
        weights.pop(index)
        counter += 1
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
            image_size: tuple[int, int] = (192, 192)
    ):
        self.image_size = image_size
        self.output_capacity = output_capacity
        self.total_capacity = total_capacity
        self.random_rotate = random_rotate
        self.base_buffer: list[DiscreteSample] = []
        self.output_buffer: list[DiscreteSample] = []

    def add(
            self,
            state: torch.Tensor,
            action: torch.Tensor,
            log_prob: torch.Tensor,
            reward: torch.Tensor,
            advantage: torch.Tensor,
            priority: float = 0.0,
    ) -> int:
        sample = DiscreteSample(state, action, log_prob, reward, advantage, priority)
        self.base_buffer.append(sample)
        return len(self.base_buffer)

    def is_full(self) -> bool:
        return len(self.base_buffer) >= self.total_capacity

    def refresh_output_buffer(self) -> bool:
        selected, remained = weighted_sample_selection(self.base_buffer, self.output_capacity)
        self.output_buffer = selected
        self.base_buffer = selected + remained
        if len(selected) < self.output_capacity:
            print("Warning: Not enough samples to fill replay buffer.")
            return False
        return True

    def shrink_base_buffer(self) -> bool:
        selected, remained = weighted_sample_selection(
            self.base_buffer, len(self.base_buffer) - self.total_capacity, invert_priority=True, used_only=True,
        )
        self.base_buffer = remained
        if len(self.base_buffer) > self.total_capacity:
            print("Warning: Base buffer is still oversize.")
            return False
        return True

    def strict_shrink_base_buffer(self) -> None:
        selected, remained = weighted_sample_selection(
            self.base_buffer, len(self.base_buffer) - self.total_capacity, invert_priority=True, used_only=False,
        )
        self.base_buffer = remained

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self.output_buffer[index]
        sample.use()
        old_state = sample.state
        old_action = sample.action
        old_log_prob = sample.log_prob
        advantage = sample.advantage
        reward = sample.reward
        # Resize transformation
        resize_transform = transforms.Compose([
            transforms.ToPILImage(),  # Convert the tensor to a PIL image
            transforms.Resize(self.image_size),  # Resize the image
            transforms.ToTensor()  # Convert back to tensor
        ])
        old_state = resize_transform(old_state)
        if self.random_rotate:
            # Pick a random number of 90-degree rotations (0, 1, 2, or 3 times)
            k = random.choice([1, 2, 3])  # 0 is excluded since it would mean no rotation
            # Rotate the image by 90 degrees 'k' times
            old_state = torch.rot90(old_state, k, [1, 2])  # Rotates on the plane of the last two dimensions (H, W)
        return old_state, old_action, old_log_prob, reward, advantage

    def __len__(self) -> int:
        if self.output_capacity <= len(self.output_buffer):
            return self.output_capacity
        else:
            return len(self.output_buffer)
