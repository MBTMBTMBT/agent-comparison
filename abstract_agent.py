from abc import ABC, abstractmethod
import torch


class AbstractAgent(ABC):
    @abstractmethod
    def select_action(self, state: torch.Tensor, return_distribution: bool) -> tuple[int, torch.Tensor]:
        """
        Selects an action to take given the current state of the environment.

        :param return_distribution: Bool value to tell
        :param state: The current state of the environment.
        :return: A tuple containing:
            - The action chosen by the agent as an integer.
            - The distribution of actions.
        """
        pass

    @abstractmethod
    def update(self) -> None:
        pass
