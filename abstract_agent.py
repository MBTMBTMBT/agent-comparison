from abc import ABC, abstractmethod
import torch


class AbstractAgent(ABC):
    @abstractmethod
    def act(self, state: torch.Tensor) -> tuple[int, torch.Tensor]:
        """
        Selects an action to take given the current state of the environment.

        :param state: The current state of the environment.
        :return: A tuple containing:
            - The action chosen by the agent as an integer.
            - The distribution of actions.
        """
        pass
