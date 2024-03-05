import torch
import torch.nn as nn
import torchvision.models as models
import random
import numpy as np
from dqn_replay_buffer import DiscretePrioritizedReplayBuffer
from torch.utils.data import DataLoader
from tqdm import tqdm


class QModel(nn.Module):
    def __init__(self, action_size, input_size=4):
        super(QModel, self).__init__()
        self.action_size = action_size
        self.input_size = input_size
        self.fc1 = nn.Linear(self.input_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, self.action_size)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class FlexibleImageEncoder(nn.Module):
    def __init__(self, input_channels, output_size):
        super(FlexibleImageEncoder, self).__init__()
        self.squeezenet = models.squeezenet1_0(pretrained=False)
        self.squeezenet.features[0] = nn.Conv2d(input_channels, 96, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.adapt_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, output_size)
        self.squeezenet.classifier = nn.Identity()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.squeezenet.features(x)
        x = self.adapt_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.tanh(x)
        return x


class QAgentWIthImageEncoder:
    def __init__(
            self,
            input_channels: int,
            state_dim: int,
            action_dim: int,
            lr_q: float = 0.001,
            lr_encoder: float = 0.001,
            gamma: float = 0.99,
            epsilon: float = 0.2,
            device='cpu',
    ):
        self.action_dim = action_dim
        self.epsilon = epsilon
        self.gamma = gamma
        self.device = device
        self.encoder = FlexibleImageEncoder(input_channels, state_dim).to(device=torch.device(device))
        self.q_model = QModel(action_dim, state_dim).to(device=torch.device(device))
        self.target_model = QModel(action_dim, state_dim).to(device=torch.device(device))
        self.optimizer_encoder = torch.optim.Adam(self.encoder.parameters(), lr=lr_encoder)
        self.optimizer_q = torch.optim.Adam(self.q_model.parameters(), lr=lr_q)
        self.loss_fn = nn.MSELoss()
        self.loss = 0

    def select_action(self, state: torch.Tensor, training=False) -> int:
        if training:
            self.q_model.train()
            self.encoder.train()
        else:
            self.q_model.eval()
            self.encoder.eval()
        self.encoder(torch.FloatTensor(state).to(device=torch.device(self.device)))
        action = random.randrange(self.action_dim) if np.random.rand() < self.epsilon else \
            torch.argmax(self.q_model(self.encoder(torch.FloatTensor(state).to(device=torch.device(self.device))))).item()
        # a = self.q_model(torch.FloatTensor(state).to(device=torch.device(self.device)))
        self.q_model.eval()
        self.encoder.eval()
        return action

    def learn(
            self,
            replay_buffer: DiscretePrioritizedReplayBuffer,
            batch_size: int = 32,
            num_epochs: int = 20,
            num_workers: int = 4,
    ):
        self.q_model.train()
        self.encoder.train()
        self.target_model.eval()
        dataloader = DataLoader(replay_buffer, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        for epoch in range(num_epochs):
            for old_states_batch, old_actions_batch, next_state_batch, reward_batch, terminal_batch in tqdm(
                    dataloader, desc='Epoch %d/%d' % (epoch+1, num_epochs)):
                # y_batch = torch.FloatTensor().to(torch.device(self.device))
                # for i in range(len(terminal_batch)):
                #     if terminal_batch[i]:
                #         y_batch = torch.cat((y_batch, torch.FloatTensor([reward_batch[i]]).to(torch.device(self.device))), 0)
                #     else:
                #         # xxx = next_state_batch[i]
                #         next_state_q = torch.max(self.target_model(self.encoder(torch.unsqueeze(next_state_batch[i], dim=0).to(torch.device(self.device)))))
                #         y = torch.FloatTensor([reward_batch[i] + self.gamma * next_state_q]).to(torch.device(self.device))
                #         y_batch = torch.cat((y_batch, y), 0).to(torch.device(self.device))
                #
                # current_state_q = torch.max(self.q_model(self.encoder(old_states_batch.to(torch.device(self.device)))), dim=1)[0]
                #
                # self.loss = self.loss_fn(current_state_q, y_batch).mean()
                # self.optimizer_q.zero_grad()
                # self.optimizer_encoder.zero_grad()
                # self.loss.backward()
                # self.optimizer_q.step()
                # self.optimizer_encoder.step()

                # Step 1: Prepare the data
                device = torch.device(self.device)  # Assuming self.device is a string like 'cuda' or 'cpu'
                next_state_batch = next_state_batch.to(device)
                reward_batch = reward_batch.to(device)
                terminal_batch = terminal_batch.to(device)

                # Step 2: Compute Q-values for all next states
                next_state_q_values = self.target_model(self.encoder(next_state_batch)).max(dim=1)[0]

                # Step 3: Initialize y_batch with reward_batch
                y_batch = reward_batch.clone()

                # Step 4: Update y_batch for non-terminal states
                non_terminal_mask = ~terminal_batch.bool()
                y_batch[non_terminal_mask] += self.gamma * next_state_q_values[non_terminal_mask]

                # Processing current states
                current_state_q = self.q_model(self.encoder(old_states_batch.to(device))).max(dim=1)[0]

                # Compute and backpropagate loss
                loss = self.loss_fn(current_state_q, y_batch).mean()
                self.optimizer_q.zero_grad()
                self.optimizer_encoder.zero_grad()
                loss.backward()
                self.optimizer_q.step()
                self.optimizer_encoder.step()

        self.target_model.load_state_dict(self.q_model.state_dict())
        self.q_model.eval()
        self.encoder.eval()

    def save(self, checkpoint_path, counter=-1, performance=0.0):
        torch.save(
            {
                'counter': counter,
                'encoder': self.encoder.state_dict(),
                'model': self.q_model.state_dict(),
                'optimizer_encoder': self.optimizer_encoder.state_dict(),
                'optimizer': self.optimizer_q.state_dict(),
                'performance': performance,
            },
            checkpoint_path,
        )

    def load(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.q_model.load_state_dict(checkpoint['model'])
        self.optimizer_encoder.load_state_dict(checkpoint['optimizer_encoder'])
        self.optimizer_q.load_state_dict(checkpoint['optimizer'])
        return checkpoint['counter'], checkpoint['performance']
