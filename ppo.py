import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import torchvision.models as models
from abstract_agent import AbstractAgent
from torch.utils.data import TensorDataset, DataLoader


class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init):
        super(ActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)

        # actor
        if has_continuous_action_space :
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, action_dim),
                nn.Tanh()
            )
        else:
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, action_dim),
                nn.Softmax(dim=-1)
            )


        # critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def set_action_std(self, new_action_std):

        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def forward(self):
        raise NotImplementedError

    def act(self, state):

        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)

        return action.detach(), action_logprob.detach(), state_val.detach()

    def evaluate(self, state, action):

        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(device)
            dist = MultivariateNormal(action_mean, cov_mat)

            # for single action continuous environments
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)

        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy


class FlexibleImageEncoder(nn.Module):
    def __init__(self, input_channels, output_size):
        super(FlexibleImageEncoder, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # self.adapt_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.resnet.fc.in_features, output_size)
        self.resnet.fc = nn.Identity()

    def forward(self, x):
        x = self.resnet(x)
        # x = self.adapt_pool(x)
        # x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class ActorCriticWithImageEncoder(ActorCritic):
    def __init__(self, input_channels, img_output_size, action_dim, has_continuous_action_space, action_std_init,
                 device='cpu'):
        # Initialize parent ActorCritic with modified state_dim = img_output_size
        super(ActorCriticWithImageEncoder, self).__init__(img_output_size, action_dim, has_continuous_action_space,
                                                          action_std_init)
        self.encoder = FlexibleImageEncoder(input_channels, img_output_size)
        self.device = device

    def act(self, state_img):
        state = self.encoder(state_img).to(self.device)
        return super(ActorCriticWithImageEncoder, self).act(state)

    def evaluate(self, state_img, action):
        state = self.encoder(state_img).to(self.device)
        return super(ActorCriticWithImageEncoder, self).evaluate(state, action)


class PPO (AbstractAgent):
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, batch_size, eps_clip, has_continuous_action_space, action_std_init=0.6, device='cpu'):
        self.device = torch.device(device)

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_std = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.batch_size = batch_size

        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        self.policy_old = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def set_action_std(self, new_action_std):

        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)

        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print("--------------------------------------------------------------------------------------------")

        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if (self.action_std <= min_action_std):
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)

        else:
            print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")

        print("--------------------------------------------------------------------------------------------")

    def select_action(self, state, return_distribution=True):

        if self.has_continuous_action_space:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(self.device)
                action, action_logprob, state_val = self.policy_old.act(state)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)

            return action.detach().cpu().numpy().flatten()

        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(self.device)
                action, action_logprob, state_val = self.policy_old.act(state)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)

            if return_distribution:
                return action.item(), action_logprob

            return action.item()

    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(self.device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(self.device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(self.device)

        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach()

        try:
            # Creating dataset and dataloader for mini-batch processing
            dataset = TensorDataset(old_states, old_actions, old_logprobs, advantages, rewards)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        except IndexError:
            self.buffer.clear()
            return

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            for old_states_batch, old_actions_batch, old_logprobs_batch, advantages_batch, rewards_batch in dataloader:
                # Evaluating old actions and values for the mini-batch
                logprobs, state_values, dist_entropy = self.policy.evaluate(old_states_batch, old_actions_batch)

                # match state_values tensor dimensions with rewards tensor for the mini-batch
                state_values = torch.squeeze(state_values)

                # Finding the ratio (pi_theta / pi_theta__old) for the mini-batch
                ratios = torch.exp(logprobs - old_logprobs_batch.detach())

                # Finding Surrogate Loss for the mini-batch
                surr1 = ratios * advantages_batch
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages_batch

                # final loss of clipped objective PPO for the mini-batch
                loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards_batch) - 0.01 * dist_entropy

                # take gradient step
                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

    def save_model(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load_model(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))

    def save(self, checkpoint_path, counter=-1, performance=0.0):
        torch.save(
            {
                'counter': counter,
                'model': self.policy_old.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'performance': performance,
            },
            checkpoint_path,
        )

    def load(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.policy_old.load_state_dict(checkpoint['model'])
        self.policy.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        return checkpoint['counter'], checkpoint['performance']


class PPOWithImageEncoder(PPO):
    def __init__(self, input_channels, img_output_size, action_dim, lr_encoder, lr_actor, lr_critic, gamma, K_epochs, batch_size, eps_clip,
                 has_continuous_action_space, action_std_init=0.6, device='cpu'):
        super().__init__(img_output_size, action_dim, lr_actor, lr_critic, gamma, K_epochs, batch_size, eps_clip,
                         has_continuous_action_space, action_std_init, device)
        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_std = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.buffer = RolloutBuffer()

        # Initialize ActorCriticWithImageEncoder instead of ActorCritic
        self.policy = ActorCriticWithImageEncoder(input_channels, img_output_size, action_dim, has_continuous_action_space, action_std_init, device).to(device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.encoder.parameters(), 'lr': lr_encoder},
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        self.policy_old = ActorCriticWithImageEncoder(input_channels, img_output_size, action_dim, has_continuous_action_space, action_std_init, device).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

