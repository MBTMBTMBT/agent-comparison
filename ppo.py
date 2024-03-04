import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import torchvision.models as models
from abstract_agent import AbstractAgent
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from replay_buffer import DiscretePrioritizedReplayBuffer


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
        self.squeezenet = models.squeezenet1_0(pretrained=True)
        self.squeezenet.features[0] = nn.Conv2d(input_channels, 96, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.adapt_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, output_size)
        self.squeezenet.classifier = nn.Identity()

    def forward(self, x):
        x = self.squeezenet.features(x)
        x = self.adapt_pool(x)
        x = torch.flatten(x, 1)
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
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, batch_size, replay_size, eps_clip, has_continuous_action_space, action_std_init=0.6, device='cpu'):
        self.device = torch.device(device)

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_std = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.batch_size = batch_size
        self.replay_size = replay_size

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
            self.buffer.state_values.append(torch.squeeze(state_val, dim=1))

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
            discounted_reward = (reward + (self.gamma * discounted_reward)) / len(self.buffer.rewards)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        rewards = torch.where(torch.isnan(rewards), torch.zeros_like(rewards), rewards)

        # Convert list to tensor and ensure minimum size
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0), dim=1).detach().to(self.device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0), dim=1).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0), dim=1).detach().to(self.device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0), dim=1).detach().to(self.device)

        try:
            # Ensure all tensors are at least self.replay_size in the first dimension
            old_states = repeat_tensor_to_size(old_states, self.replay_size)
            old_actions = repeat_tensor_to_size(old_actions, self.replay_size)
            old_logprobs = repeat_tensor_to_size(old_logprobs, self.replay_size)
            old_state_values = repeat_tensor_to_size(old_state_values, self.replay_size)
            rewards = repeat_tensor_to_size(rewards, self.replay_size)
            advantages = rewards.detach() - old_state_values.detach()

            # Recalculate advantages if necessary
            advantages = repeat_tensor_to_size(advantages, self.replay_size)
        except IndexError:
            self.buffer.clear()
            return

        try:
            # Creating dataset and dataloader for mini-batch processing
            dataset = TensorDataset(old_states, old_actions, old_logprobs, advantages, rewards)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        except IndexError:
            self.buffer.clear()
            return

        # Optimize policy for K epochs
        for epoch in tqdm(range(self.K_epochs), desc='Epochs'):
            # Initialize variables to track progress within an epoch
            epoch_loss = 0.0
            epoch_surr1 = 0.0
            epoch_surr2 = 0.0
            epoch_entropy = 0.0
            batch_count = 0

            for old_states_batch, old_actions_batch, old_logprobs_batch, advantages_batch, rewards_batch in dataloader:
                # Evaluating old actions and values for the mini-batch
                logprobs, state_values, dist_entropy = self.policy.evaluate(old_states_batch, old_actions_batch)

                # Match state_values tensor dimensions with rewards tensor for the mini-batch
                state_values = torch.squeeze(state_values)

                # Finding the ratio (pi_theta / pi_theta__old) for the mini-batch
                ratios = torch.exp(logprobs - old_logprobs_batch.detach())

                # Finding Surrogate Loss for the mini-batch
                surr1 = ratios * advantages_batch
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages_batch

                # Final loss of clipped objective PPO for the mini-batch
                loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards_batch) - 0.01 * dist_entropy

                # Take gradient step
                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()

                # Update epoch tracking variables
                epoch_loss += loss.mean().item()
                epoch_surr1 += surr1.mean().item()
                epoch_surr2 += surr2.mean().item()
                epoch_entropy += dist_entropy.mean().item()
                batch_count += 1

            # Calculate averages for the epoch
            avg_epoch_loss = epoch_loss / batch_count
            avg_epoch_surr1 = epoch_surr1 / batch_count
            avg_epoch_surr2 = epoch_surr2 / batch_count
            avg_epoch_entropy = epoch_entropy / batch_count

            # Display detailed information for the epoch
            # tqdm.write(
            #     f'Epoch {epoch + 1}/{self.K_epochs} - Loss: {avg_epoch_loss:.4f}, Surr1: {avg_epoch_surr1:.4f}, Surr2: {avg_epoch_surr2:.4f}, Entropy: {avg_epoch_entropy:.4f}')

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

    def charge_replay_buffer(self, replay_buffer: DiscretePrioritizedReplayBuffer) -> int:
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = (reward + (self.gamma * discounted_reward)) / len(self.buffer.rewards)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        rewards = torch.where(torch.isnan(rewards), torch.zeros_like(rewards), rewards)

        # Convert list to tensor and ensure minimum size
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0), dim=1).detach().to(self.device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0), dim=1).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0), dim=1).detach().to(self.device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0), dim=1).detach().to(self.device)

        # Ensure all tensors are at least self.replay_size in the first dimension
        old_states = repeat_tensor_to_size(old_states, self.replay_size)
        old_actions = repeat_tensor_to_size(old_actions, self.replay_size)
        old_logprobs = repeat_tensor_to_size(old_logprobs, self.replay_size)
        old_state_values = repeat_tensor_to_size(old_state_values, self.replay_size)
        rewards = repeat_tensor_to_size(rewards, self.replay_size)
        advantages = rewards.detach() - old_state_values.detach()

        # Recalculate advantages if necessary
        advantages = repeat_tensor_to_size(advantages, self.replay_size)

        # Creating dataset and dataloader for mini-batch processing
        dataset = TensorDataset(old_states, old_actions, old_logprobs, rewards, advantages)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

        for old_states_batch, old_actions_batch, old_logprobs_batch, rewards_batch, advantages_batch in dataloader:
            old_state = torch.squeeze(old_states_batch, dim=0)
            old_action = torch.squeeze(old_actions_batch, dim=0)
            old_logprob = torch.squeeze(old_logprobs_batch, dim=0)
            advantage = torch.squeeze(advantages_batch, dim=0)
            reward = torch.squeeze(rewards_batch, dim=0)
            replay_buffer.add(old_state, old_action, old_logprob, reward, advantage, priority=float(reward))

        return replay_buffer.is_full()

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


# Function to repeat tensor until it reaches the desired size
def repeat_tensor_to_size(tensor, target_size, dim=0):
    repeat_times = (target_size + tensor.size(dim) - 1) // tensor.size(
        dim)  # Calculate how many times to repeat
    repeated_tensor = tensor.repeat(repeat_times, *[1] * (tensor.dim() - 1))  # Repeat tensor
    return repeated_tensor[:target_size]  # Trim excess


class PPOWithImageEncoder(PPO):
    def __init__(self, input_channels, img_output_size, action_dim, lr_encoder, lr_actor, lr_critic, gamma, K_epochs, batch_size, replay_size, eps_clip,
                 has_continuous_action_space, action_std_init=0.6, device='cpu'):
        super().__init__(img_output_size, action_dim, lr_actor, lr_critic, gamma, K_epochs, batch_size, replay_size, eps_clip,
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

