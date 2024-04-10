import torch
from stable_baselines3.common.vec_env import VecEnvWrapper


class FeatureWrapper(VecEnvWrapper):
    def __init__(self, venv, feature_extractor: torch.nn.Module, device=torch.device('cpu')):
        super(FeatureWrapper, self).__init__(venv)
        self.feature_extractor = feature_extractor.to(device)
        self.device = device

    def reset(self):
        obs = self.venv.reset()
        return self.process_obs(obs)

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        return self.process_obs(obs), rewards, dones, infos

    def process_obs(self, obs):
        obs = torch.tensor(obs, device=self.device).float()
        with torch.no_grad():
            obs = self.feature_extractor(obs).cpu().numpy()
        return obs
