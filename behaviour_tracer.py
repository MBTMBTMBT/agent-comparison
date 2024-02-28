import gymnasium as gym
import abstract_agent


class TimeStep:
    pass


class BehaviourTracer:
    def __init__(
            self,
            env: gym.Env,
            agent: abstract_agent.AbstractAgent,
            prior_agent: abstract_agent.AbstractAgent,
    ):
        self.env = env
        self.agent = agent
        self.prior_agent = prior_agent
        self.trajectory: list[TimeStep] = []
