import json

from ijcai2022nmmo import CompetitionConfig, TeamBasedEnv
from nmmo import Env
from pydantic import BaseModel

from torchbeast.core.environment import Environment
from utils import TrainWrapper
from model import ResnetEncoder
import sys
from sample_factory.algo.utils.context import global_model_factory
from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.envs.env_utils import RewardShapingInterface, TrainingInfoInterface, register_env
from sample_factory.train import run_rl
from sf_examples.train_custom_env_custom_model import make_custom_encoder, override_default_params
import json

from ijcai2022nmmo import CompetitionConfig, TeamBasedEnv
from nmmo import Env

from torchbeast.core.environment import Environment
from utils import TrainWrapper
from gym import ObservationWrapper, spaces
import torch
import numpy as np
from gym import Wrapper
import gym

class ObservationWrapper(Wrapper):

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return self.observation(obs)

    def step(self, action):
        observation, reward, terminated, info = self.env.step(action)
        return self.observation(observation), reward, terminated, info

    def observation(self, observation):
        """Returns a modified observation."""
        raise NotImplementedError

def add_extra_params_func(parser):
    """
    Specify any additional command line arguments for this family of custom environments.
    """
    p = parser
    p.add_argument("--custom_env_episode_len", default=10, type=int, help="Number of steps in the episode")

def parse_custom_args(argv=None, evaluation=False):
    parser, cfg = parse_sf_args(argv=argv, evaluation=evaluation)
    add_extra_params_func(parser)
    override_default_params(parser)
    # second parsing pass yields the final configuration
    cfg = parse_full_cfg(parser, argv)
    return cfg

class ListWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self._step = 0
        self.max_step = 1024
        self._team_index = 0
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(17, 15, 15), dtype=np.float32)
        self.entity_maxed = np.array([10, 100, 1024, 100, 100, 100, 3])[:, None, None]
        
    def _convert_observation(self, obs):
       #TODO: replace with expand dims if possible
       terrain, camp, entity = obs["terrain"], obs["camp"], obs["entity"]
       
       terrain = np.moveaxis(np.eye(6)[terrain], [0, 1, 2], [1, 2, 0])
       camp = np.moveaxis(np.eye(4)[camp], [0, 1, 2], [1, 2, 0])
       
       entity =  entity / self.entity_maxed

       return np.concatenate([terrain, camp, entity], axis=0)
    
    
    def step(self, actions):
        self._step += 1
        observations, rewards, dones, infos = self.env.step(actions)
        
        rewards = [rewards[key] for key in sorted(rewards.keys())]

        terminated = [dones[key] for key in sorted(dones.keys())]
        
        truncated = [self._step >= self.max_step for _ in dones]
        # todo get information from infos
        infos = [{'is_active': dones[key]} for key in range(len(dones))]
        return self.observation(observations), rewards, terminated, truncated, infos 

    def observation(self, observations):
        results = []
        for key in sorted(observations.keys()):
            results.append(self._convert_observation(observations[key]))
        return results


class TruncatedTerminatedWrapper(Wrapper):
    def step(self, actions):
        observations, rewards, terminated, truncated, infos = self.env.step(actions)
        return observations, rewards, terminated, truncated, infos

    def reset(self, *args, **kwargs):
        observations = self.env.reset()
        return observations, {}

class AutoResetWrapper(Wrapper):
    def step(self, action):
        observations, rewards, terminated, truncated, infos = self.env.step(action)
        if all(terminated) or all(truncated):
            observations, _ = self.env.reset()
        return observations, rewards, terminated, truncated, infos

class IsMultiAgentWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)

        self.is_multiagent = True
    
    def get_num_agents(self):
        return 8

    @property
    def num_agents(self):
        return self.get_num_agents()

class OurNeuralMMO(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, ):
        cfg = CompetitionConfig()
        cfg.NMAPS = 400
        self._env = TrainWrapper(TeamBasedEnv(config=cfg))
        self.action_space = self._env.action_space

    def step(self, actions):
        return self._env.step(actions)

    def reset(self):
        return self._env.reset()

    def close(self):
        self._env.close()

def create_env():
    env = OurNeuralMMO() 
    env = ListWrapper(env)
    env = TruncatedTerminatedWrapper(env)
    env = IsMultiAgentWrapper(env)
    env = AutoResetWrapper(env)
    return env

def make_custom_multi_env_func(full_env_name, cfg=None, _env_config=None, render_mode = None):
    return create_env()

def make_custom_encoder(cfg, obs_space):
    """Factory function as required by the API."""
    return ResnetEncoder(cfg, obs_space)

def register_custom_components():
    register_env("neuralmmo", make_custom_multi_env_func)
    global_model_factory().register_encoder_factory(make_custom_encoder)



def main():
    #env = make_custom_multi_env_func(1, 2, 3, 4)
    #q = env.reset()
    #print(len(q)), exit(0) 
    """Script entry point."""
    register_custom_components()
    cfg = parse_custom_args()
    status = run_rl(cfg)
    return status


if __name__ == "__main__":
    sys.exit(main())
