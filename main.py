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

def create_env():
    cfg = CompetitionConfig()
    cfg.NMAPS = 400
    return TrainWrapper(TeamBasedEnv(config=cfg))

def make_custom_multi_env_func(full_env_name, cfg=None, _env_config=None, render_mode = None):
    return create_env()

def make_custom_encoder(cfg, obs_space):
    """Factory function as required by the API."""
    return ResnetEncoder(cfg, obs_space)

def register_custom_components():
    register_env("neuralmmo", make_custom_multi_env_func)
    global_model_factory().register_encoder_factory(make_custom_encoder)



def main():
    """Script entry point."""
    register_custom_components()
    cfg = parse_custom_args()
    status = run_rl(cfg)
    return status


if __name__ == "__main__":
    sys.exit(main())
