import json

from ijcai2022nmmo import CompetitionConfig, TeamBasedEnv
from nmmo import Env
from pydantic import BaseModel

from torchbeast.core.environment import Environment
from utils import TrainWrapper


def create_env():
    cfg = CompetitionConfig()
    cfg.NMAPS = 400
    return TrainWrapper(TeamBasedEnv(config=cfg))


def main():
    env = create_env()
    obs = env.reset()
    agent = obs[0]
    print('\n')
    for key in agent:
        print(key, agent[key].shape)

if __name__ == '__main__':
    main()
