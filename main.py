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
    # env = create_env()
    env = Environment(create_env())

    env_output = env.initial()

    # obs = env.reset()
    # print(obs)


if __name__ == '__main__':
    main()
