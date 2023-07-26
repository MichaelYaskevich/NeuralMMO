from gym import ObservationWrapper
import torch


class TrainWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
    
    def observation(self, obs):
        terrain, camp, entity = obs["terrain"], obs["camp"], obs["entity"]
        #terrain (15, 15)
        #camp (15, 15)
        #entity (7, 15, 15)
        terrain = terrain.unsqueeze(0)
        camp = camp.unsqueeze(0)
        return torch.cat([terrain, camp, entity], dim=0)


# if __name__=='__main__':
#     tw = TrainWrapper(None)
#     obs = {"terrain": torch.rand(15, 15), 
#            "camp": torch.rand(15, 15), 
#            "entity": torch.rand(7, 15, 15)}
#     res = tw.observation(obs)
#     print(res.shape)
#     print('ok')
