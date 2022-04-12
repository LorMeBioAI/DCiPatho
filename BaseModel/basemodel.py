import torch
import torch.nn as nn
from DCiPatho_config import Config

config = Config()


class BaseModel(nn.Module):
    def __init__(self, config):
        super(BaseModel, self).__init__()
        self._config = config

    def saveModel(self):
        torch.save(self.state_dict(), config.model_name)

    def loadModel(self, map_location):
        state_dict = torch.load(config.model_name, map_location=map_location)
        self.load_state_dict(state_dict, strict=False)
