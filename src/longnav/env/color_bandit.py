from longnav.env.env_base import DummyEnvActor
import numpy as np
import random
from typing import Dict, Any
class ColorBanditEnvActor(DummyEnvActor):
    def __init__(self):
        super().__init__()
        # mapping from int to color in RGB space
        self.colors = [
            [255,0,0], # red
            [0,255,0], # green
            [0,0,255], # blue
            [255,255,0], # yellow
            [255,0,255], # magenta
            [0,255,255] # cyan
        ]
        self.color_idx = random.randint(0,len(self.colors)-1) # randomly pick a color as the correct answer at the start of each episode
    
    def step(self, action: int, supplementary_logs: Dict[str, Any] = None):
        # generate random RGB data and state dict
        _,state = super().step(action, supplementary_logs)
        correct_action = self.color_idx % 4
        state['reward'] = 1.0 if action == correct_action else -0.01 # reward of 1 for correct action, small negative reward otherwise
        if action == correct_action and action == 0: # if the correct action is stop, end episode immediately
            state['reward']+=2.0
        state['done'] = state['done'] or action==0 # end episode
        self.color_idx = random.randint(0,len(self.colors)-1)
        rgb = self._color_to_image(self.color_idx)
        return rgb, state
    
    def _color_to_image(self,color_idx):
        '''
        convert a color index to a 256x256 RGB image of that color.
        '''
        color = self.colors[color_idx]
        img = np.ones((256,256,3),dtype=np.uint8)*np.array(color,dtype=np.uint8).reshape((1,1,3))
        return img
    
    def reset(self):
        print("resetting superclass")

        self.color_idx = random.randint(0,len(self.colors)-1) # randomly pick a color as the correct answer at the start of each episode
        _, state = super().reset()
        state['obs']['instr_or_goal'] = f"guess the action according to the color."
        rgb = self._color_to_image(self.color_idx)
        return rgb, state