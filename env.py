import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Tuple
import numpy as np

class OpenEnvTaskRouter(gym.Env):
    def __init__(self):
        super().__init__()
        self.tools = ["web", "math", "file", "code", "llm"]
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Dict({
            "task": spaces.Text(max_length=100),
            "step": spaces.Box(low=0, high=100, shape=(1,), dtype=np.int32)
        })
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.max_steps = 20
        
        tasks = [
            ("Find latest AI news", 0),
            ("Solve 2x+3=7", 1), 
            ("Read config.json", 2),
            ("Run python code", 3),
            ("Explain quantum", 4)
        ]
        task_idx = np.random.randint(len(tasks))
        self.current_task, self.correct_action = tasks[task_idx]
        
        obs = {"task": self.current_task, "step": np.array([0])}
        info = {"correct_tool": self.correct_action}
        return obs, info
    
    def step(self, action):
        reward = 1.0 if action == self.correct_action else -0.2
        
        self.step_count += 1
        terminated = self.step_count >= self.max_steps
        truncated = False
        
        if not terminated:
            tasks = [("Find latest AI news", 0), ("Solve 2x+3=7", 1)]
            task_idx = np.random.randint(len(tasks))
            self.current_task, self.correct_action = tasks[task_idx]
        
        obs = {"task": self.current_task, "step": np.array([self.step_count])}
        info = {"reward": reward, "success": action == self.correct_action}
        
        return obs, reward, terminated, truncated, info
    
    def state(self):
        return {
            "step": self.step_count,
            "done": self.step_count >= self.max_steps,
            "env": "TaskRouter-v1"
        }
