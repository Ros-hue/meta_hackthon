import gymnasium as gym
import numpy as np
from env import OpenEnvTaskRouter

def run_baseline_agent(env, episodes=10):
    """Phase 2: Standard agent - Judge will call this"""
    total_rewards = []
    
    for ep in range(episodes):
        obs, _ = env.reset()
        total_reward = 0
        
        while True:
            # Simple rule-based agent
            action = 0 if "news" in obs["task"] else 1
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                break
        
        total_rewards.append(total_reward)
    
    avg_reward = np.mean(total_rewards)
    print(f"Avg reward: {avg_reward:.2f}")
    return avg_reward

if __name__ == "__main__":
    env = OpenEnvTaskRouter()
    score = run_baseline_agent(env)