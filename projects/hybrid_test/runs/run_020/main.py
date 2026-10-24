#import numpy as np
#import yaml
#from hybrid_agent import HybridAgent
#from .env_grid import GridEnvironment
def run_experiment(config_path="config.yaml"):
    cfg = yaml.safe_load(open(config_path))
    #env = GridEnvironment(**cfg["environment"])
    #agent = HybridAgent(nn_weight=cfg["agent"]["nn_weight"])

    total_reward = 0
    #state = env.reset()
    #for _ in range(cfg["agent"]["energy_budget"]):
        #action = agent.act(state)
        #state, reward, done = env.step(action)
        #total_reward += reward
        #if done:
        #    break
    #return total_reward

if __name__ == "__main__":
    result = run_experiment()
    print(f"Hybrid agent total reward: {result:.3f}")
