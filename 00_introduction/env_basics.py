import gymnasium as gym

#Creating an environment
env = gym.make("LunarLander-v3")

#Sample an Action
sample_action = env.action_space.sample()
print("Sample Action: ", sample_action)

#Sample an Observation
sample_obs = env.observation_space.sample()
print("Observation Sample: ", sample_obs)

