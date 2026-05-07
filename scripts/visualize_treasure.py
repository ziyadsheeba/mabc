import mo_gymnasium

if __name__ == "__main__":
    env = mo_gymnasium.make("deep-sea-treasure-v0", render_mode="human")
    obs, info = env.reset()
    for i in range(1000):
        env.render()
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(reward)
        env.render()
