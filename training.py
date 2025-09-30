#Convenience factory
from avellaneda_stoikov_gym_env import MarketMakingEnv, make_env
env = make_env(seed=0)
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())



def make_env(cfg: EnvConfig | None = None, seed: int | None = None) -> MarketMakingEnv:
    cfg = cfg or EnvConfig()
    if seed is not None:
        cfg.seed = seed
    return MarketMakingEnv(cfg)


# --------------------------- Training Example (SB3) ----------------------------
if __name__ == "__main__":
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
        from stable_baselines3.common.monitor import Monitor
    except Exception as e:
        print("Stable-Baselines3 not available. Install with: pip install stable-baselines3")
        raise

    # Create and wrap env
    def _factory():
        env = make_env(seed=42)
        return Monitor(env)

    vec_env = DummyVecEnv([_factory])

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        n_steps=2048,
        batch_size=256,
        gae_lambda=0.95,
        gamma=0.999,          # long-ish horizon
        learning_rate=3e-4,
        clip_range=0.2,
        ent_coef=0.01,
        tensorboard_log=None,
        device="auto",
    )

    timesteps = 200_000
    model.learn(total_timesteps=timesteps)

    # Quick evaluation rollout
    env = make_env(seed=123)
    obs, info = env.reset()
    total_reward = 0.0
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    final_pnl = info["pnl"]
    print(f"Episode finished. Total reward: {total_reward:.2f}  Final PnL: {final_pnl:.2f}")