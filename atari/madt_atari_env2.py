


def atari_env(env_id, env_conf, args):
    logger.info(f"env_id:{env_id}")
    env = gym.make(env_id)

    if 'NoFrameskip' in env_id:
        logger.info("NoFrameskip")
        assert 'NoFrameskip' in env.spec.id
        env._max_episode_steps = args.max_episode_length * args.skip_rate
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=args.skip_rate)
    else:
        logger.info("Frameskip max_episode_length:{args.max_episode_length}")
        env._max_episode_steps = args.max_episode_length
    env = EpisodicLifeEnv(env)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        logger.info("FIRE")
        env = FireResetEnv(env)
    env._max_episode_steps = args.max_episode_length
    env = AtariRescale(env, env_conf)
    env = NormalizedEnv(env)
    return env