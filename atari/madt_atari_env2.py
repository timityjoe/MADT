from __future__ import division
import gym
import numpy as np
from collections import deque
from gym.spaces.box import Box
from cv2 import resize
import random

from jax import tree_util
from tqdm import tqdm

from loguru import logger
# logger.remove()
# logger.add(sys.stdout, level="INFO")
# logger.add(sys.stdout, level="SUCCESS")
# logger.add(sys.stdout, level="WARNING")


# You can add your own logic and any other collection code here.
def _batch_rollout2(rng, envs, policy_fn, num_steps=2500, log_interval=None):
  logger.info("_batch_rollout_atari()")

  
  """Roll out a batch of environments under a given policy function."""
  # observations are dictionaries. Merge into single dictionary with batched
  # observations.
  obs_list = [env.reset() for env in envs]
  num_batch = len(envs)
  obs = tree_util.tree_map(lambda *arr: np.stack(arr, axis=0), *obs_list)
  ret = np.zeros([num_batch, 8])
  done = np.zeros(num_batch, dtype=np.int32)
  rew_sum = np.zeros(num_batch, dtype=np.float32)
  frames = []
  for t in tqdm(range(num_steps)):

    # Collect observations
    frames.append(
        np.concatenate([o['observations'][-1, ...] for o in obs_list], axis=1))
    done_prev = done

    actions, rng = policy_fn(rng, obs)

    # Collect step results and stack as a batch.
    step_results = [env.step(act) for env, act in zip(envs, actions)]
    obs_list = [result[0] for result in step_results]
    obs = tree_util.tree_map(lambda *arr: np.stack(arr, axis=0), *obs_list)
    rew = np.stack([result[1] for result in step_results])
    done = np.stack([result[2] for result in step_results])
    # Advance state.
    done = np.logical_or(done, done_prev).astype(np.int32)
    rew = rew * (1 - done)
    rew_sum += rew

    if log_interval and t % log_interval == 0:
      print('step: %d done: %s reward: %s' % (t, done, rew_sum))

    # Don't continue if all environments are done.
    if np.all(done):
      logger.info("np.all(done)..!")
      break
  return rew_sum, frames, rng


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


def process_frame(frame, conf):
    frame = frame[conf["crop1"]:conf["crop2"] + 160, :160]
    frame = frame.mean(2)
    frame = frame.astype(np.float32)
    frame *= (1.0 / 255.0)
    frame = resize(frame, (80, conf["dimension2"]))
    frame = resize(frame, (80, 80))
    frame = np.reshape(frame, [1, 80, 80])
    return frame


class AtariRescale(gym.ObservationWrapper):
    def __init__(self, env, env_conf):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = Box(0.0, 1.0, [1, 80, 80], dtype=np.uint8)
        self.conf = env_conf

    def observation(self, observation):
        return process_frame(observation, self.conf), [observation, self.conf]


class NormalizedEnv(gym.ObservationWrapper):
    def __init__(self, env=None):
        gym.ObservationWrapper.__init__(self, env)
        self.state_mean = 0
        self.state_std = 0
        self.alpha = 0.9999
        self.num_steps = 0

    def observation(self, observation_raw):
        observation, raw = observation_raw
        self.num_steps += 1
        self.state_mean = self.state_mean * self.alpha + \
            observation.mean() * (1 - self.alpha)
        self.state_std = self.state_std * self.alpha + \
            observation.std() * (1 - self.alpha)

        unbiased_mean = self.state_mean / (1 - pow(self.alpha, self.num_steps))
        unbiased_std = self.state_std / (1 - pow(self.alpha, self.num_steps))

        return (observation - unbiased_mean) / (unbiased_std + 1e-8), raw


class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        logger.info("NoopResetEnv::reset")
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            # noops = self.unwrapped.np_random.randint(1, self.noop_max + 1)  #pylint: disable=E1101
            noops = self.unwrapped.np_random.integers(1, self.noop_max + 1)  #pylint: disable=E1101
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)

        logger.info(f"  len(obs):{len(obs)}")

        return obs

    def step(self, ac):
        return self.env.step(ac)


class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        """Take action on reset for environments that are fixed until firing."""
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        logger.info("FireResetEnv::reset")
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)


class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condtion for a few frames
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, self.was_real_done

    def reset(self, **kwargs):
        logger.info("EpisodicLifeEnv::reset")
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros(
            (2, ) + env.observation_space.shape, dtype=np.uint8)
        self._skip = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2: self._obs_buffer[0] = obs
            if i == self._skip - 1: self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        logger.info("MaxAndSkipEnv::reset")
        return self.env.reset(**kwargs)
