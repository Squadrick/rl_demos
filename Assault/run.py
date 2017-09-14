import tensorflow as tf
import gym
import numpy as np
from PIL import Image
from gym import spaces
from collections import deque

class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
      #  assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def _reset(self):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset()
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(1, self.noop_max + 1) #pylint: disable=E1101
       # assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(0)
            if done:
                obs = self.env.reset()
        return obs

class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        """Take action on reset for environments that are fixed until firing."""
        gym.Wrapper.__init__(self, env)
       # assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
       # assert len(env.unwrapped.get_action_meanings()) >= 3

    def _reset(self):
        self.env.reset()
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset()
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset()
        return obs

class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done  = True

    def _step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert somtimes we stay in lives == 0 condtion for a few frames
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def _reset(self):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset()
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
        self._obs_buffer = deque(maxlen=2)
        self._skip       = skip

    def _step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)

        return max_frame, total_reward, done, info

    def _reset(self):
        """Clear past frame buffer and init. to first obs. from inner env."""
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs

class ClipRewardEnv(gym.RewardWrapper):
    def _reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign."""
        return np.sign(reward)

class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env):
        """Warp frames to 84x84 as done in the Nature paper and later work."""
        gym.ObservationWrapper.__init__(self, env)
        self.res = 84
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.res, self.res, 1))

    def _observation(self, obs):
        frame = np.dot(obs.astype('float32'), np.array([0.299, 0.587, 0.114], 'float32'))
        frame = np.array(Image.fromarray(frame).resize((self.res, self.res),
            resample=Image.BILINEAR), dtype=np.uint8)
        return frame.reshape((self.res, self.res, 1))

class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Buffer observations and stack across channels (last axis)."""
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        assert shp[2] == 1  # can only stack 1-channel frames
        self.observation_space = spaces.Box(low=0, high=255, shape=(shp[0], shp[1], k))

    def _reset(self):
        """Clear buffer and re-fill by duplicating the first observation."""
        ob = self.env.reset()
        for _ in range(self.k): self.frames.append(ob)
        return self._observation()

    def _step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._observation(), reward, done, info

    def _observation(self):
        assert len(self.frames) == self.k
        return np.concatenate(self.frames, axis=2)

def wrap_deepmind(env, episode_life=True, clip_rewards=True):
    """Configure environment for DeepMind-style Atari.

    Note: this does not include frame stacking!"""
   # assert 'NoFrameskip' in env.spec.id  # required for DeepMind-style skip
    if episode_life:
        env = EpisodicLifeEnv(env)
    #env = NoopResetEnv(env, noop_max=30)
    #env = MaxAndSkipEnv(env, skip=4)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = WarpFrame(env)
    if clip_rewards:
        env = ClipRewardEnv(env)
    return env

env = gym.make("Assault-v0").env
env = wrap_deepmind(env)
env = FrameStack(env, 4)
human_agent_action = 0
human_wants_restart = False
human_sets_pause = False
flag = True

def get_right_action(key):
    if key == 1: return 6
    if key == 2: return 4
    if key == 3: return 2
    if key == 4: return 3
    return key

def key_press(key, mod):
    global human_agent_action, human_wants_restart, human_sets_pause, flag
    if key==0xff0d: human_wants_restart = True
    if key==32: human_sets_pause = not human_sets_pause
    a = int(key - ord('0'))
    a = get_right_action(a)
    if a == 0: flag = not flag
    if a<0 or a >= env.action_space.n: return
    human_agent_action = a

def key_release(key, mod):
    global human_agent_action
    a = int(key - ord('0'))
    a = get_right_action(a)
    if a<=0 or a >= env.action_space.n: return
    if human_agent_action == a:
        human_agent_action = 0

env.render()
env.unwrapped.viewer.window.on_key_press = key_press
env.unwrapped.viewer.window.on_key_release = key_release 

with tf.Session() as sess:
    saver = tf.train.import_meta_graph("./DeepDrive - MeanRew: 30.890000-3001.meta")
    saver.restore(sess = sess, save_path="./DeepDrive - MeanRew: 30.890000-3001")
    graph = tf.get_default_graph()

    ob = graph.get_tensor_by_name("pi/ob:0")
    actions = graph.get_tensor_by_name("pi/add_1:0")
    action = tf.argmax(input = actions, name = "action", axis = 1)
    c_rew = 0

    while True:
        observation = env.reset()
        done = False
        while not done:
            observation = np.expand_dims(observation, axis = 0)
            ac = sess.run([action], feed_dict = {ob : observation})
            a = human_agent_action 
            if flag:
                observation, rew, done, info = env.step(ac)
            else:
                observation, rew, done, info = env.step(a)
            env.render()

            if human_wants_restart: break
            while human_sets_pause:
                env.render()
                import time
                time.sleep(0.1)
