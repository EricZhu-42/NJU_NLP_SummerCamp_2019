import argparse
import itertools
import os
import os.path as osp
import pickle
import random
import sys
import time
from collections import namedtuple

import gym
import gym.spaces
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
from gym import wrappers

import DQN_learn
import logz
from atari_wrappers import wrap_deepmind
from DQN_train import atari_model, get_env, get_session
from DQN_utils import (LinearSchedule, PiecewiseSchedule, ReplayBuffer,
                       get_wrapper_by_name,
                       initialize_interdependent_variables, minimize_and_clip)

OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs", "lr_schedule"])

def learn(env,
          q_func,
          optimizer_spec,
          session,
          exploration=LinearSchedule(1000000, 0.1),
          stopping_criterion=None,
          replay_buffer_size=1000000,
          batch_size=32,
          gamma=0.99,
          learning_starts=50000,
          learning_freq=4,
          frame_history_len=4,
          target_update_freq=10000,
          grad_norm_clipping=10):
    """
    Run Deep Q-learning algorithm.

    """
    assert type(env.observation_space) == gym.spaces.Box
    assert type(env.action_space)      == gym.spaces.Discrete

    model_initialized = False
    experiment_dir = os.path.abspath("./experiments/{}".format(env.spec.id))
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    monitor_path = os.path.join(experiment_dir, "monitor")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(monitor_path):
        os.makedirs(monitor_path)

    ###############
    # BUILD MODEL #
    ###############

    if len(env.observation_space.shape) == 1:
        # This means we are running on low-dimensional observations (e.g. RAM)
        input_shape = env.observation_space.shape
    else:
        img_h, img_w, img_c = env.observation_space.shape
        input_shape = (img_h, img_w, frame_history_len * img_c)
    num_actions = env.action_space.n

    # set up placeholders
    # placeholder for current observation (or state)
    obs_t_ph              = tf.placeholder(tf.uint8, [None] + list(input_shape))
    # placeholder for current action
    act_t_ph              = tf.placeholder(tf.int32,   [None])
    # placeholder for current reward
    rew_t_ph              = tf.placeholder(tf.float32, [None])
    # placeholder for next observation (or state)
    obs_tp1_ph            = tf.placeholder(tf.uint8, [None] + list(input_shape))
    # placeholder for end of episode mask
    # this value is 1 if the next state corresponds to the end of an episode,
    # in which case there is no Q-value at the next state; at the end of an
    # episode, only the current state reward contributes to the target, not the
    # next state Q-value (i.e. target is just rew_t_ph, not rew_t_ph + gamma * q_tp1)
    done_mask_ph          = tf.placeholder(tf.float32, [None])

    # casting to float on GPU ensures lower data transfer times.
    obs_t_float   = tf.cast(obs_t_ph,   tf.float32) / 255.0
    obs_tp1_float = tf.cast(obs_tp1_ph, tf.float32) / 255.0

    q = q_func(obs_t_float, num_actions, scope="q_func", reuse=False)
    target_q = q_func(obs_tp1_float, num_actions, scope="target_q_func", reuse=False)
    Q_samp = rew_t_ph + (1 - done_mask_ph) * gamma * tf.reduce_max(target_q, axis=1)
    Q_s = tf.reduce_sum(q * tf.one_hot(act_t_ph, num_actions), axis=1)
    total_error = tf.reduce_mean(tf.square(Q_samp - Q_s))
    q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q_func')
    target_q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_q_func')

    # construct optimization op (with gradient clipping)
    learning_rate = tf.placeholder(tf.float32, (), name="learning_rate")
    optimizer = optimizer_spec.constructor(learning_rate=learning_rate, **optimizer_spec.kwargs)
    train_fn = minimize_and_clip(optimizer, total_error,
                 var_list=q_func_vars, clip_val=grad_norm_clipping)

    # update_target_fn will be called periodically to copy Q network to target Q network
    update_target_fn = []
    for var, var_target in zip(sorted(q_func_vars,        key=lambda v: v.name),
                               sorted(target_q_func_vars, key=lambda v: v.name)):
        update_target_fn.append(var_target.assign(var))
    update_target_fn = tf.group(*update_target_fn)


    # construct the replay buffer
    replay_buffer = ReplayBuffer(replay_buffer_size, frame_history_len)
    print('Populating Replay Buffer')

    ###############
    # RUN ENV     #
    ###############

    num_param_updates = 0
    mean_episode_reward      = -float('nan')
    best_mean_episode_reward = -float('inf')
    last_obs = env.reset()
    LOG_EVERY_N_STEPS = 10000

    last_reward = 0
    loaded = False
    saver = tf.train.Saver()
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)

    for t in itertools.count():
        ### 1. Check stopping criterion
        if stopping_criterion is not None and stopping_criterion(env, t):
            break

        ### 2. Step the env and store the transition
        # replay memory stuff
        idx = replay_buffer.store_frame(last_obs)
        q_input = replay_buffer.encode_recent_observation()

        #if (np.random.random()<0.1):
         #   action = env.action_space.sample()
        #else:
            # chose action according to current Q and exploration
        if not model_initialized:
            action = env.action_space.sample()
        else:
            action_values = session.run(q, feed_dict={obs_t_ph: [q_input]})[0]
            action = np.argmax(action_values)
        # perform action in env
        try:
            new_state, reward, done, info = env.step(action)
        except:
            last_obs = env.reset()
        else:
            replay_buffer.store_effect(idx, action, reward, done)
            last_obs = new_state

        current_reward = 0
        if done:
            last_obs = env.reset()
            episode_rewards = get_wrapper_by_name(env, "Monitor").get_episode_rewards()
            try:
                current_reward = episode_rewards[-1]
            except:
                pass
            if last_reward!=current_reward:
                last_reward = current_reward
                print('Current episode reward:',current_reward)

        # at this point, the environment should have been advanced one step (and
        # reset if done was true), and last_obs should point to the new latest
        # observation

        ### 3. Perform experience replay and train the network.
        # note that this is only done if the replay buffer contains enough samples
        # for us to learn something useful -- until then, the model will not be
        # initialized and random actions should be taken
        if (t > learning_starts and
                t % learning_freq == 0 and
                replay_buffer.can_sample(batch_size)):

            s_batch, a_batch, r_batch, sp_batch, done_mask_batch = replay_buffer.sample(batch_size)

            # Load a previous checkpoint if we find one

            if not model_initialized:
                print('Populating Replay Buffer Succeed!')
                initialize_interdependent_variables(session, tf.global_variables(),
                                                    {obs_t_ph: s_batch, obs_tp1_ph: sp_batch, })
                model_initialized = True

            if not loaded and latest_checkpoint:
                print("Loading model checkpoint {}...\n".format(latest_checkpoint))
                saver.restore(session, latest_checkpoint)
                loaded = True

            env.render()
            feed_dict = {obs_t_ph:  s_batch,
                         act_t_ph: a_batch,
                         rew_t_ph: r_batch,
                         obs_tp1_ph: sp_batch,
                         done_mask_ph: done_mask_batch,
                         learning_rate: optimizer_spec.lr_schedule.value(t)}
            session.run(train_fn, feed_dict=feed_dict)
            num_param_updates += 1
            if num_param_updates % target_update_freq == 0:
                session.run(update_target_fn)
                num_param_updates = 0

            #####

        ### 4. Log progress
        episode_rewards = get_wrapper_by_name(env, "Monitor").get_episode_rewards()
        if len(episode_rewards) > 0:
            mean_episode_reward = np.mean(episode_rewards[-16:])
        if len(episode_rewards) > 16:
            best_mean_episode_reward = max(best_mean_episode_reward, mean_episode_reward)
        if t % LOG_EVERY_N_STEPS == 0 and model_initialized:
            logz.log_tabular('Timestep', t)
            logz.log_tabular('MeanReward', mean_episode_reward)
            logz.log_tabular('BestMeanReward', best_mean_episode_reward)
            logz.log_tabular('episodes', len(episode_rewards))
            logz.log_tabular('exploration', exploration.value(t))
            logz.log_tabular('learning_rate', optimizer_spec.lr_schedule.value(t))
            logz.dump_tabular()
            sys.stdout.flush()


def atari_learn(env,
                session,
                num_timesteps):
    # This is just a rough estimate
    num_iterations = float(num_timesteps) / 4.0

    lr_multiplier = 1.0
    lr_schedule = PiecewiseSchedule([
                                         (0,                   1e-4 * lr_multiplier),
                                         (num_iterations / 10, 1e-4 * lr_multiplier),
                                         (num_iterations / 2,  5e-5 * lr_multiplier),
                                    ],
                                    outside_value=5e-5 * lr_multiplier)
    optimizer = DQN_learn.OptimizerSpec(
        constructor=tf.train.AdamOptimizer,
        kwargs=dict(epsilon=1e-4),
        lr_schedule=lr_schedule
    )

    def stopping_criterion(env, t):
        # notice that here t is the number of steps of the wrapped env,
        # which is different from the number of steps in the underlying env
        return get_wrapper_by_name(env, "Monitor").get_total_steps() >= num_timesteps

    exploration_schedule = PiecewiseSchedule(
        [
            (0, 0.1),
            (1e6, 0.1),
            (num_iterations / 2, 0.01),
        ], outside_value=0.01
    )

    learn(
        env,
        q_func=atari_model,
        optimizer_spec=optimizer,
        session=session,
        exploration=exploration_schedule,
        stopping_criterion=stopping_criterion,
        replay_buffer_size=100000,
        batch_size=16,
        gamma=0.99,
        learning_starts=5000,
        learning_freq=4,
        frame_history_len=4,
        target_update_freq=10000,
        grad_norm_clipping=10
    )
    env.close()

def main():

    PROJECT_ROOT =osp.dirname(osp.realpath(__file__))
    logz.configure_output_dir(osp.join(PROJECT_ROOT, "log/"+"_RAM_"+time.strftime("%d-%m-%Y_%H-%M-%S")))
    seed = 0
    env = get_env('SpaceInvaders-v0', seed)
    session = get_session()
    atari_learn(env, session, num_timesteps=40000000)

if __name__ == "__main__":
    main()
