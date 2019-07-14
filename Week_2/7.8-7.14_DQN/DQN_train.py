import argparse
import os.path as osp
import random
import time

import gym
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
from gym import wrappers

import DQN_learn
import logz
from atari_wrappers import wrap_deepmind
from DQN_utils import PiecewiseSchedule, get_wrapper_by_name


def atari_model(img_in, num_actions, scope, reuse=False):
    #Model Structure
    with tf.variable_scope(scope, reuse=reuse):
        out = img_in
        with tf.variable_scope("Conv_net"):
            out = layers.convolution2d(out, num_outputs=32, kernel_size=8, stride=4, activation_fn=tf.nn.relu)
            out = layers.convolution2d(out, num_outputs=64, kernel_size=4, stride=2, activation_fn=tf.nn.relu)
            out = layers.convolution2d(out, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu)
            out = layers.flatten(out)

        with tf.variable_scope("Value"):
            fc1 = layers.fully_connected(out,   num_outputs=512,  activation_fn=tf.nn.relu)
            Value = layers.fully_connected(fc1, num_outputs=1,    activation_fn=None)
        
        with tf.variable_scope('Advantage'):
            fc2 = layers.fully_connected(out,       num_outputs=512,         activation_fn=tf.nn.relu)
            Advantage = layers.fully_connected(fc2, num_outputs=num_actions, activation_fn=None)

        out = Value + (Advantage - tf.reduce_mean(Advantage, axis=1, keep_dims=True))
        return out

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
            (0, 1.0),
            (1e6, 0.1),
            (num_iterations / 2, 0.01),
        ], outside_value=0.01
    )

    DQN_learn.learn(
        env,
        q_func=atari_model,
        optimizer_spec=optimizer,
        session=session,
        exploration=exploration_schedule,
        stopping_criterion=stopping_criterion,
        replay_buffer_size=1000000,
        batch_size=32,
        gamma=0.99,
        learning_starts=50000,
        learning_freq=4,
        frame_history_len=4,
        target_update_freq=10000,
        grad_norm_clipping=10
    )
    env.close()


def set_global_seeds(i):
    try:
        import tensorflow as tf
    except ImportError:
        pass
    else:
        tf.set_random_seed(i) 
    np.random.seed(i)
    random.seed(i)

def get_session():
    tf.reset_default_graph()
    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=1,
        intra_op_parallelism_threads=1)
    #tf_config.gpu_options.per_process_gpu_memory_fraction = 0.7
    #tf_config.gpu_options.allow_growth=True
    session = tf.Session(config=tf_config)
    return session

def get_env(name,seed):
    env = gym.make(name)

    set_global_seeds(seed)
    env.seed(seed)

    expt_dir = './tmp/'
    env = wrappers.Monitor(env, osp.join(expt_dir, "gym"), force=True)
    env = wrap_deepmind(env)

    return env

def main():
    # Get Atari games.

    # Change the index to select a different game.
    PROJECT_ROOT =osp.dirname(osp.realpath(__file__))
    logz.configure_output_dir(osp.join(PROJECT_ROOT, "log/"+"_RAM_"+time.strftime("%d-%m-%Y_%H-%M-%S")))

    # Run training
    seed = 0 # Use a seed of zero (you may want to randomize the seed!)
    env = get_env('SpaceInvadersNoFrameskip-v4', seed)
    session = get_session()
    atari_learn(env, session, num_timesteps=40000000)

if __name__ == "__main__":
    main()
