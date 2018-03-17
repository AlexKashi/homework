#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python run_expert.py experts/Humanoid-v1.pkl Humanoid-v1 --render \
            --num_rollouts 20

            python run_expert.py experts/Humanoid-v1.pkl Humanoid-v1 --render --num_rollouts 20
            python run_expert.py experts/Hopper-v1.pkl Hopper-v1 --render --num_rollouts 1

Author of this script and included expert policies: Jonathan Ho (hoj@openai.com)
"""

import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy
import matplotlib.pyplot as plt

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    parser.add_argument('--DAgger', action='store_true')
    parser.add_argument('--train', action='store_true')
    args = parser.parse_args()

    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(args.expert_policy_file)
    print('loaded and built')
    with tf.Session():
        tf_util.initialize()

        import gym
        env = gym.make(args.envname)
        env_size = (env.observation_space.shape[0], env.action_space.shape[0])
       # print(env_size)
        #assert False

        max_steps = args.max_timesteps or env.spec.timestep_limit

        n_hidden = 500
        #Define placeholders
        x = tf.placeholder(tf.float32, shape=[None, env_size[0]])
        h = tf.placeholder(tf.float32, shape=[None, n_hidden])
        h2 = tf.placeholder(tf.float32, shape=[None, n_hidden])
        y_ = tf.placeholder(tf.float32, shape=[None, env_size[1]])

        #Define Layer weights
        W1 = tf.Variable(tf.truncated_normal([env_size[0],n_hidden], stddev=0.01))
        b1 = tf.Variable(tf.constant(0.01, shape=[n_hidden]))

        W2 = tf.Variable(tf.truncated_normal([n_hidden,n_hidden], stddev=0.01))
        b2 = tf.Variable(tf.constant(0.01, shape=[n_hidden]))

        W3 = tf.Variable(tf.truncated_normal([n_hidden,env_size[1]], stddev=0.01))
        b3 = tf.Variable(tf.constant(0.01, shape=[env_size[1]]))

        #Setup Network
        h1 = tf.nn.relu(tf.matmul(x, W1) + b1)
        h2 = tf.nn.relu(tf.matmul(h1, W2) + b2)
        y = 5*tf.nn.tanh(tf.matmul(h2,W3) + b3)
        mean_squared_error = tf.losses.mean_squared_error(y_,y)

        beta = 0.001 #regularization rate
        regularizer = tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2) + tf.nn.l2_loss(b1) + tf.nn.l2_loss(b2) + tf.nn.l2_loss(W3) + tf.nn.l2_loss(b3)
        regularized_loss = tf.reduce_mean(mean_squared_error + beta * regularizer)
        train_step = tf.train.AdamOptimizer().minimize(regularized_loss)

        returns = []
        observations = []
        actions = []
        #train = False
        if args.train:
            for i in range(args.num_rollouts):
                print('iter', i)
                obs = env.reset()
                done = False
                totalr = 0.
                steps = 0
                while not done:
                    action = policy_fn(obs[None,:])
                    observations.append(obs)
                    actions.append(action[0])
                    obs, r, done, _ = env.step(action)
                    totalr += r
                    steps += 1
                    if args.render:
                        env.render()
                    if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                    if steps >= max_steps:
                        break
                returns.append(totalr)
        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())
        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))
        expert_data = {'observations': np.array(observations),
                       'actions': np.array(actions)}

      #  print(expert_data['observations'][0].shape)

        xName = "X_" + str(args.envname)
        yName = "Y_" + str(args.envname)
        print(xName)
        if args.train:
            np.save(xName, expert_data['observations'])
            np.save(yName, expert_data['actions'])
        print(expert_data['observations'].shape)
        print(expert_data['actions'].shape)
        X = np.load(xName + ".npy")
        Y = np.load(yName + ".npy")
        print(X.shape)
        print(Y.shape)
        expert_data = {'observations': X,
                       'actions': Y}
        dataset = tf.contrib.data.Dataset.from_tensor_slices((X, Y))
     #   dataset = dataset.map(...)
        dataset = dataset.batch(100)
        iterator = dataset.make_initializable_iterator()
        next_element = iterator.get_next()
        print(expert_data['observations'].shape)
       # assert False
        # # Compute for 100 epochs.
        losses = []
        for i in range(5):
            sess.run(iterator.initializer)
            while True:
                try:
                 #   print("ALEX ALEX")
                    x1, y1 = sess.run(next_element)
                #    print(x1.shape)
                  #  print("BOB BOB")
                    train_step.run(feed_dict={x:x1, y_:y1})
                except tf.errors.OutOfRangeError:
                   loss = sess.run(regularized_loss, feed_dict={x:expert_data['observations'], y_:expert_data['actions']})
                   print("Step #: "  + str(i), "Training Loss: " + str(loss))
                   losses.append(loss)
                   break
        plt.plot([i for i in range(len(losses))],losses)
        plt.title("Behavioral Cloning (BC) Training Loss " + args.envname)
        plt.xlabel("Epochs")
        plt.ylabel("Training Loss")
        #plt.show()

        def runEnv(expert_data, DAgger = False):
            for i in range(args.num_rollouts):
                print('iter', i)
                obs = env.reset()
                done = False
                totalr = 0.
                observations = []
                actions = []
                steps = 0
                while not done:
                    action = sess.run(y,feed_dict={x:obs.reshape(1,env_size[0])})
                    expertAction = policy_fn(obs[None,:])
                    observations.append(obs)
                    actions.append(expertAction[0])
                    obs, r, done, _ = env.step(action)
                    totalr += r
                    steps += 1
                    if args.render:
                        env.render()
                    if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                    if steps >= max_steps:
                       break

                print("Total Reward: " + str(totalr))
                if DAgger:
                    X = np.concatenate((expert_data["observations"], np.array(observations)))
                    Y = np.concatenate((expert_data["actions"], np.array(actions)))
                    expert_data = {'observations': X,
                                   'actions':      Y}
                    dataset = tf.contrib.data.Dataset.from_tensor_slices((expert_data["observations"], expert_data["actions"]))
                    dataset = dataset.shuffle(buffer_size=10000)
                    dataset = dataset.batch(100)
                    iterator = dataset.make_initializable_iterator()

                    next_element = iterator.get_next()
                    for i in range(20):
                        sess.run(iterator.initializer)
                        while True:
                            try:
                                x1, y1 = sess.run(next_element)
                                train_step.run(feed_dict={x:x1, y_:y1})
                            except tf.errors.OutOfRangeError:
                                loss = sess.run(regularized_loss, feed_dict={x:expert_data['observations'], y_:expert_data['actions']})
                                print("Step #: "  + str(i), "DAgger Training Loss: " + str(loss))
                                break
        runEnv(expert_data,args.DAgger)
     #   print(expert_data)

if __name__ == '__main__':
    main()
