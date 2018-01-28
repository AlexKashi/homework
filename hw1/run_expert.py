#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python run_expert.py experts/Humanoid-v1.pkl Humanoid-v1 --render \
            --num_rollouts 20

            run_expert.py experts/Humanoid-v1.pkl Humanoid-v1 --render --num_rollouts 20
            run_expert.py experts/Hopper-v1.pkl Hopper-v1 --render --num_rollouts 1

Author of this script and included expert policies: Jonathan Ho (hoj@openai.com)
"""

import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy

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
    args = parser.parse_args()

    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(args.expert_policy_file)
    print('loaded and built')
    with tf.Session():
        tf_util.initialize()

        import gym
        env = gym.make(args.envname)
        size = (376, 17)
        size = (17,6)
        #size = (11, 3)
        size = (376, 17)
        max_steps = args.max_timesteps or env.spec.timestep_limit
        x = tf.placeholder(tf.float32, shape=[None, size[0]])
        h = tf.placeholder(tf.float32, shape=[None, 500])
        h2 = tf.placeholder(tf.float32, shape=[None, 500])
        y_ = tf.placeholder(tf.float32, shape=[None, size[1]])


        W1 = tf.Variable(tf.truncated_normal([size[0],500], stddev=0.01))
        b1 = tf.Variable(tf.constant(0.01, shape=[500]))

        W3 = tf.Variable(tf.truncated_normal([500,500], stddev=0.01))
        b3 = tf.Variable(tf.constant(0.01, shape=[500]))

        W2 = tf.Variable(tf.truncated_normal([500,size[1]], stddev=0.01))
        b2 = tf.Variable(tf.constant(0.01, shape=[size[1]]))

        h = tf.nn.relu(tf.matmul(x, W1) + b1)
        h2 = tf.nn.relu(tf.matmul(h, W3) + b3)
        #y = tf.matmul(h,W2) + b2
        y = 5*tf.nn.tanh(tf.matmul(h2,W2) + b2)
        #y = tf.nn.relu_layer(h,W2,b2)
       # regularizer = tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2) + tf.nn.l2_loss(b1) + tf.nn.l2_loss(b2)
      #  y = tf.nn.relu_layer(h,W2,b2)#tf.matmul(h,W2) + b2
     #   cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits = y))
        cross_entropy = tf.losses.mean_squared_error(y_,y)
        #cross_entropy = tf.nn.l2_loss(y_ - y)
        beta = 0.001#.001
        regularizer = tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2) + tf.nn.l2_loss(b1) + tf.nn.l2_loss(b2) + tf.nn.l2_loss(W3) + tf.nn.l2_loss(b3)
        loss = tf.reduce_mean(cross_entropy + beta * regularizer)
        train_step = tf.train.AdamOptimizer().minimize(cross_entropy)
        returns = []
        observations = []
        actions = []
        train = True
        for i in range(args.num_rollouts):
         #   print(obs.shape)
            if not train:
                break
            print('iter', i)
            obs = env.reset()
         #   print(obs.shape)
            done = False
            totalr = 0.
            steps = 0
            while not done:
                action = policy_fn(obs[None,:])
              #  print(obs.shape)
                print("ACTION", action)


                observations.append(obs)
                actions.append(action[0])
              #  print(action.shape)
              #  print(obs.shape)
               # print(action.shape)
              #  print(action.shape)
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
        if train:
            np.save(xName, expert_data['observations'])
            np.save(yName, expert_data['actions'])
        print(expert_data['observations'].shape)
        print(expert_data['actions'].shape)
       # X = np.load(xName + ".npy")
       # Y = np.load(yName + ".npy")
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
        for i in range(20):
            sess.run(iterator.initializer)
            while True:
                try:
                 #   print("ALEX ALEX")
                    x1, y1 = sess.run(next_element)
                #    print(x1.shape)
               #     print("BOB BOB")
                    train_step.run(feed_dict={x:x1, y_:y1})
                except tf.errors.OutOfRangeError:
                   loss = sess.run(cross_entropy, feed_dict={x:expert_data['observations'], y_:expert_data['actions']})
                   print("Step #: "  + str(i), "Loss: " + str(loss))
                   break

        input("WAIT")
    #    action = sess.run(y,feed_dict={x:obs.reshape(1,size[0])})
        # for i in range(10000):
        #     train_step.run(feed_dict={x:expert_data['observations'], y_:expert_data['actions']})
        #     loss = sess.run(cross_entropy, feed_dict={x:expert_data['observations'], y_:expert_data['actions']})
        #     print("Step #: "  + str(i), "Loss: " + str(loss))
        
        # out = sess.run(y,feed_dict={x:expert_data['observations'][0].reshape(1,376)})
        # print(out)
        # assert False
        for i in range(args.num_rollouts):
            print('iter', i)
            obs = env.reset()
            done = False
            totalr = 0.
            observations = []
            actions = []
            steps = 0
            while not done:
                action = sess.run(y,feed_dict={x:obs.reshape(1,size[0])})
                expertAction = policy_fn(obs[None,:])
              #  print(obs.shape)

                print(action)
               # print(action)
                print(steps)
             #   action = policy_fn(obs[None,:])
                observations.append(obs)
                actions.append(expertAction[0])
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                print(done)
                if args.render:
                    env.render()
                if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                if steps >= max_steps:
                   break
            print(totalr)
            print(expert_data['observations'].shape, np.array(observations).shape)
            X = np.concatenate((expert_data["observations"], np.array(observations)))
            Y = np.concatenate((expert_data["actions"], np.array(actions)))
            expert_data = {'observations': X,
                           'actions':      Y}
            dataset = tf.contrib.data.Dataset.from_tensor_slices((expert_data["observations"], expert_data["actions"]))
            #   dataset = dataset.map(...)
            dataset = dataset.shuffle(buffer_size=10000)
            dataset = dataset.batch(100)
            iterator = dataset.make_initializable_iterator()

            next_element = iterator.get_next()
            for i in range(20):
                sess.run(iterator.initializer)
                while True:
                    try:
                 #       print("ALEX ALEX")
                        x1, y1 = sess.run(next_element)
                #    print(x1.shape)
               #     print("BOB BOB")
                        train_step.run(feed_dict={x:x1, y_:y1})
                    except tf.errors.OutOfRangeError:
                        loss = sess.run(cross_entropy, feed_dict={x:expert_data['observations'], y_:expert_data['actions']})
                        print("Step #: "  + str(i), "Loss: " + str(loss))
                        break
     #   print(expert_data)

if __name__ == '__main__':
    main()
