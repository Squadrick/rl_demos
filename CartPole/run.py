import tensorflow as tf
import gym
import numpy as np

env = gym.make("CartPole-v1").env

with tf.Session() as sess:
    saver = tf.train.import_meta_graph("./DeepDrive - MeanRew: 200.000000-163.meta")
    saver.restore(sess = sess, save_path="./DeepDrive - MeanRew: 200.000000-163")
    graph = tf.get_default_graph()

    ob = graph.get_tensor_by_name("pi/ob:0")
    action = graph.get_tensor_by_name("pi/cond/Merge:0")
    sto = graph.get_tensor_by_name("pi/Placeholder:0")

    c_rew = 0
    done = False
    observation = env.reset()

    while not done:
        observation = np.expand_dims(observation, axis = 0)
        ac = sess.run([action], feed_dict = {ob: observation, sto: False})
        observation, rew, done, info = env.step(ac[0][0])
        env.render()
        c_rew += rew

print(c_rew)
