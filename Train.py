# coding: utf-8
import tensorflow as tf
import scipy.io as sio
import numpy as np
import os
import re
import Model
import h5py
from Utils import *


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

dataset = 'AVIRIS-3'

# Parameters
phase = 10              # Max phase number of HQS
epoch_num = 1200        # Max epoch number in training
learn_rate = 0.0001     # Learning rate
rank = 15               # CP rank
N_hidden = 24

# Date path
train_data = './Data/%s.mat' % dataset
model_dir = 'Model/%s/' % dataset
output_file_name = 'Model/log/Log_%s.txt' % dataset
if not os.path.exists(model_dir):
    os.makedirs(model_dir)


# Load training data, 输入数据转换成 [样本数，宽，高，通道数]
print("...............................")
print("Load training data...")
Training_data_org = sio.loadmat(train_data)['data']
Training_data_label = np.expand_dims(Training_data_org, axis=0)
Training_data = Preproce_data(Training_data_org, rate=0.5)
Training_data = np.expand_dims(Training_data, axis=0)
_, row, col, channel = Training_data.shape
print('Training_dat shape: ', Training_data.shape)

# Define variables
global_steps = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(learning_rate=learn_rate, global_step=global_steps,
                                           decay_steps=400, decay_rate=0.95,
                                           staircase=True)

X_input = tf.placeholder(tf.float32, [None, row, col, channel])
X_label = tf.placeholder(tf.float32, [None, row, col, channel])

# Model
Prediction = Model.Interface(X_input, phase, rank, channel, N_hidden, reuse=False)
ssim_loss = 1 - tf.reduce_mean(tf.image.ssim(X_label, Prediction, max_val=1.0))
cost_all = tf.reduce_mean(tf.square(Prediction - X_label))+ 0.5*ssim_loss
optm_all = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost_all/(row*col), global_step=global_steps)
init = tf.global_variables_initializer()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
sess = tf.Session(config=config)
sess.run(init)

print("Phase number: %d" % phase)
print("Max epoch number: %d" % epoch_num)
print("CP rank: %s" % rank)
print("Dataset: %s" % dataset)
print("...............................\n")

#  Training
print('Initial epoch: %d' % 0)
print("Start Training...")

for epoch_i in range(0, epoch_num):
    batch_ys = Training_data
    feed_dict = {X_input: batch_ys, X_label: Training_data_label}
    sess.run(optm_all, feed_dict=feed_dict)
    output_data = "[Epoch: %03d] Loss: %.6f learning_rate: %.6f \n" % (
        epoch_i, sess.run(cost_all, feed_dict=feed_dict),
        sess.run(learning_rate, feed_dict=feed_dict))
    print(output_data)

    output_file = open(output_file_name, 'a')
    output_file.write(output_data)
    output_file.close()

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if (epoch_i + 1) % 100 == 0:
        saver.save(sess, './%s/model_%d.cpkt' % (model_dir, epoch_i), write_meta_graph=False)
sess.close()
print("Training Finished")







