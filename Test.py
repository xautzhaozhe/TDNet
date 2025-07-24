# coding: utf-8
import tensorflow as tf
import scipy.io as sio
import numpy as np
from time import time
import matplotlib.pylab as plt
import os
import Model
from Utils import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

# Parameters
dataset = 'AVIRIS-3'

phase = 10  # Max phase number of HQS
rank = 15  # CP rank
N_hidden = 24
# load dataset
test_data = './Data/%s.mat' % dataset
model_dir = 'Model/%s/' % dataset
result_dir = 'Result/%s' % dataset

if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# Load test data, 输入数据转换成 [样本数，宽，高，通道数]
print("...............................")
print("Load test data...")
Test_data = sio.loadmat(test_data)['data']
gt = sio.loadmat(test_data)['map']
Test_data = np.expand_dims(Test_data, axis=0)
_, row, col, channel = Test_data.shape
print('Test_data shape: ', Test_data.shape)

X_input = tf.placeholder(tf.float32, [None, row, col, channel])
Prediction = Model.Interface(X_input, phase, rank, channel, N_hidden, reuse=False)

# Model
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# 加载模型并进行预测
saver = tf.train.Saver()
sess = tf.Session(config=config)
start = time()

module_file = tf.train.latest_checkpoint(checkpoint_dir="./%s" % model_dir)
saver.restore(sess, module_file)

# 预测输出
Prediction_value = sess.run(Prediction, feed_dict={X_input: Test_data})
end = time()
sess.close()
time_used = end - start
print('测试时间为: %d' % time_used)
print("Reconstruction Finished")
re_data = Prediction_value.reshape(row, col, channel)
rx_result = Mahalanobis(re_data)

# 与原始图像的残差
Test_data_org = sio.loadmat(test_data)['data']
residual = Residual(re_data, Test_data_org)
print('重构误差检测结果：')
ROC_AUC(residual, gt)
plt.imshow(residual)
plt.axis('off')

plt.gcf().set_size_inches(512 / row, 512 / col)
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
plt.margins(0, 0)

plt.savefig('result/%s/%s_result.png' % (dataset, dataset), dpi=600)
plt.show()







