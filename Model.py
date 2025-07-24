import tensorflow as tf
import numpy as np


def def_con2d_weight(w_shape, w_name):
    # Define the net weights
    weights = tf.get_variable(shape=w_shape, initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                              name='Weights_%s' % w_name)
    return weights


def RTGB(input, order_name, filter_num=64):
    # Rank-1 tensor generating block，与文章流程图基本对应 input->[B, W, H, C]
    gap_Height = tf.reduce_mean(tf.reduce_mean(input, axis=2, keepdims=True), axis=3, keepdims=True)
    gap_Weight = tf.reduce_mean(tf.reduce_mean(input, axis=1, keepdims=True), axis=3, keepdims=True)
    gap_Channel = tf.reduce_mean(tf.reduce_mean(input, axis=1, keepdims=True), axis=2, keepdims=True)

    weights_H = def_con2d_weight([1, 1, 1, 1], 'cp_con1d_hconv_%s' % order_name)
    weights_W = def_con2d_weight([1, 1, 1, 1], 'cp_con1d_wconv_%s' % order_name)
    weights_C = def_con2d_weight([1, 1, filter_num, filter_num], 'cp_con1d_cconv_%s' % order_name)

    convHeight_GAP = tf.nn.sigmoid(tf.nn.conv2d(gap_Height, weights_H, strides=[1, 1, 1, 1], padding='SAME'),
                                   name='sig_hgap_%s' % order_name)
    convWeight_GAP = tf.nn.sigmoid(tf.nn.conv2d(gap_Weight, weights_W, strides=[1, 1, 1, 1], padding='SAME'),
                                   name='sig_wgap_%s' % order_name)
    convChannel_GAP = tf.nn.sigmoid(tf.nn.conv2d(gap_Channel, weights_C, strides=[1, 1, 1, 1], padding='SAME'),
                                    name='sig_cgap_%s' % order_name)

    vecConHeight_GAP = tf.reshape(convHeight_GAP, [tf.shape(convHeight_GAP)[0], tf.shape(convHeight_GAP)[1], 1])
    vecConWeight_GAP = tf.reshape(convWeight_GAP, [tf.shape(convWeight_GAP)[0], 1, tf.shape(convWeight_GAP)[2]])
    vecConChannel_GAP = tf.reshape(convChannel_GAP, [tf.shape(convChannel_GAP)[0], 1, tf.shape(convChannel_GAP)[3]])

    matHWmulT = tf.matmul(vecConHeight_GAP, vecConWeight_GAP)
    vecHWmulT = tf.reshape(matHWmulT, [tf.shape(matHWmulT)[0], tf.shape(matHWmulT)[1] * tf.shape(matHWmulT)[2], 1])
    matHWCmulT = tf.matmul(vecHWmulT, vecConChannel_GAP)
    recon = tf.reshape(matHWCmulT, [tf.shape(input)[0], tf.shape(input)[1], tf.shape(input)[2], tf.shape(input)[3]])
    return recon


def ResBlock(input, order, filter_num):
    # Residual block for DRTLM
    xup = RTGB(input, 'up_order_%d' % order, filter_num)
    x_res = input - xup
    xdn = RTGB(x_res, 'down_order_%d' % order, filter_num)
    return xup, xdn


def DRTLM(input, rank, filter_num):
    # Discriminative rank-1 tensor learning module
    (xup, xdn) = ResBlock(input, 0, filter_num)
    temp_xup = xdn
    output = xup  # 对应流程图中的 O1
    for i in range(1, rank):
        (temp_xup, temp_xdn) = ResBlock(temp_xup, i, filter_num)
        xup = xup + temp_xup  # 先加起来，O1, O2=O1+O2, O3=O2+O3, ..., O(n)=O(n-1) + O(n)
        output = tf.concat([output, xup], 3)  # 拼接，将 O1,..., O(n) 拼接到一块
        temp_xup = temp_xdn
    return output


def Encoding(input, filter_size, filter_num):
    weights_pro_0 = def_con2d_weight([filter_size, filter_size, filter_num, filter_num], 'fproject_con2d_conv_0')
    input_temp = tf.nn.relu(tf.nn.conv2d(input, weights_pro_0, strides=[1, 1, 1, 1], padding='SAME'))

    weights_pro_1 = def_con2d_weight([filter_size, filter_size, filter_num, filter_num], 'fproject_con2d_conv_1')
    output = tf.nn.conv2d(input_temp, weights_pro_1, strides=[1, 1, 1, 1], padding='SAME')
    return output


def Fusion(input, xt, filter_size, filter_num, N_hidden):
    weights_attention = def_con2d_weight([filter_size, filter_size, filter_num, N_hidden],
                                         'IRecon_attention_con2d_conv')
    attention_map = tf.nn.conv2d(input, weights_attention, strides=[1, 1, 1, 1], padding='SAME')
    output = tf.multiply(xt, attention_map)
    return output


def Recon(x, channel=31, rank=4, N_hidden=24):
    # Parameters
    filter_size = 3
    filter_num = 64

    # 编码器  首先定义三层卷积，将原始输入 HSI 降维到指定维度
    weights_main_0 = def_con2d_weight([1, 1, channel, 128], 'main_con2d_conv_0')
    weights_main_1 = def_con2d_weight([1, 1, 128, 64], 'main_con2d_conv_1')
    weights_main_2 = def_con2d_weight([1, 1, 64, N_hidden], 'main_con2d_conv_2')

    x_feature_0 = tf.nn.leaky_relu(tf.nn.conv2d(x, weights_main_0, strides=[1, 1, 1, 1], padding='SAME'), alpha=0.2)
    x_feature_1 = tf.nn.leaky_relu(tf.nn.conv2d(x_feature_0, weights_main_1, strides=[1, 1, 1, 1], padding='SAME'), alpha=0.2)
    x_feature_2 = tf.nn.leaky_relu(tf.nn.conv2d(x_feature_1, weights_main_2, strides=[1, 1, 1, 1], padding='SAME'), alpha=0.2)

    # Low-rank Tensor Recovery；将降维后的 tensor进行 CP 分解; 其输出大小与 x_feature_2 相同
    attention_map_cat = DRTLM(x_feature_2, rank, N_hidden)  # 对应流程图中的 DRTLM
    # 将输入特征与 CP 分解得到的 Attention Map 进行点乘融合， 得到 Low_Rank Tensor
    x_feature_lowrank = Fusion(attention_map_cat, x_feature_2, filter_size, N_hidden * rank, N_hidden)

    # 解码器  将 Low_Rank Tensor 解码到输入大小
    DE_weights_0 = def_con2d_weight([1, 1, N_hidden, 64], 'main_decon2d_conv_0')
    DE_weights_1  = def_con2d_weight([1, 1, 64, 128], 'main_decon2d_conv_1')
    DE_weights_2  = def_con2d_weight([1, 1, 128, channel], 'main_decon2d_conv_2')

    DE_feature_0 = tf.nn.leaky_relu(tf.nn.conv2d(x_feature_lowrank, DE_weights_0, strides=[1, 1, 1, 1], padding='SAME'), alpha=0.2)
    DE_feature_0 = tf.add(DE_feature_0, x_feature_1)
    DE_feature_1 = tf.nn.leaky_relu(tf.nn.conv2d(DE_feature_0, DE_weights_1, strides=[1, 1, 1, 1], padding='SAME'), alpha=0.2)
    DE_feature_1 = tf.add(DE_feature_1, x_feature_0)
    DE_feature_2 = tf.nn.leaky_relu(tf.nn.conv2d(DE_feature_1, DE_weights_2, strides=[1, 1, 1, 1], padding='SAME'), alpha=0.2)

    return DE_feature_2


def Interface(x, phase, rank, channel, N_hidden, reuse):
    for i in range(phase):
        with tf.variable_scope('Phase_%d' % i, reuse=reuse):
            xt = Recon(x, channel, rank, N_hidden)

    return xt
