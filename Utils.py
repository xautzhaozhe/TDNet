# coding: utf-8

import tensorflow as tf
import numpy as np
import math
from scipy import linalg
from sklearn import metrics


def Mahalanobis(data):
    # 计算每个像素和均值向量之间的马氏距离 RX算法
    row, col, band = data.shape
    data = data.reshape(row * col, band)
    # 先求协方差矩阵
    mean_vector = np.mean(data, axis=0)
    mean_matrix = np.tile(mean_vector, (row * col, 1))
    re_matrix = data - mean_matrix
    matrix = np.dot(re_matrix.T, re_matrix) / (row * col - 1)
    # 在计算过程中有的矩阵是奇异阵，所有这里求的是伪逆矩阵
    variance_covariance = np.linalg.pinv(matrix)

    # 计算每个像素的马氏距离
    distances = np.zeros([row * col, 1])
    for i in range(row * col):
        re_array = re_matrix[i]
        re_var = np.dot(re_array, variance_covariance)
        distances[i] = np.dot(re_var, np.transpose(re_array))
    distances = distances.reshape(row, col)

    return distances


def ROC_AUC(target2d, groundtruth):
    """
    :param target2d: the 2D anomaly component
    :param groundtruth: the groundtruth
    :return: auc: the AUC value
    """
    rows, cols = groundtruth.shape
    label = groundtruth.transpose().reshape(1, rows * cols)
    target2d = target2d.transpose().reshape(1, rows * cols)
    result = np.zeros((1, rows * cols))
    for i in range(rows * cols):
        result[0, i] = np.linalg.norm(target2d[:, i])

    # result = hyper.hypernorm(result, "minmax")
    fpr, tpr, thresholds = metrics.roc_curve(label.transpose(), result.transpose())
    auc = metrics.auc(fpr, tpr)
    print('AUC value is: ', auc)

    return auc


def Residual(contr_data, org_data):
    # contr_data = (contr_data - np.min(contr_data)) / (np.max(contr_data) - np.min(contr_data))
    # org_data = (org_data - np.min(org_data)) / (np.max(org_data) - np.min(org_data))
    row, col, band = org_data.shape
    residual = np.square(org_data - contr_data)
    result = np.zeros((row, col))
    for i in range(row):
        for j in range(col):
            R = np.mean(residual[i, j, :])
            result[i, j] = R
    # result = (result - np.min(result)) / (np.max(result) - np.min(result))

    return result


def Preproce_data(data, rate=0.5):
    row, col, band = data.shape
    RX_result = Mahalanobis(data)
    vec_RX_result = RX_result.flatten()
    vec_RX_result.sort()
    # 设置选择背景样本的比率
    thred = vec_RX_result[int(col * row * rate - 1)]
    weight_mat = np.zeros((row, col), dtype=float)
    for i in range(row):
        for j in range(col):
            if RX_result[i, j] < thred:
                weight_mat[i, j] = 1
            else:
                weight_mat[i, j] = 0

    weight_mat = np.tile(weight_mat, [band, 1, 1])
    weight_mat = weight_mat.transpose([1, 2, 0])
    re_data = np.multiply(data, weight_mat)

    return re_data


def Encode_CASSI(x, Mask):
    y = tf.multiply(x, Mask)  # multiply 对应元素相乘
    y = tf.reduce_sum(y, axis=3)  # 按照通道求和
    return y


def Init_CASSI(y, Mask, channel):
    y1 = tf.expand_dims(y, axis=3)
    y2 = tf.tile(y1, [1, 1, 1, channel])
    x0 = tf.multiply(y2, Mask)
    return x0


def hypernorm(data2d, flag):
    normdata = np.zeros(data2d.shape)
    if flag == "minmax":
        minval = np.min(data2d)
        maxval = np.max(data2d)
        normdata = data2d - minval
        if maxval == minval:
            normdata = np.zeros(data2d.shape)
        else:
            normdata /= maxval - minval
    elif flag == "L2_norm":
        for i in range(data2d.shape[1]):
            col_norm = linalg.norm(data2d[:, i])
            normdata[:, i] = data2d[:, i] / col_norm

    return normdata


def Cal_mse(im1, im2):
    return np.mean(np.square(im1 - im2), dtype=np.float64)


def Cal_PSNR(im_true, im_test):
    channel = im_true.shape[2]
    im_true = 255 * im_true
    im_test = 255 * im_test

    psnr_sum = 0
    for i in range(channel):
        band_true = np.squeeze(im_true[:, :, i])
        band_test = np.squeeze(im_test[:, :, i])
        err = Cal_mse(band_true, band_test)
        max_value = np.max(np.max(band_true))
        psnr_sum = psnr_sum + 10 * np.log10((max_value ** 2) / err)

    return psnr_sum / channel


def Fusion(patch_image, gt_image, patch_cassi, block_size, stride):
    height = gt_image.shape[0]
    width = gt_image.shape[1]
    channel = gt_image.shape[2]
    result_image = np.zeros(gt_image.shape)
    weight_image = np.zeros(gt_image.shape)
    cassi_image = np.zeros([height, width])

    len_cassi = 542

    count = 0
    for x in range(0, height - channel + 1 - block_size + 1, stride):
        for y in range(0, width - block_size + 1, stride):
            result_image[x:x + block_size, y:y + block_size, :] = \
                result_image[x:x + block_size, y:y + block_size, :] + patch_image[:, :, :, count]

            weight_image[x:x + block_size, y:y + block_size, :] = \
                weight_image[x:x + block_size, y:y + block_size, :] + 1

            cassi_image[x:x + block_size, y:y + block_size] = \
                cassi_image[x:x + block_size, y:y + block_size] + np.squeeze(patch_cassi[count, :, :])
            count = count + 1
    for ch in range(channel):
        result_image[:, :, ch] = np.roll(result_image[:, :, ch], shift=ch, axis=0)
        weight_image[:, :, ch] = np.roll(weight_image[:, :, ch], shift=ch, axis=0)
    moreRow = int((math.floor((len_cassi - block_size + stride - 1) / stride)) * stride + block_size - len_cassi)
    result_image = result_image[
                   int(30 + math.floor(moreRow / 2)):int(height - 30 - (moreRow - math.floor(moreRow / 2))),
                   8:width - 8, :]
    gt_image = gt_image[int(30 + math.floor(moreRow / 2)):int(height - 30 - (moreRow - math.floor(moreRow / 2))),
               8:width - 8, :]
    cassi_image = cassi_image[int(math.floor(moreRow / 2)):int(height - 30 - (moreRow - math.floor(moreRow / 2))),
                  8:width - 8]
    weight_cassi = weight_image[int(math.floor(moreRow / 2)):int(height - 30 - (moreRow - math.floor(moreRow / 2))),
                   8:width - 8, 0]
    weight_image = weight_image[
                   int(30 + math.floor(moreRow / 2)):int(height - 30 - (moreRow - math.floor(moreRow / 2))),
                   8:width - 8, :]

    result_image = result_image / weight_image
    cassi_image = cassi_image / weight_cassi
    return result_image, gt_image, cassi_image


# loss

def tf_ssim(Prediction, X_label, max_val=1.0):

    return 1 - tf.reduce_mean(tf.image.ssim(X_label, Prediction, max_val=max_val))


def ssim_l1_loss(gt, y_pred, max_val=1.0, weight=0.1):
    ssim_loss = 1 - tf.reduce_mean(tf.image.ssim(gt, y_pred, max_val=max_val))
    L1 = tf.reduce_mean(tf.abs(gt - y_pred))
    return ssim_loss + L1 * weight
