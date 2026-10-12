from tensorflow.keras import backend
from tensorflow.keras import backend as K

import numpy as np
import math

def Iou_score(smooth = 1e-5, threhold = 0.5):
    def _Iou_score(y_true, y_pred):
        # score calculation
        y_pred = backend.greater(y_pred, threhold)
        y_pred = backend.cast(y_pred, backend.floatx())
        intersection = backend.sum(y_true[...,:-1] * y_pred, axis=[0,1,2])
        union = backend.sum(y_true[...,:-1] + y_pred, axis=[0,1,2]) - intersection

        score = (intersection + smooth) / (union + smooth)
        return score
    return _Iou_score

def f_score(beta=1, smooth = 1e-5, threhold = 0.5):
    def _f_score(y_true, y_pred):
        y_pred = backend.greater(y_pred, threhold)
        y_pred = backend.cast(y_pred, backend.floatx())

        tp = backend.sum(y_true[...,:-1] * y_pred, axis=[0,1,2])
        fp = backend.sum(y_pred         , axis=[0,1,2]) - tp
        fn = backend.sum(y_true[...,:-1], axis=[0,1,2]) - tp

        score = ((1 + beta ** 2) * tp + smooth) \
                / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
        return score
    return _f_score

def ConfusionMatrix(numClass, imgPredict, Label):
    #  返回混淆矩阵
    mask = (Label >= 0) & (Label < numClass)
    label = numClass * Label[mask] + imgPredict[mask]
    count = np.bincount(label, minlength=numClass ** 2)
    confusionMatrix = count.reshape(numClass, numClass)
    return confusionMatrix


def OverallAccuracy(confusionMatrix):
    #  返回所有类的整体像素精度OA
    # acc = (TP + TN) / (TP + TN + FP + TN)
    OA = np.diag(confusionMatrix).sum() / confusionMatrix.sum()
    return OA

def Kappa(confusionMatrix):
    #  返回所有类的整体像素精度OA
    # acc = (TP + TN) / (TP + TN + FP + TN)
    ##[[0 1]
    ##[2 3]]
    po= np.diag(confusionMatrix).sum() / confusionMatrix.sum()
    sum_row = confusionMatrix.sum(axis=0) ##[1 5]
    sum_col = confusionMatrix.sum(axis=1) ##[2 4]
    pe = (sum_row * sum_col).sum() / (confusionMatrix.sum()*confusionMatrix.sum())
    kappa = (po-pe)/(1-pe)
    return kappa


def Precision(confusionMatrix):
    #  返回所有类别的精确率precision
    precision = np.diag(confusionMatrix) / confusionMatrix.sum(axis=1)
    return precision


def Recall(confusionMatrix):
    #  返回所有类别的召回率recall
    recall = np.diag(confusionMatrix) / confusionMatrix.sum(axis=0)
    return recall


def F1Score(confusionMatrix):
    precision = np.diag(confusionMatrix) / confusionMatrix.sum(axis=1)
    recall = np.diag(confusionMatrix) / confusionMatrix.sum(axis=0)
    f1score = 2 * precision * recall / (precision + recall)
    return f1score


def IntersectionOverUnion(confusionMatrix):
    #  返回交并比IoU
    intersection = np.diag(confusionMatrix)
    union = np.sum(confusionMatrix, axis=1) + np.sum(confusionMatrix, axis=0) - np.diag(confusionMatrix)
    IoU = intersection / union
    return IoU

def MeanIntersectionOverUnion(confusionMatrix):
    #  返回平均交并比mIoU
    intersection = np.diag(confusionMatrix)
    union = np.sum(confusionMatrix, axis=1) + np.sum(confusionMatrix, axis=0) - np.diag(confusionMatrix)
    IoU = intersection / union
    mIoU = np.nanmean(IoU)
    return mIoU


def Frequency_Weighted_Intersection_over_Union(confusionMatrix):
    #  返回频权交并比FWIoU
    freq = np.sum(confusionMatrix, axis=1) / np.sum(confusionMatrix)
    iu = np.diag(confusionMatrix) / (
            np.sum(confusionMatrix, axis=1) +
            np.sum(confusionMatrix, axis=0) -
            np.diag(confusionMatrix))
    FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
    return FWIoU

def iou(y_true, y_pred):
    # IOU = TP / (TP + FN + FP)
    I = _true_positives(y_true, y_pred)
    U = I + _false_negatives(y_true, y_pred) + _false_positives(y_true, y_pred)
    return K.switch(K.equal(U, 0), 1.0, I / U)
    # if K.is_tensor(y_pred):
    #     return K.switch(K.equal(U, 0), 1.0, I / U)
    # else:
    #     return 1 if U is 0 else I / U

def _true_positives(y_true, y_pred):
    # determine number of matching values and return the sum
    return K.sum(y_true * K.round(y_pred))
    # if K.is_tensor(y_pred):
    #     return K.sum(y_true * K.round(y_pred))
    # else:
    #     return (y_true * np.round(y_pred)).sum()

def _true_negatives(y_true, y_pred):
    # invert and then treat as true positives
    y_true = (y_true - 1) * -1
    y_pred = (y_pred - 1) * -1

    return _true_positives(y_true, y_pred)


def _false_negatives(y_true, y_pred):
    return K.sum(K.clip((y_true - K.round(y_pred)), 0, 1))

    # if K.is_tensor(y_pred):
    #     return K.sum(K.clip((y_true - K.round(y_pred)), 0, 1))
    # else:
    #     return (y_true - np.round(y_pred)).clip(0).sum()


def _false_positives(y_true, y_pred):
    # switch positions and treat as false negatives
    return _false_negatives(y_pred, y_true)

