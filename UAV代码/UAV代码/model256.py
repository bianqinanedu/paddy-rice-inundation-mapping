import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from datetime import datetime
from tensorflow.keras.models import Model
from nets.mobilenet import mobilenetV2
from nets.Xception import Xception
from tensorflow.keras.layers import (Activation, BatchNormalization,MaxPooling2D,concatenate,
                                     Concatenate, Conv2D, DepthwiseConv2D,UpSampling2D,
                                     Dropout, GlobalAveragePooling2D, Input,Conv2D,
                                     Lambda, Softmax, ZeroPadding2D,Conv2DTranspose,Cropping2D,add,Reshape,Add,Multiply,
                                     LayerNormalization,Dense,AveragePooling2D)

from tensorflow.keras import layers, backend as K
from tensorflow.keras.initializers import RandomNormal, Constant
import numpy as np
epsilon = 1e-5
smooth = 1

# ==========================================
# 创新模块 2：Transformer 全局上下文瓶颈模块
# 位置：U-Net 底部瓶颈层
# 功能：利用 Transformer 捕捉长距离依赖，解决大目标断裂问题
# ==========================================
def edge_depthwise(x, kernel_3x3):
    """对每个通道用固定算子做Depthwise卷积, 不可训练"""
    def _init(shape, dtype=None):
        # shape = (3,3, in_channels, 1)
        k = np.array(kernel_3x3, dtype=np.float32).reshape(3,3,1,1)
        k = np.tile(k, (1, 1, shape[2], 1))
        return k
    return DepthwiseConv2D(3, padding='same', use_bias=False,
                           depthwise_initializer=_init,
                           trainable=False)(x)
def sobel_x(x):
    """ Sobel X方向边缘检测 """
    kernel = tf.constant([[-1, 0, 1],
                          [-2, 0, 2],
                          [-1, 0, 1]], dtype=tf.float32)
    # kernel = tf.reshape(kernel, [3, 3, 1, 1])
    # kernel = tf.tile(kernel, [1, 1, tf.shape(x)[-1], 1])  ##    3 3 NONe 1
    # x = tf.nn.conv2d(x, kernel, strides=1, padding='SAME')  ##  NONe  32 32  1
    return edge_depthwise(x, kernel)

def sobel_y(x):
    """ Sobel Y方向边缘检测 """
    kernel = tf.constant([[-1, -2, -1],
                          [0, 0, 0],
                          [1, 2, 1]], dtype=tf.float32)
    # kernel = tf.reshape(kernel, [3, 3, 1, 1])
    # kernel = tf.tile(kernel, [1, 1, tf.shape(x)[-1], 1])
    # x = tf.nn.conv2d(x, kernel, strides=1, padding='SAME')### None 32 32 1
    return edge_depthwise(x, kernel)

def laplacian(x):
    """ Laplacian边缘检测 """
    kernel = tf.constant([[1, 1, 1],
                          [1, -8, 1],
                          [1, 1, 1]], dtype=tf.float32)
    return edge_depthwise(x, kernel)

def diag_45(x, name="diag_45"):
    kernel = [[0, 1, 2],
              [-1, 0, 1],
              [-2, -1, 0]] # 45度对角
    return edge_depthwise(x, kernel)

def diag_135(x, name="diag_135"):
    kernel = [[2, 1, 0],
              [1, 0, -1],
              [0, -1, -2]] # 135度对角
    return edge_depthwise(x, kernel)

def edge_feature_extraction(inputs, filters, name="edge_extract"):
    """ 对应图中左侧的边缘特征提取模块 """
    # 三个分支：Sobel X, Sobel Y, Laplacian
    # 分支1: Sobel-X
    x1 = Conv2D(filters, 1, padding='same')(inputs)
    x1 = sobel_x(x1)  #### ## None 32 32 256

    # 分支2: Sobel-Y
    x2 = Conv2D(filters, 1, padding='same')(inputs)
    x2 = sobel_y(x2)## None 32 32 256

    # 分支3: Laplacian
    x3 = Conv2D(filters, 1, padding='same')(inputs)
    x3 = laplacian(x3)  ## None 32 32 256

    x4 = Conv2D(filters, 1, padding='same')(inputs)
    x4 = diag_45(x4)## None 32 32 256

    x5 = Conv2D(filters, 1, padding='same')(inputs)
    x5 = diag_135(x5)## None 32 32 256

    # 拼接三个分支
    concat = Concatenate()([x1, x2, x3, x4, x5])
    # 1x1卷积融合
    out = Conv2D(256, 1, padding='same', use_bias=False)(concat)  # ## None 32 32 256
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    return out# ## None 32 32 256

def spatial_attention_enhanced(inputs, name="spatial_attn"):
    """ 对应图中右侧的空间注意力增强模块 """
    # 并行AvgPool和MaxPool
    avg_pool = AveragePooling2D(pool_size=1, strides=1, padding='same', name=f"{name}_avgpool")(inputs)
    max_pool = MaxPooling2D(pool_size=1, strides=1, padding='same', name=f"{name}_maxpool")(inputs)

    # 拼接池化结果
    concat = Concatenate(name=f"{name}_pool_concat")([avg_pool, max_pool])

    # 7x7卷积 + Sigmoid生成注意力权重
    attn_map = Conv2D(1, 7, padding='same', activation='sigmoid', name=f"{name}_conv7x7")(concat)

    # 特征增强（乘法）
    enhanced = Multiply(name=f"{name}_multiply")([inputs, attn_map])
    return enhanced

def edge_attention_block(inputs, filters, name="EAB"):
    """ 完整的边缘注意力增强模块，对应你给的整个图 """
    # 1. 边缘特征提取
    edge_feat = edge_feature_extraction(inputs, filters)# None 16 16 1024   # ## None 32 32 256
    # 2. 空间注意力增强
    enhanced_edge = spatial_attention_enhanced(edge_feat)# None 16 16 1024
    # 3. 残差连接（可选，根据你的模型结构决定是否加）
    out = Add()([enhanced_edge, inputs])
    return out

def TransformerBottleneck(inputs, num_heads=4, projection_dim=64, name="TGB"):
    # 获取形状
    b, h, w, c = inputs.shape  #None 16 16 1024    NONE 32 32 256

    # 1) 展平成序列 (B, H, W, C) -> (B, H*W, C)
    # x = Reshape((h * w, c), name=f"{name}_reshape1")(inputs) #None 256 1024

    # 2) 第一残差段：LayerNorm + 多头自注意力
    x1 = LayerNormalization(epsilon=1e-6)(inputs)#None 16 16 1024    NONE 32 32 256

    attn_out = edge_attention_block(x1, filters=projection_dim)#None 16 16  1024    NONE 32 32 256

    x2 = Add()([attn_out, inputs])#None 16 16  1024    NONE 32 32 256

    # 3) 第二残差段：LayerNorm + MLP
    x3 = LayerNormalization(epsilon=1e-6)(x2)#None 16 16  1024

    # MLP 结构：channels * 4 → channels
    x3 = Dense(c * 4, activation="gelu")(x3)
    x3 = Dropout(0.1)(x3)
    x3 = Dense(c)(x3)  ## NONE 32 32 256

    x_out = Add()([x3, x2])    ## NONE 32 32 256

    # # 4) 重塑回图像形状
    # x_out = Reshape((h, w, c), name=f"{name}_reshape2")(x_out)

    return x_out

def DualAttentionGate(x_encoder, x_decoder, filters, name="CSAG"):
    """
    双注意力门（空间+通道）
    x_encoder: 编码器特征（跳跃连接）
    x_decoder: 解码器特征（门控信号）
    filters: 通道数
    """

    # ==================== 空间注意力 ====================
    g = Conv2D(filters, 1, padding='same')(x_decoder)#None 128 128 256--> None 128 128 256
    x = Conv2D(filters, 1, padding='same')(x_encoder)#None 128 128 48--> None 128 128 256

    sum_features = Add()([g, x])#None 128 128 256
    sum_features = Activation('relu')(sum_features)  ##None 128 128 256

    spatial_attention = Conv2D(1, 1, padding='same')(sum_features)#None 128 128 1
    spatial_attention = tf.keras.activations.sigmoid(spatial_attention)#None 128 128 1
    spatial_refined = Multiply()([sum_features, spatial_attention])#None 128 128 1--> None 128 128 128

    # ==================== 通道注意力 ====================
    y = GlobalAveragePooling2D()(spatial_refined)#None 128 128 256--> None 256
    y = Dense(filters // 4, activation='relu')(y)#None 256--> None 64
    y = Dense(filters, activation='sigmoid')(y)#None 64--> None 256

    # reshape + 广播
    channel_attention = tf.reshape(y, [-1, 1, 1, filters])#None 256 --> NONE 1 1 256
    channel_attention = tf.broadcast_to(channel_attention, tf.shape(spatial_refined))#NONE 1 1 256-->None 128 128 256

    # 应用通道注意力
    output = Multiply()([spatial_refined, channel_attention])#None 128 128 256-->None 128 128 256
    output = BatchNormalization()(output)#None 128 128 256-->None 128 128 256

    return output


def iou_score(y_true, y_pred):
    # 对sigmoid输出做二值化（阈值0.5）
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    # 计算交集和并集
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    # 避免除0，加小常数
    iou = (intersection + 1e-7) / (union + 1e-7)
    return iou

def layer1(x):
    x = Bottleneck(x, 64, downsampe=True)
    x = Bottleneck(x, 64)
    x = Bottleneck(x, 64)
    x = Bottleneck(x, 64)

    return x

def Bottleneck(x, size, downsampe=False):
    residual = x
    out = conv(x, size, 1, padding_='valid')
    out = BatchNormalization(epsilon=1e-5, momentum=0.1)(out)
    out = Activation('relu')(out)

    out = conv(out, size, 3)
    out = BatchNormalization(epsilon=1e-5, momentum=0.1)(out)
    out = Activation('relu')(out)

    out = conv(out, size * 4, 1, padding_='valid')
    out = BatchNormalization(epsilon=1e-5, momentum=0.1)(out)

    if downsampe:
        residual = conv(x, size * 4, 1, padding_='valid')
        residual = BatchNormalization(epsilon=1e-5, momentum=0.1)(residual)

    out = Add()([out, residual])
    out = Activation('relu')(out)

    return out

def transition_layer(x, in_channels, out_channels):
    num_in = len(in_channels)
    num_out = len(out_channels)
    out = []

    for i in range(num_out):
        if i < num_in:
            if in_channels[i] != out_channels[i]:
                residual = conv(x[i], out_channels[i], 3)
                residual = BatchNormalization(
                    epsilon=1e-5, momentum=0.1)(residual)
                residual = Activation('relu')(residual)
                out.append(residual)
            else:
                out.append(x[i])
        else:
            residual = conv(x[-1], out_channels[i], 3, strides_=2)
            residual = BatchNormalization(epsilon=1e-5, momentum=0.1)(residual)
            residual = Activation('relu')(residual)
            out.append(residual)

    return out

def BasicBlock(x, size, downsampe=False):
    residual = x

    out = conv(x, size, 3)
    out = BatchNormalization(epsilon=1e-5, momentum=0.1)(out)
    out = Activation('relu')(out)

    out = conv(out, size, 3)
    out = BatchNormalization(epsilon=1e-5, momentum=0.1)(out)

    if downsampe:
        residual = conv(x, size, 1, padding_='valid')
        residual = BatchNormalization(epsilon=1e-5, momentum=0.1)(residual)

    out = Add()([out, residual])
    out = Activation('relu')(out)

    return out

def branches(x, block_num, channels):
    out = []
    for i in range(len(channels)):
        residual = x[i]
        for j in range(block_num):
            residual = BasicBlock(residual, channels[i])
        out.append(residual)
    return out


def fuse_layers(x, channels, multi_scale_output=True):
    out = []

    for i in range(len(channels) if multi_scale_output else 1):
        residual = x[i]
        for j in range(len(channels)):
            if j > i:
                y = conv(x[j], channels[i], 1, padding_='valid')
                y = BatchNormalization(epsilon=1e-5, momentum=0.1)(y)
                y = UpSampling2D(size=2 ** (j - i))(y)
                residual = Add()([residual, y])
            elif j < i:
                y = x[j]
                for k in range(i - j):
                    if k == i - j - 1:
                        y = conv(y, channels[i], 3, strides_=2)
                        y = BatchNormalization(epsilon=1e-5, momentum=0.1)(y)
                    else:
                        y = conv(y, channels[j], 3, strides_=2)
                        y = BatchNormalization(epsilon=1e-5, momentum=0.1)(y)
                        y = Activation('relu')(y)
                residual = Add()([residual, y])

        residual = Activation('relu')(residual)
        out.append(residual)

    return out

def HighResolutionModule(x, channels, multi_scale_output=True):
    residual = branches(x, 4, channels)
    out = fuse_layers(residual, channels,
                      multi_scale_output=multi_scale_output)
    return out

def stage(x, num_modules, channels, multi_scale_output=True):
    out = x
    for i in range(num_modules):
        if i == num_modules - 1 and multi_scale_output == False:
            out = HighResolutionModule(out, channels, multi_scale_output=False)
        else:
            out = HighResolutionModule(out, channels)

    return out

def conv(x, outsize, kernel_size, strides_=1, padding_='same', activation=None):
    return Conv2D(outsize, kernel_size, strides=strides_, padding=padding_, kernel_initializer=RandomNormal(
        stddev=0.001), use_bias=False, activation=activation)(x)

def conv3x3(x, out_filters, strides=(1, 1)):
    x = Conv2D(out_filters, 3, padding='same', strides=strides, use_bias=False, kernel_initializer='he_normal')(x)
    return x

def basic_Block(input, out_filters, strides=(1, 1), with_conv_shortcut=False):
    x = conv3x3(input, out_filters, strides)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = conv3x3(x, out_filters)
    x = BatchNormalization(axis=3)(x)

    if with_conv_shortcut:
        residual = Conv2D(out_filters, 1, strides=strides, use_bias=False, kernel_initializer='he_normal')(input)
        residual = BatchNormalization(axis=3)(residual)
        x = add([x, residual])
    else:
        x = add([x, input])

    x = Activation('relu')(x)
    return x

def bottleneck_Block(input, out_filters, strides=(1, 1), with_conv_shortcut=False):
    expansion = 4
    de_filters = int(out_filters / expansion)

    x = Conv2D(de_filters, 1, use_bias=False, kernel_initializer='he_normal')(input)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = Conv2D(de_filters, 3, strides=strides, padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = Conv2D(out_filters, 1, use_bias=False, kernel_initializer='he_normal')(x)
    x = BatchNormalization(axis=3)(x)

    if with_conv_shortcut:
        residual = Conv2D(out_filters, 1, strides=strides, use_bias=False, kernel_initializer='he_normal')(input)
        residual = BatchNormalization(axis=3)(residual)
        x = add([x, residual])
    else:
        x = add([x, input])

    x = Activation('relu')(x)
    return x

def stem_net(input):
    x = Conv2D(16, 3, strides=(2, 2), padding='same', use_bias=False, kernel_initializer='he_normal')(input)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = bottleneck_Block(x, 64, with_conv_shortcut=True)
    x = bottleneck_Block(x, 64, with_conv_shortcut=False)
    x = bottleneck_Block(x, 64, with_conv_shortcut=False)
    x = bottleneck_Block(x, 64, with_conv_shortcut=False)

    return x

def transition_layer1(x, out_filters_list=[8, 16]):
    x0 = Conv2D(out_filters_list[0], 3, padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    x0 = BatchNormalization(axis=3)(x0)
    x0 = Activation('relu')(x0)

    x1 = Conv2D(out_filters_list[1], 3, strides=(2, 2),
                padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    x1 = BatchNormalization(axis=3)(x1)
    x1 = Activation('relu')(x1)

    return [x0, x1]


def make_branch1_0(x, out_filters=8):
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    return x


def make_branch1_1(x, out_filters=16):
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    return x


def fuse_layer1(x):
    x0_0 = x[0]
    x0_1 = Conv2D(8, 1, use_bias=False, kernel_initializer='he_normal')(x[1])
    x0_1 = BatchNormalization(axis=3)(x0_1)
    x0_1 = UpSampling2D(size=(2, 2))(x0_1)
    x0 = add([x0_0, x0_1])

    x1_0 = Conv2D(16, 3, strides=(2, 2), padding='same', use_bias=False, kernel_initializer='he_normal')(x[0])
    x1_0 = BatchNormalization(axis=3)(x1_0)
    x1_1 = x[1]
    x1 = add([x1_0, x1_1])
    return [x0, x1]


def transition_layer2(x, out_filters_list=[8, 16, 32]):
    x0 = Conv2D(out_filters_list[0], 3, padding='same', use_bias=False, kernel_initializer='he_normal')(x[0])
    x0 = BatchNormalization(axis=3)(x0)
    x0 = Activation('relu')(x0)

    x1 = Conv2D(out_filters_list[1], 3, padding='same', use_bias=False, kernel_initializer='he_normal')(x[1])
    x1 = BatchNormalization(axis=3)(x1)
    x1 = Activation('relu')(x1)

    x2 = Conv2D(out_filters_list[2], 3, strides=(2, 2),
                padding='same', use_bias=False, kernel_initializer='he_normal')(x[1])
    x2 = BatchNormalization(axis=3)(x2)
    x2 = Activation('relu')(x2)

    return [x0, x1, x2]


def make_branch2_0(x, out_filters=8):
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    return x


def make_branch2_1(x, out_filters=16):
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    return x


def make_branch2_2(x, out_filters=32):
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    return x


def fuse_layer2(x):
    x0_0 = x[0]
    x0_1 = Conv2D(8, 1, use_bias=False, kernel_initializer='he_normal')(x[1])
    x0_1 = BatchNormalization(axis=3)(x0_1)
    x0_1 = UpSampling2D(size=(2, 2))(x0_1)
    x0_2 = Conv2D(8, 1, use_bias=False, kernel_initializer='he_normal')(x[2])
    x0_2 = BatchNormalization(axis=3)(x0_2)
    x0_2 = UpSampling2D(size=(4, 4))(x0_2)
    x0 = add([x0_0, x0_1, x0_2])

    x1_0 = Conv2D(16, 3, strides=(2, 2), padding='same', use_bias=False, kernel_initializer='he_normal')(x[0])
    x1_0 = BatchNormalization(axis=3)(x1_0)
    x1_1 = x[1]
    x1_2 = Conv2D(16, 1, use_bias=False, kernel_initializer='he_normal')(x[2])
    x1_2 = BatchNormalization(axis=3)(x1_2)
    x1_2 = UpSampling2D(size=(2, 2))(x1_2)
    x1 = add([x1_0, x1_1, x1_2])

    x2_0 = Conv2D(8, 3, strides=(2, 2), padding='same', use_bias=False, kernel_initializer='he_normal')(x[0])
    x2_0 = BatchNormalization(axis=3)(x2_0)
    x2_0 = Activation('relu')(x2_0)
    x2_0 = Conv2D(32, 3, strides=(2, 2), padding='same', use_bias=False, kernel_initializer='he_normal')(x2_0)
    x2_0 = BatchNormalization(axis=3)(x2_0)
    x2_1 = Conv2D(32, 3, strides=(2, 2), padding='same', use_bias=False, kernel_initializer='he_normal')(x[1])
    x2_1 = BatchNormalization(axis=3)(x2_1)
    x2_2 = x[2]
    x2 = add([x2_0, x2_1, x2_2])
    return [x0, x1, x2]


def transition_layer3(x, out_filters_list=[8, 16, 32, 64]):
    x0 = Conv2D(out_filters_list[0], 3, padding='same', use_bias=False, kernel_initializer='he_normal')(x[0])
    x0 = BatchNormalization(axis=3)(x0)
    x0 = Activation('relu')(x0)

    x1 = Conv2D(out_filters_list[1], 3, padding='same', use_bias=False, kernel_initializer='he_normal')(x[1])
    x1 = BatchNormalization(axis=3)(x1)
    x1 = Activation('relu')(x1)

    x2 = Conv2D(out_filters_list[2], 3, padding='same', use_bias=False, kernel_initializer='he_normal')(x[2])
    x2 = BatchNormalization(axis=3)(x2)
    x2 = Activation('relu')(x2)

    x3 = Conv2D(out_filters_list[3], 3, strides=(2, 2),
                padding='same', use_bias=False, kernel_initializer='he_normal')(x[2])
    x3 = BatchNormalization(axis=3)(x3)
    x3 = Activation('relu')(x3)

    return [x0, x1, x2, x3]


def make_branch3_0(x, out_filters=8):
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    return x


def make_branch3_1(x, out_filters=16):
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    return x


def make_branch3_2(x, out_filters=32):
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    return x


def make_branch3_3(x, out_filters=64):
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    return x


def fuse_layer3(x):
    x0_0 = x[0]
    x0_1 = Conv2D(8, 1, use_bias=False, kernel_initializer='he_normal')(x[1])
    x0_1 = BatchNormalization(axis=3)(x0_1)
    x0_1 = UpSampling2D(size=(2, 2))(x0_1)
    x0_2 = Conv2D(8, 1, use_bias=False, kernel_initializer='he_normal')(x[2])
    x0_2 = BatchNormalization(axis=3)(x0_2)
    x0_2 = UpSampling2D(size=(4, 4))(x0_2)
    x0_3 = Conv2D(8, 1, use_bias=False, kernel_initializer='he_normal')(x[3])
    x0_3 = BatchNormalization(axis=3)(x0_3)
    x0_3 = UpSampling2D(size=(8, 8))(x0_3)
    x0 = concatenate([x0_0, x0_1, x0_2, x0_3], axis=-1)
    return x0


def final_layer(x, classes=1):
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(classes, 1, use_bias=False, kernel_initializer='he_normal')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('sigmoid', name='Classification')(x)
    return x


def final_layer2(x, classes=1):
    x = UpSampling2D(size=(4, 4))(x)
    x = Conv2D(classes, 1, use_bias=False, kernel_initializer='he_normal')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('sigmoid', name='Classification')(x)
    return x

class ONet(object):
    def __init__(self, img_rows=256, img_cols=256, weight_filepath=None, model_name = 'unet',LOSSMODE = 'cce'):
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channel = 3
        self.lossmode = LOSSMODE
        if model_name == 'unet':
            self.model = self.unet()
        elif model_name == 'deeplabv3+':
            self.model = self.deeplabv3()
        elif model_name == 'deeplabv3plus_improve':
            self.model = self.deeplabv3plus_improve()
        elif model_name == 'seg_hrnet':
            self.model = self.seg_hrnet()
        else:
            print('define model error!')
        self.current_epoch = 0
        self.weight_filepath = weight_filepath
        if weight_filepath !=None:
            self.model = tf.keras.models.load_model(weight_filepath)
        self.model_name = model_name
        print(self.model_name)

    def predict(self, sample):
        return self.model.predict(sample)

    def summary(self):
        """Get summary of the UNet model"""
        print(self.model.summary())
    def save(self):
        self.model.save_weights(self.current_weightfile())

    def load(self, filepath):
        # Create UNet-like model
        if self.model_name == 'unet':
            self.model = self.unet()
        elif self.model_name == 'deeplabv3+':
            self.model = self.deeplabv3()
        elif self.model_name == 'deeplabv3plus_improve':
            self.model = self.deeplabv3plus_improve()
        elif self.model_name == 'fcn_8s':
            self.model = self.fcn_8s()
        elif self.model_name == 'seg_hrnet':
            self.model = self.seg_hrnet()
        else:
            print('load load_weights error!')
        self.model.load_weights(filepath)

    def current_weightfile(self):
        assert self.weight_filepath != None, 'Must specify location of logs'
        return self.weight_filepath + "{}_weights_{}.h5".format(self.current_epoch, self.current_timestamp())

    def fit(self, generator, epochs=10, plot_callback=None, *args, **kwargs):
        # Loop over epochs
        for _ in range(epochs):
            self.model.fit_generator(
                generator,
                epochs=self.current_epoch + 1,
                initial_epoch=self.current_epoch,
                *args, **kwargs
            )

            self.current_epoch += 1

            # After each epoch predict on test images & show them
            if plot_callback:
                plot_callback(self.model)

            # Save logfile
            if self.weight_filepath:
                self.save()

    @staticmethod
    def current_timestamp():
        return datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    def SepConv_BN(self, x, filters, prefix, stride=1, kernel_size=3, rate=1, depth_activation=False, epsilon=1e-3):
        # 计算padding的数量，hw是否需要收缩
        if stride == 1:
            depth_padding = 'same'
        else:
            kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
            pad_total = kernel_size_effective - 1
            pad_beg = pad_total // 2
            pad_end = pad_total - pad_beg
            x = ZeroPadding2D((pad_beg, pad_end))(x)
            depth_padding = 'valid'

        # 如果需要激活函数
        if not depth_activation:
            x = Activation('relu')(x)

        # 分离卷积，首先3x3分离卷积，再1x1卷积
        # 3x3采用膨胀卷积
        x = DepthwiseConv2D((kernel_size, kernel_size), strides=(stride, stride), dilation_rate=(rate, rate),
                            padding=depth_padding, use_bias=False, name=prefix + '_depthwise')(x)
        x = BatchNormalization(name=prefix + '_depthwise_BN', epsilon=epsilon)(x)
        if depth_activation:
            x = Activation('relu')(x)

        # 1x1卷积，进行压缩
        x = Conv2D(filters, (1, 1), padding='same',
                   use_bias=False, name=prefix + '_pointwise')(x)
        x = BatchNormalization(name=prefix + '_pointwise_BN', epsilon=epsilon)(x)
        if depth_activation:
            x = Activation('relu')(x)

        return x

    def deeplabv3(self, backbone="mobilenet", downsample_factor=16):
        inputs = Input((self.img_rows, self.img_cols, self.channel))#none 256 256 3
        if backbone == "xception":
            x, atrous_rates, skip1 = Xception(inputs, alpha=1., downsample_factor=downsample_factor)
        elif backbone == "mobilenet":###x none 32 32 320  atrous_rates 6 12 18  skip1 128 128 24
            x, atrous_rates, skip1 = mobilenetV2(inputs, alpha=1., downsample_factor=downsample_factor)
        else:
            raise ValueError('Unsupported backbone - `{}`, Use mobilenet, xception.'.format(backbone))

        size_before = tf.keras.backend.int_shape(x)#x none 32 32 320

        # 分支0
        b0 = Conv2D(256, (1, 1), padding='same', use_bias=False, name='aspp0')(x)
        b0 = BatchNormalization(name='aspp0_BN', epsilon=1e-5)(b0)
        b0 = Activation('relu', name='aspp0_activation')(b0)  # None 32,32,256

        # 分支1 rate = 6 (12) b1: None, 32, 32, 256
        b1 = self.SepConv_BN(x, 256, 'aspp1',
                             rate=atrous_rates[0], depth_activation=True, epsilon=1e-5)# None 32,32,256
        # 分支2 rate = 12 (24)b2: None, 16, 16, 256
        b2 = self.SepConv_BN(x, 256, 'aspp2',
                             rate=atrous_rates[1], depth_activation=True, epsilon=1e-5)
        # 分支3 rate = 18 (36)
        b3 = self.SepConv_BN(x, 256, 'aspp3',
                             rate=atrous_rates[2], depth_activation=True, epsilon=1e-5)

        # 分支4 全部求平均后，再利用expand_dims扩充维度，之后利用1x1卷积调整通道
        b4 = GlobalAveragePooling2D()(x)  # None, 320
        b4 = Lambda(lambda x: K.expand_dims(x, 1))(b4)  # None,1, 320
        b4 = Lambda(lambda x: K.expand_dims(x, 1))(b4)  # None,1,1,320

        b4 = Conv2D(256, (1, 1), padding='same', use_bias=False, name='image_pooling')(b4) ### None,1,1,256
        b4 = BatchNormalization(name='image_pooling_BN', epsilon=1e-5)(b4)
        b4 = Activation('relu')(b4)### None,1,1,256
        # 直接利用resize_images扩充hw

        # b4 = Lambda(lambda x: tf.image.resize_images(x, size_before[1:3], align_corners=True))(b4)###
        # b4 = Lambda(lambda x: tf.image.resize(x, size_before[1:3], align_corners=True))(b4)###
        b4 = Lambda(lambda x: tf.compat.v1.image.resize_images(x, size_before[1:3], align_corners=True))(b4)### None,32,32,256
        # b4 = Lambda(lambda x: tf.image.resize(x, size_before[1:3]))(b4)####None,16,16,256

        # -----------------------------------------#
        #   将五个分支的内容堆叠起来
        #   然后1x1卷积整合特征。
        # -----------------------------------------#
        x = Concatenate()([b4, b0, b1, b2, b3])  ####None,32,32,1280
        # 利用conv2d压缩 32,32,256
        x = Conv2D(256, (1, 1), padding='same', use_bias=False, name='concat_projection')(x)  ####None,32,32,256
        x = BatchNormalization(name='concat_projection_BN', epsilon=1e-5)(x)  ####None,32,32,256
        x = Activation('relu')(x)  ####None,32,32,256
        x = Dropout(0.1)(x)  ####None,32,32,256

        skip_size = tf.keras.backend.int_shape(skip1)  # None 128 128 24
        # -----------------------------------------#
        #   将加强特征边上采样
        # -----------------------------------------#
        # x = Lambda(lambda xx: tf.image.resize_images(xx, skip_size[1:3], align_corners=True))(x)
        # x = Lambda(lambda xx: tf.image.resize(xx, skip_size[1:3]))(x) #None 64 64 256
        x = Lambda(lambda xx: tf.compat.v1.image.resize_images(xx, skip_size[1:3], align_corners=True))(x)###None 128 128 256
        # ----------------------------------#
        #   skip1浅层特征边
        # ----------------------------------#
        dec_skip1 = Conv2D(48, (1, 1), padding='same', use_bias=False, name='feature_projection0')(
            skip1)  # None,128,128,48
        dec_skip1 = BatchNormalization(name='feature_projection0_BN', epsilon=1e-5)(dec_skip1)
        dec_skip1 = Activation(tf.nn.relu)(dec_skip1)###None 128 128 48

        # -----------------------------------------#
        #   与浅层特征堆叠后利用卷积进行特征提取
        # -----------------------------------------#
        x = Concatenate()([x, dec_skip1])  ####None,128 128,304
        x = self.SepConv_BN(x, 256, 'decoder_conv0',
                            depth_activation=True, epsilon=1e-5)  ####None,128 128,256
        x = self.SepConv_BN(x, 256, 'decoder_conv1',
                            depth_activation=True, epsilon=1e-5)  ####None,128 128,256

        # -----------------------------------------#
        #   获得每个像素点的分类
        # -----------------------------------------#
        # 512,512
        size_before3 = tf.keras.backend.int_shape(inputs)  # None,256,256,5
        # 512,512,21
        x = Conv2D(1, (1, 1), padding='same')(x)
        # x = Lambda(lambda xx:tf.image.resize(xx,size_before3[1:3]))(x)#None,256,256,1
        x = Lambda(lambda xx: tf.compat.v1.image.resize_images(xx, size_before3[1:3], align_corners=True))(x)
        # x = Softmax()(x)

        # x = UpSampling2D(size=(4, 4))(x) #####None,256,256,256
        x = Conv2D(1, (1, 1), activation="sigmoid")(x)  ##output 0-1 grayscale image, need threshold.#####None,256,256,1
        # conv10 = Conv2D(n_label, (1, 1), activation="softmax")(conv9) ##直接选取两类中最大概率的类别

        model = Model(inputs, x, name='deeplabv3plus')
        model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

        return model

    def deeplabv3plus_improve(self, backbone="mobilenet", downsample_factor=16):
        inputs = Input((self.img_rows, self.img_cols, self.channel))#none 256 256 3
        if backbone == "xception":
            x, atrous_rates, skip1 = Xception(inputs, alpha=1., downsample_factor=downsample_factor)
        elif backbone == "mobilenet":###x none 32 32 320  atrous_rates 6 12 18  skip1 128 128 24
            x, atrous_rates, skip1 = mobilenetV2(inputs, alpha=1., downsample_factor=downsample_factor)
        else:
            raise ValueError('Unsupported backbone - `{}`, Use mobilenet, xception.'.format(backbone))

        size_before = tf.keras.backend.int_shape(x)#x none 32 32 320

        # 分支0
        b0 = Conv2D(256, (1, 1), padding='same', use_bias=False, name='aspp0')(x)
        b0 = BatchNormalization(name='aspp0_BN', epsilon=1e-5)(b0)
        b0 = Activation('relu', name='aspp0_activation')(b0)  # None 32,32,256

        # 2-6. 5个空洞卷积分支 rate=3,6,9,12,18
        rates = [3, 6, 9, 12, 18]
        branches = [b0]
        for i, rate in enumerate(rates):
            b = Conv2D(256, 3, padding='same', dilation_rate=rate, use_bias=False)(x)
            b = BatchNormalization()(b)
            b = Activation('relu')(b)
            branches.append(b)

        # 7. Image Pooling分支 对应图中粉色框
        # x shape: (None, 32, 32, 320)
        b_pool = GlobalAveragePooling2D()(x)  # (None, 320)  压缩空间维度，只剩batch+channel
        b_pool = tf.expand_dims(b_pool, axis=1)  # (None, 1, 320)
        b_pool = tf.expand_dims(b_pool, axis=1)  # (None, 1, 1, 320)  恢复H,W=1，变成4D张量

        b_pool = Conv2D(256, 1, padding='same', use_bias=False)(b_pool)  # (None,1,1,256)
        b_pool = BatchNormalization()(b_pool)
        b_pool = Activation('relu')(b_pool)
        # 上采样回原尺寸
        b_pool = Lambda(lambda xx: tf.compat.v1.image.resize_images(xx, size_before[1:3], align_corners=True))(b_pool)
        branches.append(b_pool)  # 共7个分支

        concat = Concatenate(axis=-1)(branches)  # B,H,W,filters*7  NOne 32 32 1792
        gap = GlobalAveragePooling2D()(concat)  # B, filters*7 NONE 1792
        s = Dense(256, activation='relu')(gap)   ##NONE 256
        s = Dense(len(branches), activation='softmax')(s)  # B,7  对应图中深橙色S   ##NONE 256

        weighted_branches = []
        for i, b in enumerate(branches):
            # 取出第i个权重 B,1
            w = Lambda(lambda x: x[:, i:i + 1])(s)
            w = Reshape((1, 1, 1))(w)  # B,1,1,1 可广播
            # ⊗ = Hadamard乘积 / 逐元素乘法
            wb = Multiply()([b, w])
            weighted_branches.append(wb)   ##  7个分支，none 32 32 256

        xx0_proj = Conv2D(256, kernel_size=1, padding="same", use_bias=False)(x)
        xx0_proj = BatchNormalization()(xx0_proj)

        weighted_branches.append(xx0_proj)
        fused = Add()(weighted_branches)  # 也可用Concatenate+Conv1x1
        fused = Conv2D(256, 1, padding='same', use_bias=False)(fused)# NONE 32 32 256

        high_level = TransformerBottleneck(fused, num_heads=8, projection_dim=256)  ###NONE 32 32 256

        # ----- Decoder 对应图红色虚线框 -----
        # -----------------------------------------#
        #   将加强特征边上采样
        # -----------------------------------------#
        # High-level Features Upsample x4
        skip_size = tf.keras.backend.int_shape(skip1)  # None 128 128 24
        high_up = Lambda(lambda xx: tf.compat.v1.image.resize_images(xx, skip_size[1:3], align_corners=True))(high_level)  # None 128 128 256

        # ----------------------------------#
        #   skip1浅层特征边# Low-level Features Conv1x1
        # ----------------------------------#
        dec_skip1 = Conv2D(48, (1, 1), padding='same', use_bias=False, name='feature_projection0')(skip1)  # None,128,128,48
        dec_skip1 = BatchNormalization(name='feature_projection0_BN', epsilon=1e-5)(dec_skip1)
        dec_skip1 = Activation(tf.nn.relu)(dec_skip1)###None 128 128 48

        dec = DualAttentionGate(dec_skip1, high_up,  filters=256)#None 128 128 256-->None 128 128 256
        dec = Conv2D(128, 3, padding='same', activation='relu')(dec)
        dec = BatchNormalization()(dec)
        dec = Dropout(0.2)(dec)

        size_input = tf.keras.backend.int_shape(inputs)
        out = UpSampling2D(size=(size_input[1] // skip_size[1], size_input[2] // skip_size[2]),interpolation='bilinear')(dec)

        # Upsample x4 还原到原图
        # size_before3 = tf.keras.backend.int_shape(inputs)
        # out = Lambda(lambda xx: tf.compat.v1.image.resize_images(xx, size_before3[1:3], align_corners=True))(dec)##>None 256 256 64

        out = Conv2D(64, 3, padding='same', activation='relu')(out)
        out = BatchNormalization()(out)
        out = Dropout(0.1)(out)
        out = Conv2D(1, 1, padding='same', activation='sigmoid', name='output')(out)

        model = Model(inputs, out, name="deeplabv3plus_improve")
        model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

        return model

    def unet(self):
        self.channel  = 3
        # ------------------- 输入层 -------------------
        inputs = Input((self.img_rows, self.img_cols, self.channel))

        # ------------------- 编码器（下采样） -------------------
        # Block 1
        conv1 = Conv2D(64, (3, 3), activation="relu", padding="same", kernel_initializer="he_normal")(inputs)
        conv1 = Conv2D(64, (3, 3), activation="relu", padding="same", kernel_initializer="he_normal")(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        # Block 2
        conv2 = Conv2D(128, (3, 3), activation="relu", padding="same", kernel_initializer="he_normal")(pool1)
        conv2 = Conv2D(128, (3, 3), activation="relu", padding="same", kernel_initializer="he_normal")(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        # Block 3
        conv3 = Conv2D(256, (3, 3), activation="relu", padding="same", kernel_initializer="he_normal")(pool2)
        conv3 = Conv2D(256, (3, 3), activation="relu", padding="same", kernel_initializer="he_normal")(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        # Block 4
        conv4 = Conv2D(512, (3, 3), activation="relu", padding="same", kernel_initializer="he_normal")(pool3)
        conv4 = Conv2D(512, (3, 3), activation="relu", padding="same", kernel_initializer="he_normal")(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        # 瓶颈层
        conv5 = Conv2D(1024, (3, 3), activation="relu", padding="same", kernel_initializer="he_normal")(pool4)
        conv5 = Conv2D(1024, (3, 3), activation="relu", padding="same", kernel_initializer="he_normal")(conv5)

        # ------------------- 解码器（上采样+跳跃连接） -------------------
        # Block 6
        up6 = UpSampling2D(size=(2, 2))(conv5)
        up6 = concatenate([up6, conv4], axis=-1)  # axis=-1 更通用（通道维度）
        conv6 = Conv2D(512, (3, 3), activation="relu", padding="same", kernel_initializer="he_normal")(up6)
        conv6 = Conv2D(512, (3, 3), activation="relu", padding="same", kernel_initializer="he_normal")(conv6)

        # Block 7
        up7 = UpSampling2D(size=(2, 2))(conv6)
        up7 = concatenate([up7, conv3], axis=-1)
        conv7 = Conv2D(256, (3, 3), activation="relu", padding="same", kernel_initializer="he_normal")(up7)
        conv7 = Conv2D(256, (3, 3), activation="relu", padding="same", kernel_initializer="he_normal")(conv7)

        # Block 8
        up8 = UpSampling2D(size=(2, 2))(conv7)
        up8 = concatenate([up8, conv2], axis=-1)
        conv8 = Conv2D(128, (3, 3), activation="relu", padding="same", kernel_initializer="he_normal")(up8)
        conv8 = Conv2D(128, (3, 3), activation="relu", padding="same", kernel_initializer="he_normal")(conv8)

        # Block 9
        up9 = UpSampling2D(size=(2, 2))(conv8)
        up9 = concatenate([up9, conv1], axis=-1)
        conv9 = Conv2D(64, (3, 3), activation="relu", padding="same", kernel_initializer="he_normal")(up9)
        conv9 = Conv2D(64, (3, 3), activation="relu", padding="same", kernel_initializer="he_normal")(conv9)

        # ------------------- 输出层 -------------------
        # 7分类，softmax激活
        # outputs = Conv2D(7, (1, 1), activation="softmax")(conv9)
        outputs = Conv2D(1, (1, 1), activation="sigmoid")(conv9)

        # ------------------- 构建&编译模型 -------------------
        model = Model(inputs=inputs, outputs=outputs, name='standard_unet')
        # 多分类编译：损失函数用categorical_crossentropy，优化器Adam
        # model.compile(
        #     optimizer=Adam(learning_rate=1e-4),
        #     loss='categorical_crossentropy',
        #     metrics=['accuracy']
        # )
        model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

        return model

    def seg_hrnet(self):
        print('Network is seg_hrnet')
        inputs = Input((self.img_rows, self.img_cols, 3))
        # inputs = Input(batch_shape=(batch_size,) + (height, width, channel))

        x = stem_net(inputs)

        x = transition_layer1(x)
        x0 = make_branch1_0(x[0])
        x1 = make_branch1_1(x[1])
        x = fuse_layer1([x0, x1])
        # x = Dropout(0.9)(x)

        x = transition_layer2(x)
        x0 = make_branch2_0(x[0])
        x1 = make_branch2_1(x[1])
        x2 = make_branch2_2(x[2])
        x = fuse_layer2([x0, x1, x2])
        # x = Dropout(0.9)(x)

        x = transition_layer3(x)
        x0 = make_branch3_0(x[0])
        x1 = make_branch3_1(x[1])
        x2 = make_branch3_2(x[2])
        x3 = make_branch3_3(x[3])
        x = fuse_layer3([x0, x1, x2, x3])
        # x = Dropout(0.9)(x)

        out = final_layer(x, classes=1)

        model = Model(inputs=inputs, outputs=out)

        model.compile(optimizer=Adam(lr=1e-4),
                      loss="binary_crossentropy",
                      #loss="categorical_crossentropy",
                      metrics=['accuracy'])
        return model





