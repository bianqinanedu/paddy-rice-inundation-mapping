import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def conv(inputs, filters, kernel_size, strides_=1, padding_='same', activation=None):
    x = Conv2D(filters, kernel_size, strides=strides_, padding=padding_)(inputs)
    x = BatchNormalization(epsilon=1e-5, momentum=0.1)(x)
    if activation is not None:
        x = Activation(activation)(x)
    return x

def transition_layer(inputs, in_channels, out_channels):
    outputs = []
    for i in range(len(inputs)):
        x = inputs[i]
        x = conv(x, out_channels[i], 3)
        outputs.append(x)
    return outputs

def Bottleneck(inputs, filters):
    # 🔴 替换为你的多尺度空洞残差模块
    return MultiScaleDilatedResidualBlock(inputs, filters)

def basic(inputs, filters):
    return MultiScaleDilatedResidualBlock(inputs, filters)

def stage(inputs, num_modules, channels, multi_scale_output=True):
    outputs = inputs
    for _ in range(num_modules):
        feats = []
        for i in range(len(outputs)):
            x = outputs[i]
            for j in range(len(outputs)):
                if j < i:
                    x = conv(x, channels[i], 3)
                elif j > i:
                    x = conv(x, channels[i], 3)
            x = bottleneck(x, channels[i])
            feats.append(x)
        outputs = feats
    return outputs if multi_scale_output else [outputs[-1]]

# -----------------------------------------------------------------------------
# ✅ 最终改进版 HRNet (已集成全部3个创新模块)
# -----------------------------------------------------------------------------
def hrnet_keras(self):
    print('Network is hrnet (improved with 3 modules)')
    input_size = (self.img_rows, self.img_cols, 3)

    channels_2 = [32, 64]
    channels_3 = [32, 64, 128]
    channels_4 = [32, 64, 128, 256]
    num_modules_2 = 1
    num_modules_3 = 3
    num_modules_4 = 2

    inputs = Input(input_size)

    # ===================== 主干网络 =====================
    x = conv(inputs, 64, 3, strides_=2)
    x = conv(x, 64, 3, strides_=2)
    x = Activation('relu')(x)

    la1 = layer1(x)
    tr1 = transition_layer([la1], [256], channels_2)
    st2 = stage(tr1, num_modules_2, channels_2)
    tr2 = transition_layer(st2, channels_2, channels_3)
    st3 = stage(tr2, num_modules_3, channels_3)
    tr3 = transition_layer(st3, channels_3, channels_4)
    st4 = stage(tr3, num_modules_4, channels_4, multi_scale_output=False)

    # ===================== 创新模块2：Transformer瓶颈 =====================
    st4_transformer = TransformerBottleneck(st4[0])

    # ===================== 解码器上采样 + 创新模块3：双注意力门 =====================
    up1 = UpSampling2D()(st4_transformer)
    up1 = conv(up1, 32, 3)
    up1 = Activation('relu')(up1)

    up2 = UpSampling2D()(up1)
    up2 = conv(up2, 32, 3)
    up2 = Activation('relu')(up2)

    # 跳跃连接 → 双注意力门
    gate = DualAttentionGate(st3[0], up1, 32)
    up2 = Add()([up2, gate])

    up2 = Dropout(0.9)(up2)
    final = conv(up2, 1, 1, padding_='valid', activation='sigmoid')

    model = Model(inputs=inputs, outputs=final)
    model.compile(optimizer=Adam(lr=1e-5), loss="binary_crossentropy", metrics=['accuracy'])
    return model