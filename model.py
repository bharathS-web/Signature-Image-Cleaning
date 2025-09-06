"""
Lightweight U-Net variant optimized for signature denoising.
- Keeps input shape at (224, 224, 1) by default (do not change if you rely on an existing model).
- Uses depthwise-separable convolutions (SeparableConv2D) to reduce parameters + FLOPs.
- Uses a small number of downsampling levels (3) to keep compute low.
- Uses bilinear UpSampling (no transposed conv) to reduce decoder params.
- Optional SE (squeeze-and-excite) channel attention block (disabled by default).
- Minimal, effective, and easy to train.

Usage:
    model = build_light_unet(input_shape=(224,224,1), base_filters=16, use_se=False)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
"""

import tensorflow as tf
from tensorflow.keras.layers import ( # type: ignore
    Input, SeparableConv2D, Conv2D, MaxPooling2D, UpSampling2D,
    Concatenate, BatchNormalization, Activation, GlobalAveragePooling2D,
    Reshape, Multiply, Dense
)
from tensorflow.keras.models import Model # type: ignore


def se_block(x, reduction=8):
    """Squeeze-and-Excite block (lightweight)."""
    filters = x.shape[-1]
    se = GlobalAveragePooling2D()(x)
    se = Dense(max(filters // reduction, 4), activation='relu')(se)
    se = Dense(filters, activation='sigmoid')(se)
    se = Reshape((1, 1, filters))(se)
    return Multiply()([x, se])


def conv_block_sep(x, filters, kernel_size=3, activation='relu', use_bn=True):
    """Depthwise-separable conv block: SeparableConv2D -> BN -> Activation -> SeparableConv2D -> BN -> Activation"""
    x = SeparableConv2D(filters, kernel_size, padding='same', use_bias=False)(x)
    if use_bn:
        x = BatchNormalization()(x)
    x = Activation(activation)(x)

    x = SeparableConv2D(filters, kernel_size, padding='same', use_bias=False)(x)
    if use_bn:
        x = BatchNormalization()(x)
    x = Activation(activation)(x)
    return x


def build_light_unet(input_shape=(224, 224, 1), base_filters=16, use_se=False, name="light_unet"):
    """
    Build a lightweight U-Net.

    Parameters
    ----------
    input_shape : tuple
        Input image shape, e.g. (224,224,1). DO NOT CHANGE if your model expects this shape.
    base_filters : int
        Number of filters in the first encoder block. Typical small value: 8-32. Default = 16.
    use_se : bool
        Whether to include Squeeze-and-Excite blocks after each encoder block (adds small overhead).
    name : str
        Model name.

    Returns
    -------
    tf.keras.Model
        Compiled model architecture (not compiled with optimizer/loss here).
    """

    inputs = Input(shape=input_shape)

    # Encoder (3 downsampling levels)
    c1 = conv_block_sep(inputs, base_filters)           # e.g., 16
    if use_se:
        c1 = se_block(c1)
    p1 = MaxPooling2D(pool_size=(2, 2))(c1)             # 112x112

    c2 = conv_block_sep(p1, base_filters * 2)           # e.g., 32
    if use_se:
        c2 = se_block(c2)
    p2 = MaxPooling2D(pool_size=(2, 2))(c2)             # 56x56

    c3 = conv_block_sep(p2, base_filters * 4)           # e.g., 64
    if use_se:
        c3 = se_block(c3)
    p3 = MaxPooling2D(pool_size=(2, 2))(c3)             # 28x28

    # Bottleneck
    bn = conv_block_sep(p3, base_filters * 8)           # e.g., 128
    if use_se:
        bn = se_block(bn)

    # Decoder (symmetric, bilinear upsampling + 1x1 conv to reduce channels if needed)
    u3 = UpSampling2D(size=(2, 2), interpolation='bilinear')(bn)  # 56x56
    u3 = Concatenate()([u3, c3])
    c4 = conv_block_sep(u3, base_filters * 4)                   # 64

    u2 = UpSampling2D(size=(2, 2), interpolation='bilinear')(c4) # 112x112
    u2 = Concatenate()([u2, c2])
    c5 = conv_block_sep(u2, base_filters * 2)                   # 32

    u1 = UpSampling2D(size=(2, 2), interpolation='bilinear')(c5) # 224x224
    u1 = Concatenate()([u1, c1])
    c6 = conv_block_sep(u1, base_filters)                       # 16

    # Output: single-channel (sigmoid) for denoising / mask / reconstruction
    outputs = Conv2D(1, kernel_size=1, activation='sigmoid')(c6)

    model = Model(inputs, outputs, name=name)
    return model


# Example usage (uncomment to run under normal Python environment):
# model = build_light_unet(input_shape=(224,224,1), base_filters=16, use_se=False)
# model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='binary_crossentropy', metrics=['accuracy'])
# model.summary()
