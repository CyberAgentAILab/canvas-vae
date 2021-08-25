import logging

import tensorflow as tf
import tensorflow.keras.layers as layers

logger = logging.getLogger(__name__)


def SequentialDecoder(output_channels):
    return tf.keras.Sequential([
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2DTranspose(filters=256,
                                        kernel_size=2,
                                        strides=2,
                                        activation='relu'),  # 8 x 8 -> 16 x 16
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2DTranspose(
            filters=256, kernel_size=2, strides=2,
            activation='relu'),  # 16 x 16 -> 32 x 32
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2DTranspose(
            filters=128, kernel_size=2, strides=2,
            activation='relu'),  # 32 x 32 -> 64 x 64
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2DTranspose(
            filters=128, kernel_size=2, strides=2,
            activation='relu'),  # 64 x 64 -> 128 x 128
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2DTranspose(
            filters=64, kernel_size=2, strides=2,
            activation='relu'),  # 128 x 128 -> 256 x 256
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(
            filters=output_channels,
            kernel_size=3,
            strides=1,
            padding='same',
        ),
    ])


class ResidualBlock2(tf.keras.layers.Layer):
    """A residual block.

    # Arguments
        filters: integer, filters of the bottleneck layer.
        kernel_size: default 3, kernel size of the bottleneck layer.
        stride: default 1, stride of the first layer.
        conv_shortcut: default False, use convolution shortcut if True,
            otherwise identity shortcut.
        name: string, block label.
    # Returns
        Output tensor for the residual block.
    """
    def __init__(
        self,
        filters,
        kernel_size=3,
        stride=1,
        conv_shortcut=False,
        epsilon=1.001e-5,
        name=None,
    ):
        super().__init__(name=name)
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv_shortcut = conv_shortcut

        self.bn0 = layers.BatchNormalization(epsilon=epsilon,
                                             name=name + '_preact_bn')
        self.relu0 = layers.Activation('relu', name=name + '_preact_relu')

        self.shortcut = None
        if self.conv_shortcut:
            self.shortcut = layers.Conv2DTranspose(
                4 * filters,
                stride,
                strides=stride,
                name=name + '_0_conv',
            )
        elif self.stride > 1:
            self.shortcut = layers.UpSampling2D(stride)

        self.conv1 = layers.Conv2D(
            filters,
            1,
            strides=1,
            use_bias=False,
            name=name + '_1_conv',
        )
        self.bn1 = layers.BatchNormalization(
            epsilon=epsilon,
            name=name + '_1_bn',
        )
        self.relu1 = layers.Activation('relu', name=name + '_1_relu')
        self.conv2 = layers.Conv2DTranspose(
            filters,
            kernel_size,
            strides=stride,
            use_bias=False,
            name=name + '_2_conv',
            padding='same',
        )
        self.bn2 = layers.BatchNormalization(
            epsilon=epsilon,
            name=name + '_2_bn',
        )
        self.relu2 = layers.Activation('relu', name=name + '_2_relu')
        self.conv3 = layers.Conv2D(4 * filters, 1, name=name + '_3_conv')
        self.add = layers.Add(name=name + '_out')

    def call(self, x):
        preact = self.bn0(x)
        preact = self.relu0(preact)
        if self.conv_shortcut:
            shortcut = self.shortcut(preact)
        elif self.stride > 1:
            shortcut = self.shortcut(x)
        else:
            shortcut = x

        x = self.conv1(preact)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.add([shortcut, x])
        return x


class ResidualStack2(tf.keras.layers.Layer):
    """A set of stacked residual blocks.

    # Arguments
        filters: integer, filters of the bottleneck layer in a block.
        blocks: integer, blocks in the stacked blocks.
        stride1: default 2, stride of the first layer in the first block.
        name: string, stack label.
    # Returns
        Output tensor for the stacked blocks.
    """
    def __init__(self, filters, blocks, stride1=2, name=None):
        super().__init__(name=name)

        self.blocks = [
            ResidualBlock2(filters, conv_shortcut=True, name=name + '_block1'),
        ]
        for i in range(2, blocks):
            self.blocks.append(
                ResidualBlock2(filters, name=name + '_block' + str(i)))
        self.blocks.append(
            ResidualBlock2(
                filters,
                stride=stride1,
                name=name + '_block' + str(blocks),
            ))

    def call(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class ResNetDecoder(tf.keras.Model):
    def __init__(self,
                 output_channels=3,
                 filters=None,
                 blocks=None,
                 name=None):
        super().__init__(name=name)

        filters = filters or [512, 256, 128, 64]
        blocks = blocks or [3, 6, 4, 3]

        self.conv1 = ResidualStack2(filters[0], blocks[0], name='conv1')
        self.conv2 = ResidualStack2(filters[1], blocks[1], name='conv2')
        self.conv3 = ResidualStack2(filters[2], blocks[2], name='conv3')
        self.conv4 = ResidualStack2(
            filters[3],
            blocks[3],
            stride1=1,
            name='conv4',
        )

        self.post_bn = layers.BatchNormalization(epsilon=1.001e-5,
                                                 name='post_bn')
        self.post_relu = layers.Activation('relu', name='post_relu')
        self.post_upsample = layers.UpSampling2D(2, name='post_upsample')
        self.post_conv = layers.Conv2DTranspose(output_channels,
                                                7,
                                                strides=2,
                                                padding='same',
                                                name='post_conv')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.post_bn(x)
        x = self.post_relu(x)
        x = self.post_upsample(x)
        x = self.post_conv(x)
        return x
