# ResNet models for Keras.
# Reference papers
# - [Deep Residual Learning for Image Recognition]
#  (https://arxiv.org/abs/1512.03385) (CVPR 2016 Best Paper Award)
# Reference implementations
# - [TensorNets]
#  (https://github.com/taehoonlee/tensornets/blob/master/tensornets/resnets.py)
# - [Caffe ResNet]
#  (https://github.com/KaimingHe/deep-residual-networks/tree/master/prototxt)
#
# Modified by Jacob M. Graving from:
# https://github.com/keras-team/keras-applications/blob/
# master/keras_applications/resnet_common.py

# to match the stride 16 ResNet found here:
# https://github.com/tensorflow/tensorflow/blob/
# master/tensorflow/contrib/slim/python/slim/nets/resnet_v1.py

# All modifications are Copyright 2019 Jacob M. Graving <jgraving@gmail.com>

# From https://github.com/jgraving/deepposekit. See imagenet_resnet_LICENSE
# Adapted by Allen Lee @ Branson Lab, JRC/HHMI Oct 2019


#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function

import os
import tensorflow.keras as keras
from tensorflow.keras.layers import Layer
import logging

import tensorflow
tf = tensorflow.compat.v1

from keras_applications import imagenet_utils

_obtain_input_shape = imagenet_utils._obtain_input_shape

backend = keras.backend
layers = keras.layers
models = keras.models
keras_utils = keras.utils


BASE_WEIGHTS_PATH = (
    "https://github.com/keras-team/keras-applications/releases/download/resnet/"
)
WEIGHTS_HASHES = {
    "resnet50": (
        "2cb95161c43110f7111970584f804107",
        "4d473c1dd8becc155b73f8504c6f6626",
    ),
    "resnet101": (
        "f1aeb4b969a6efcfb50fad2f0c20cfc5",
        "88cf7a10940856eca736dc7b7e228a21",
    ),
    "resnet152": (
        "100835be76be38e30d865e96f2aaae62",
        "ee4c566cf9a93f14d82f913c2dc6dd0c",
    ),
}


def block1(
    x, filters, kernel_size=3, stride=1, conv_shortcut=True, name=None, dilation=1
):
    """A residual block.
    # Arguments
        x: input tensor.
        filters: integer, filters of the bottleneck layer.
        kernel_size: default 3, kernel size of the bottleneck layer.
        stride: default 1, stride of the first layer.
        conv_shortcut: default True, use convolution shortcut if True,
            otherwise identity shortcut.
        name: string, block label.
    # Returns
        Output tensor for the residual block.
    """
    bn_axis = 3 if backend.image_data_format() == "channels_last" else 1

    if conv_shortcut is True:
        shortcut = layers.Conv2D(4 * filters, 1, strides=stride, name=name + "_0_conv")(
            x
        )
        shortcut = layers.BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5, name=name + "_0_bn"
        )(shortcut)
    else:
        shortcut = x

    x = layers.Conv2D(filters, 1, strides=stride, name=name + "_1_conv")(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + "_1_bn")(
        x
    )
    x = layers.Activation("relu", name=name + "_1_relu")(x)

    x = layers.Conv2D(
        filters,
        kernel_size,
        padding="SAME",
        dilation_rate=dilation,
        name=name + "_2_conv",
    )(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + "_2_bn")(
        x
    )
    x = layers.Activation("relu", name=name + "_2_relu")(x)

    x = layers.Conv2D(4 * filters, 1, name=name + "_3_conv")(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + "_3_bn")(
        x
    )

    x = layers.Add(name=name + "_add")([shortcut, x])
    x = layers.Activation("relu", name=name + "_out")(x)
    return x


def stack1(x, filters, blocks, stride1=2, name=None, dilation=1):
    """A set of stacked residual blocks.
    # Arguments
        x: input tensor.
        filters: integer, filters of the bottleneck layer in a block.
        blocks: integer, blocks in the stacked blocks.
        stride1: default 2, stride of the first layer in the first block.
        name: string, stack label.
    # Returns
        Output tensor for the stacked blocks.
    """
    x = block1(x, filters, stride=stride1, name=name + "_block1")
    for i in range(2, blocks + 1):
        x = block1(
            x,
            filters,
            conv_shortcut=False,
            name=name + "_block" + str(i),
            dilation=dilation,
        )
    return x


def ResNet(
    stack_fn,
    preact,
    use_bias,
    model_name="resnet",
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    **kwargs
):
    """Instantiates the ResNet, ResNetV2, and ResNeXt architecture.
    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.
    # Arguments
        stack_fn: a function that returns output tensor for the
            stacked residual blocks.
        use_bias: whether to use biases for convolutional layers or not
            (True for ResNet and ResNetV2, False for ResNeXt).
        model_name: string, model name.
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor
            (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 inputs channels.
        pooling: optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
    # Returns
        A Keras model instance.
    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """

    if not (weights in {"imagenet", None} or os.path.exists(weights)):
        raise ValueError(
            "The `weights` argument should be either "
            "`None` (random initialization), `imagenet` "
            "(pre-training on ImageNet), "
            "or the path to the weights file to be loaded."
        )

    if weights == "imagenet" and include_top and classes != 1000:
        raise ValueError(
            'If using `weights` as `"imagenet"` with `include_top`'
            " as true, `classes` should be 1000"
        )

    # Determine proper input shape
    input_shape = _obtain_input_shape(
        input_shape,
        default_size=224,
        min_size=32,
        data_format=backend.image_data_format(),
        require_flatten=include_top,
        weights=weights,
    )

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        #if not backend.is_keras_tensor(input_tensor):
        if False: # XXXAL
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    bn_axis = 3 if backend.image_data_format() == "channels_last" else 1

    x = layers.ZeroPadding2D(padding=((3, 3), (3, 3)), name="conv1_pad")(img_input)
    x = layers.Conv2D(64, 7, strides=2, use_bias=use_bias, name="conv1_conv")(x)

    if preact is False:
        x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name="conv1_bn")(
            x
        )
        x = layers.Activation("relu", name="conv1_relu")(x)

    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name="pool1_pad")(x)
    x = layers.MaxPooling2D(3, strides=2, name="pool1_pool")(x)

    x = stack_fn(x)

    if include_top:
        x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
        x = layers.Dense(classes, activation="softmax", name="probs")(x)
    else:
        if pooling == "avg":
            x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
        elif pooling == "max":
            x = layers.GlobalMaxPooling2D(name="max_pool")(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = keras_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    model = models.Model(inputs, x, name=model_name)

    # Load weights.
    if (weights == "imagenet") and (model_name in WEIGHTS_HASHES):
        if include_top:
            file_name = model_name + "_weights_tf_dim_ordering_tf_kernels.h5"
            file_hash = WEIGHTS_HASHES[model_name][0]
        else:
            file_name = model_name + "_weights_tf_dim_ordering_tf_kernels_notop.h5"
            file_hash = WEIGHTS_HASHES[model_name][1]
        weights_path = keras_utils.get_file(
            file_name,
            BASE_WEIGHTS_PATH + file_name,
            cache_subdir="models",
            file_hash=file_hash,
        )
        model.load_weights(weights_path)
        logging.warning('RESNET: Loaded imagenet weights from {}'.format(weights_path))
    elif weights is not None:
        model.load_weights(weights)
        logging.warning('RESNET: Loaded weights')


    return model


def ResNet50(
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    **kwargs
):
    def stack_fn(x):
        x = stack1(x, 64, 3, stride1=1, name="conv2")
        x = stack1(x, 128, 4, name="conv3")
        x = stack1(x, 256, 6, name="conv4")
        x = stack1(x, 512, 3, name="conv5", stride1=1, dilation=2)
        return x

    return ResNet(
        stack_fn,
        False,
        True,
        "resnet50",
        include_top,
        weights,
        input_tensor,
        input_shape,
        pooling,
        classes,
        **kwargs
    )


def ResNet50_32px(
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    **kwargs
):
    def stack_fn(x):
        x = stack1(x, 64, 3, stride1=1, name="conv2")
        x = stack1(x, 128, 4, name="conv3")
        x = stack1(x, 256, 6, name="conv4")
        x = stack1(x, 512, 3, name="conv5")
        return x

    return ResNet(
        stack_fn,
        False,
        True,
        "resnet50",
        include_top,
        weights,
        input_tensor,
        input_shape,
        pooling,
        classes,
        **kwargs
    )

ResNet50_16px = ResNet50

def ResNet50_8px(
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    **kwargs
):
    def stack_fn(x):
        x = stack1(x, 64, 3, stride1=1, name="conv2")
        x = stack1(x, 128, 4, name="conv3")
        x = stack1(x, 256, 6, name="conv4", stride1=1, dilation=2)
        x = stack1(x, 512, 3, name="conv5", stride1=1, dilation=2)
        return x

    return ResNet(
        stack_fn,
        False,
        True,
        "resnet50",
        include_top,
        weights,
        input_tensor,
        input_shape,
        pooling,
        classes,
        **kwargs
    )


def ResNet101(
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    **kwargs
):
    def stack_fn(x):
        x = stack1(x, 64, 3, stride1=1, name="conv2")
        x = stack1(x, 128, 4, name="conv3")
        x = stack1(x, 256, 23, name="conv4")
        x = stack1(x, 512, 3, name="conv5", stride1=1, dilation=2)
        return x

    return ResNet(
        stack_fn,
        False,
        True,
        "resnet101",
        include_top,
        weights,
        input_tensor,
        input_shape,
        pooling,
        classes,
        **kwargs
    )

def ResNet101_32px(
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    **kwargs
):
    def stack_fn(x):
        x = stack1(x, 64, 3, stride1=1, name="conv2")
        x = stack1(x, 128, 4, name="conv3")
        x = stack1(x, 256, 23, name="conv4")
        x = stack1(x, 512, 3, name="conv5")
        return x

    return ResNet(
        stack_fn,
        False,
        True,
        "resnet101",
        include_top,
        weights,
        input_tensor,
        input_shape,
        pooling,
        classes,
        **kwargs
    )

def ResNet152(
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    **kwargs
):
    def stack_fn(x):
        x = stack1(x, 64, 3, stride1=1, name="conv2")
        x = stack1(x, 128, 8, name="conv3")
        x = stack1(x, 256, 36, name="conv4")
        x = stack1(x, 512, 3, name="conv5", stride1=1, dilation=2)
        return x

    return ResNet(
        stack_fn,
        False,
        True,
        "resnet152",
        include_top,
        weights,
        input_tensor,
        input_shape,
        pooling,
        classes,
        **kwargs
    )


MODELS = {"resnet50": ResNet50, "resnet101": ResNet101, "resnet152": ResNet152}

if __name__ == "__main__":

    from tensorflow.keras.applications.resnet50 import preprocess_input
    from tensorflow.keras.layers import Input
    from tensorflow.keras import Model

    input_layer = Input((192, 192, 3))
    model = ResNet50(include_top=False, input_shape=(192, 192, 3))
    pretrained_output = model(input_layer)
    model = Model(inputs=input_layer, outputs=pretrained_output)
