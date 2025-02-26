
#import tensorflow as tf
import tensorflow.keras.layers as klayers
import tensorflow.keras.regularizers as kreg
import tensorflow.keras.initializers as kinit

'''
 VGG backbone (truncated) for Convolutional Pose Machines,
 Shih-En Wei, Varun Ramakrishna, Takeo Kanade, Yaser Sheikh
 
 https://arxiv.org/abs/1602.00134
'''


def conv(x, nf, ks, kernel_reg, **kwargs):
    x = klayers.Conv2D(nf, (ks, ks),
                       padding='same',
                       kernel_regularizer=kreg.l2(kernel_reg),
                       kernel_initializer=kinit.random_normal(stddev=0.01),
                       bias_initializer=kinit.constant(0.0),
                       **kwargs
                       )(x)
    return x

def maxpool(x, ks, st, **kwargs):
    x = klayers.MaxPooling2D((ks, ks),
                             strides=(st, st),
                             **kwargs)(x)
    return x

def vgg19_truncated(x, kernel_reg):
    # block1
    x = conv(x, 64, 3, kernel_reg, name="conv1_1", activation='relu')
    x = conv(x, 64, 3, kernel_reg, name="conv1_2", activation='relu')
    x = maxpool(x, 2, 2, name="pool1_1")

    # block2
    x = conv(x, 128, 3, kernel_reg, name="conv2_1", activation='relu')
    x = conv(x, 128, 3, kernel_reg, name="conv2_2", activation='relu')
    x = maxpool(x, 2, 2, name="pool2_1")

    # block3
    x = conv(x, 256, 3, kernel_reg, name="conv3_1", activation='relu')
    x = conv(x, 256, 3, kernel_reg, name="conv3_2", activation='relu')
    x = conv(x, 256, 3, kernel_reg, name="conv3_3", activation='relu')
    x = conv(x, 256, 3, kernel_reg, name="conv3_4", activation='relu')
    x = maxpool(x, 2, 2, name="pool3_1")

    # block4
    x = conv(x, 512, 3, kernel_reg, name="conv4_1", activation='relu')
    x = conv(x, 512, 3, kernel_reg, name="conv4_2", activation='relu')

    # could include conv 4_3 and 4_4 or further VGG layers

    # cpm
    x = conv(x, 256, 3, kernel_reg, name="conv4_3_CPM", activation='relu')
    x = conv(x, 128, 3, kernel_reg, name="conv4_4_CPM", activation='relu')

    return x


from_vgg = {
    'conv1_1': 'block1_conv1',
    'conv1_2': 'block1_conv2',
    'conv2_1': 'block2_conv1',
    'conv2_2': 'block2_conv2',
    'conv3_1': 'block3_conv1',
    'conv3_2': 'block3_conv2',
    'conv3_3': 'block3_conv3',
    'conv3_4': 'block3_conv4',
    'conv4_1': 'block4_conv1',
    'conv4_2': 'block4_conv2'
}