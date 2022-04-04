import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Flatten, Dropout, MaxPool2D
from config import cfg


class CustomConvolution(keras.layers.Layer):
    def __init__(self, filters=64, kernel_size=(1, 1), strides=(1, 1), padding='VALID', bn_momentum=0.99,
                 activation=None, apply_bn=False, name=None):
        super(CustomConvolution, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.bn_momentum = bn_momentum
        self.activation = activation
        self.weights_initializer = tf.initializers.TruncatedNormal(0.0, 0.01)
        self.weights_regularizer = tf.keras.regularizers.l2(0.0005)
        self.convolution = keras.layers.Conv2D(filters, kernel_size, strides, padding,
                                               kernel_initializer=self.weights_initializer,
                                               kernel_regularizer=self.weights_regularizer, name=name)
        self.apply_bn = apply_bn
        if self.apply_bn:
            self.bn = keras.layers.BatchNormalization(momentum=bn_momentum)

    def call(self, inputs, training=None):
        x = self.convolution(inputs)
        if self.apply_bn:
            x = self.bn(x, training=training)
        if self.activation:
            x = self.activation(x)
        return x

    def get_config(self):
        config = super(CustomConvolution, self).get_config()
        config.update(
            {
                'filters': self.filters,
                'kernel_size': self.kernel_size,
                'strides': self.strides,
                'padding': self.padding,
                'bn_momentum': self.bn_momentum,
                'activation': self.activation,
                'apply_bn': self.apply_bn

            }
        )

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class CustomDense(keras.layers.Layer):
    def __init__(self, units=256, activation=tf.nn.relu, bn_momentum=0.99, apply_bn=False, name=None):
        super(CustomDense, self).__init__()
        self.filters = units
        self.bn_momentum = bn_momentum
        self.activation = activation
        self.weights_initializer = tf.initializers.TruncatedNormal(0.0, 0.01)
        self.weights_regularizer = tf.keras.regularizers.l2(0.0005)
        self.dense = keras.layers.Dense(units, kernel_initializer=self.weights_initializer,
                                        kernel_regularizer=self.weights_regularizer, name=name)
        self.apply_bn = apply_bn
        if self.apply_bn:
            self.bn = keras.layers.BatchNormalization(momentum=bn_momentum)

    def call(self, inputs, training=None):
        x = self.dense(inputs)
        if self.apply_bn:
            x = self.bn(x, training=training)
        if self.activation:
            x = self.activation(x)
        return x

    def get_config(self):
        config = super(CustomDense, self).get_config()
        config.update(
            {
                'units': self.units,
                'activation': self.activation,
                'bn_momentum': self.bn_momentum,
                'apply_bn': self.apply_bn

            }
        )

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def get_model(bn_momentum):
    # Initialize the input layer
    input_layer = keras.Input(shape=(224, 224, 3), name='input_layer')

    # Block 1
    x = CustomConvolution(filters=64, kernel_size=(2, 2), strides=(2, 2), padding='SAME',
                          bn_momentum=bn_momentum, activation=tf.nn.relu, apply_bn=True)(
        input_layer)
    x = CustomConvolution(filters=64, kernel_size=(2, 2), strides=(2, 2), padding='SAME',
                          bn_momentum=bn_momentum, activation=tf.nn.relu, apply_bn=True)(
        x)
    x = MaxPool2D((2, 2), padding='same')(x)

    # Block 2
    x = CustomConvolution(filters=128, kernel_size=(3, 3), strides=(3, 3), padding='SAME',
                          bn_momentum=bn_momentum, activation=tf.nn.relu, apply_bn=True)(
        x)
    x = CustomConvolution(filters=128, kernel_size=(3, 3), strides=(3, 3), padding='SAME',
                          bn_momentum=bn_momentum, activation=tf.nn.relu, apply_bn=True)(
        x)
    x = MaxPool2D((2, 2), padding='same')(x)
    # Block 3
    x = CustomConvolution(filters=256, kernel_size=(3, 3), strides=(3, 3), padding='SAME',
                          bn_momentum=bn_momentum, activation=tf.nn.relu, apply_bn=True)(
        x)
    x = CustomConvolution(filters=256, kernel_size=(3, 3), strides=(3, 3), padding='SAME',
                          bn_momentum=bn_momentum, activation=tf.nn.relu, apply_bn=True)(
        x)
    x = CustomConvolution(filters=256, kernel_size=(3, 3), strides=(3, 3), padding='SAME',
                          bn_momentum=bn_momentum, activation=tf.nn.relu, apply_bn=True)(
        x)
    x = MaxPool2D((2, 2), padding='same')(x)
    # Block 4
    x = CustomConvolution(filters=512, kernel_size=(3, 3), strides=(3, 3), padding='SAME',
                          bn_momentum=bn_momentum, activation=tf.nn.relu, apply_bn=True)(
        x)
    x = CustomConvolution(filters=512, kernel_size=(3, 3), strides=(3, 3), padding='SAME',
                          bn_momentum=bn_momentum, activation=tf.nn.relu, apply_bn=True)(
        x)
    x = CustomConvolution(filters=512, kernel_size=(3, 3), strides=(3, 3), padding='SAME',
                          bn_momentum=bn_momentum, activation=tf.nn.relu, apply_bn=True)(
        x)
    x = MaxPool2D((2, 2), padding='same')(x)
    # Block 5
    x = CustomConvolution(filters=512, kernel_size=(3, 3), strides=(3, 3), padding='SAME',
                          bn_momentum=bn_momentum, activation=tf.nn.relu, apply_bn=True)(
        x)
    x = CustomConvolution(filters=512, kernel_size=(3, 3), strides=(3, 3), padding='SAME',
                          bn_momentum=bn_momentum, activation=tf.nn.relu, apply_bn=True)(
        x)
    x = CustomConvolution(filters=512, kernel_size=(3, 3), strides=(3, 3), padding='SAME',
                          bn_momentum=bn_momentum, activation=tf.nn.relu, apply_bn=True)(
        x)
    x = MaxPool2D((2, 2), padding='same')(x)

    # Flattening the layer
    flattened = Flatten()(x)

    # Fully Connected Layers

    dense1 = CustomDense(units=512, activation=tf.nn.relu, apply_bn=True)(flattened)
    dense1 = Dropout(0.5)(dense1)

    # Output layer for dimensions for bbox
    dimension_out = CustomDense(units=3, activation=None, apply_bn=False, name='dimension_out')(dense1)

    # Output layer for orientation
    orientation = CustomDense(256, activation=tf.nn.relu, apply_bn=True)(flattened)
    orientation = Dropout(0.5)(orientation)
    orientation = CustomDense(cfg.BIN * 2, activation=None, apply_bn=False)(orientation)
    orientation = tf.reshape(orientation, [-1, cfg.BIN, 2])
    orientation_out = tf.nn.l2_normalize(orientation, axis=2, name='orientation_out')

    # Computing the probabilities score
    prob_score = CustomDense(256, activation=tf.nn.relu, apply_bn=True)(flattened)
    prob_score = Dropout(0.5)(prob_score)
    prob_score = CustomDense(cfg.BIN, activation=None, name='prob_score', apply_bn=False)(prob_score)

    model = keras.Model(inputs=input_layer, outputs=[dimension_out, orientation_out, prob_score])
    return model


"""
d_label = tf.placeholder(tf.float32, shape = [None, 3])
o_label = tf.placeholder(tf.float32, shape = [None, BIN, 2])
c_label = tf.placeholder(tf.float32, shape = [None, BIN])


model = get_model(0.99)
model.summary()
"""