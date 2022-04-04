import os
import argparse
import tensorflow as tf
from model import get_model
import time
from tensorflow.keras.utils import plot_model
from tensorflow import keras
import numpy as np
from data_loader import data_generator
from config import cfg

tf.random.set_seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='test', help='train or test')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--max_epoch', type=int, default=50, help='Epoch to run [default: 250]')
parser.add_argument('--batch_size', type=int, default=4, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.99, help='Initial learning rate [default: 0.9]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.8, help='Decay rate for lr decay [default: 0.8]')

FLAGS = parser.parse_args()
MODE = FLAGS.mode
BATCH_SIZE = FLAGS.batch_size
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
MOMENTUM = FLAGS.momentum
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

log_dir = FLAGS.log_dir
if not os.path.exists(log_dir): os.mkdir(log_dir)


class OrientationLoss(keras.losses.Loss):
    def __init__(self, regularization_factor=0.1, name="custom_orientation_loss"):
        super().__init__(name=name)
        self.regularization_factor = regularization_factor

    def call(self, y_true, y_pred):
        anchors = tf.math.reduce_sum(tf.square(y_true), axis=2)
        anchors = tf.math.greater(tf.cast(anchors, tf.float32), tf.constant(0.5, dtype=tf.float32))
        anchors = tf.math.reduce_sum(tf.cast(anchors, tf.float32), 1)

        # Define the loss
        loss = (y_true[:, :, 0] * y_pred[:, :, 0] + y_true[:, :, 1] * y_pred[:, :, 1])
        loss = tf.reduce_sum((2 - 2 * tf.reduce_mean(loss, axis=0))) / anchors

        return tf.math.reduce_mean(loss)


# Initialize the metrics
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
validation_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

# Initialize the loss function
confidence_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
dimension_loss = tf.keras.losses.MeanSquaredError()
orientation_loss = OrientationLoss()
# Initialize the callback
callback = tf.keras.callbacks.EarlyStopping(verbose=1, patience=2)


# Initialize the batch normalization
def get_bn_momentum(step):
    return min(0.99, 0.5 + 0.0002 * step)


bn_momentum = tf.Variable(get_bn_momentum(0), trainable=False)


# Instantiate optimizer and loss function
def get_lr(initial_learning_rate, decay_steps, decay_rate, step, staircase=False, warm_up=True):
    if warm_up:
        coeff1 = min(1.0, step / 2000)
    else:
        coeff1 = 1.0

    if staircase:
        coeff2 = decay_rate ** (step // decay_steps)
    else:
        coeff2 = decay_rate ** (step / decay_steps)

    current = initial_learning_rate * coeff1 * coeff2
    return current


LR_ARGS = {'initial_learning_rate': BASE_LEARNING_RATE, 'decay_steps': DECAY_STEP,
           'decay_rate': DECAY_RATE, 'staircase': False, 'warm_up': True}
lr = tf.Variable(get_lr(**LR_ARGS, step=0), trainable=False)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)


@tf.function
def train_step(inputs, labels):
    with tf.GradientTape() as tape:
        dim_logits, orientation_logits, confidence_logits = model(inputs, training=True)
        dim_loss = dimension_loss(labels[0], dim_logits)
        orient_loss = orientation_loss(labels[1], orientation_logits)
        conf_loss = confidence_loss(labels[2], confidence_logits)

        total_loss = 4. * dim_loss + 8. * orient_loss + conf_loss
    gradients = tape.gradient(total_loss, model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))
    return dim_logits, orientation_logits, confidence_logits, dim_loss, orient_loss, conf_loss, total_loss


@tf.function
def validation_step(inputs, labels):
    dim_logits, orientation_logits, confidence_logits = model(inputs, training=False)
    dim_loss = dimension_loss(labels[0], dim_logits)
    orient_loss = orientation_loss(labels[1], orientation_logits)
    conf_loss = confidence_loss(labels[2], confidence_logits)

    total_loss = 4. * dim_loss + 8. * orient_loss + conf_loss
    return dim_logits, orientation_logits, confidence_logits, dim_loss, orient_loss, conf_loss, total_loss


model = get_model(bn_momentum)
print(model.summary())
plot_model(model, to_file='model.png', show_shapes=True)

# Model training
start_time = time.time()
print('Model training started : \n')
# print('Steps per epoch : {}'.format(train_length // BATCH_SIZE))
# print('Total Number of steps while training : {}'.format((train_length // BATCH_SIZE) * MAX_EPOCH))
writer = tf.summary.create_file_writer(log_dir)
step = 0
for epoch in range(MAX_EPOCH):
    print("Epoch started : {}".format(epoch))
    train_accuracy.reset_states()
    validation_accuracy.reset_states()
    total_losses = []
    dim_losses = []
    orient_losses = []
    conf_losses = []
    total_losses = []
    for img_batch, dim_batch, orientation_batch, confidence_batch in data_generator(cfg.TRAIN_DIR, shuffle=True,
                                                                                    isTest=False, batch_size=3):
        dim_logits, orientation_logits, confidence_logits, dim_loss, orient_loss, conf_loss, total_loss = train_step(
            img_batch, [dim_batch, orientation_batch, confidence_batch])
        # train_accuracy.update_state(y_train, logits)
        total_losses.append(total_loss)
        dim_losses.append(dim_loss)
        orient_losses.append(orient_loss)
        conf_losses.append(conf_loss)

        step += 1
        bn_momentum.assign(get_bn_momentum(step))
        lr.assign(get_lr(**LR_ARGS, step=step))
    train_total_loss = np.mean(total_loss)
    train_dim_losses = np.mean(dim_losses)
    train_orient_losses = np.mean(orient_losses)
    train_conf_losses = np.mean(conf_losses)

    total_losses = []
    dim_losses = []
    orient_losses = []
    conf_losses = []
    total_losses = []
    for val_img_batch, val_dim_batch, val_orientation_batch, val_confidence_batch in data_generator(cfg.TRAIN_DIR,
                                                                                                    shuffle=True,
                                                                                                    isTest=False,
                                                                                                    batch_size=3):
        val_dim_logits, val_orientation_logits, val_confidence_logits, val_dim_loss, val_orient_loss, val_conf_loss, val_total_loss = validation_step(
            val_img_batch, [val_dim_batch, val_orientation_batch, val_confidence_batch])
        total_losses.append(val_total_loss)
        dim_losses.append(val_dim_loss)
        orient_losses.append(val_orient_loss)
        conf_losses.append(val_conf_loss)
        # validation_accuracy.update_state(y_val, val_logits)
    val_total_loss = np.mean(total_loss)
    val_dim_losses = np.mean(dim_losses)
    val_orient_losses = np.mean(orient_losses)
    val_conf_losses = np.mean(conf_losses)
    with writer.as_default():
        tf.summary.scalar("training_total_loss", train_total_loss, step=epoch)
        tf.summary.scalar("train_dim_losses", train_dim_losses, step=epoch)
        tf.summary.scalar("train_orient_losses", train_orient_losses, step=epoch)
        tf.summary.scalar("train_conf_losses", train_conf_losses, step=epoch)

        tf.summary.scalar("validation_loss", val_total_loss, step=epoch)
        tf.summary.scalar("val_dim_losses ", val_dim_losses, step=epoch)
        tf.summary.scalar("val_orient_losses ", val_orient_losses, step=epoch)
        tf.summary.scalar("val_conf_losses ", val_conf_losses, step=epoch)

        tf.summary.scalar("learning_rate", lr.numpy(), step=epoch)
        tf.summary.scalar("batch_normalization", bn_momentum.numpy(), step=epoch)
    writer.flush()
    model.save_weights('model/checkpoints/' + 'iter-' + str(epoch), save_format='tf')

end_time = time.time()
time_lapsed = end_time - start_time
print('Training Completed')
print('Total time lapsed in seconds {}'.format(time_lapsed))
print('Total time lapsed in minutes {}'.format(time_lapsed / 60.))
print('Total time lapsed in hrs {}'.format(time_lapsed / (60 * 60)))
