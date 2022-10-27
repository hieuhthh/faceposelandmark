from dataset import *
from model import *
from utils import *
from callback import *

import os
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, SGD, Optimizer

os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"

set_memory_growth()

BATCH_SIZE = 256
im_size = 192
img_size = (im_size, im_size)
input_shape = (im_size, im_size, 3)
base_name = 'EfficientNetV1B1'
final_dropout = 0.3
n_landmark = 68

make_augment = True
rot_angle = 30
save_sample = 10

train_with_labels = True
train_repeat = True
train_shuffle = 512
train_augment = False

valid_with_labels = True
valid_repeat = False
valid_shuffle = False
valid_augment = False

monitor = "val_loss"
mode = 'min'

max_lr = 1e-3
min_lr = 1e-5
cycle_epoch = 20
n_cycle = 5
epochs = cycle_epoch * n_cycle

print('epochs', epochs)

SEED = 1024
seedEverything(SEED)

image_list, masks_list = get_images_marks(im_size, rot_angle=rot_angle, 
                                          save_sample=save_sample, make_augment=make_augment)

valid_size = 0.01
image_list_train, image_list_valid, masks_list_train, masks_list_valid = train_test_split(
                  image_list, masks_list, test_size=valid_size, random_state=SEED, shuffle=True)

train_n_images = len(image_list_train)
valid_n_images = len(image_list_valid)

train_decoder = build_decoder(with_labels=True, target_size=img_size)

train_dataset = build_dataset(image_list_train, masks_list_train, bsize=BATCH_SIZE, decode_fn=train_decoder,
                              repeat=train_repeat, shuffle=train_shuffle, augment=train_augment)

valid_decoder = build_decoder(with_labels=True, target_size=img_size)

valid_dataset = build_dataset(image_list_valid, masks_list_valid, bsize=BATCH_SIZE, decode_fn=valid_decoder,
                              repeat=valid_repeat, shuffle=valid_shuffle, augment=valid_augment)

strategy = auto_select_accelerator()

print('n_landmark', n_landmark)
print('train_n_images', train_n_images)
print('valid_n_images', valid_n_images)

tf.keras.backend.clear_session()
tf.compat.v1.reset_default_graph()

with strategy.scope():
    base = get_base_model(base_name, input_shape)
    model = create_model(input_shape, base, n_landmark, final_dropout)
    model.summary()

with strategy.scope():
    losses = tf.keras.losses.MeanSquaredError()
    metrics = [tf.keras.metrics.MeanSquaredError()]

    model.compile(optimizer=Adam(learning_rate=1e-3),
                loss=losses,
                metrics=metrics)

save_path = f'best_model_face_pose_esti_{base_name}_{im_size}_{n_landmark}.h5'

callbacks = get_callbacks(monitor, mode, save_path, max_lr, min_lr, cycle_epoch)

his = model.fit(train_dataset, 
                validation_data=valid_dataset,
                steps_per_epoch = train_n_images//BATCH_SIZE,
                epochs=epochs,
                verbose=1,
                callbacks=callbacks)

metric = 'loss'
visual_save_metric(his, metric)

metric = 'mean_squared_error'
visual_save_metric(his, metric)