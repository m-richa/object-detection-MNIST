import numpy as np
import tensorflow as tf
import bounding_box
import matplotlib.pyplot as plt
from preprocessing import get_training_dataset, get_validation_dataset
from visualization_utils import dataset_to_numpy_util, display_digits_with_boxes
from model import define_and_compile_model, build_model

# Strategy

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
except ValueError:
    tpu = None
    gpus = tf.config.experimental.list_logical_devices("GPU")

# Select appropriate distribution strategy
if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(
        tpu)  # Going back and forth between TPU and host is expensive. Better to run 128 batches on the TPU before reporting back.
    print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])
elif len(gpus) > 1:
    strategy = tf.distribute.MirroredStrategy([gpu.name for gpu in gpus])
    print('Running on multiple GPUs ', [gpu.name for gpu in gpus])
elif len(gpus) == 1:
    strategy = tf.distribute.get_strategy()  # default strategy that works on CPU and single GPU
    print('Running on single GPU ', gpus[0].name)
else:
    strategy = tf.distribute.get_strategy()  # default strategy that works on CPU and single GPU
    print('Running on CPU')
print("Number of accelerators: ", strategy.num_replicas_in_sync)

BATCH_SIZE = 64 * strategy.num_replicas_in_sync

with strategy.scope():
    training_dataset = get_training_dataset(strategy, BATCH_SIZE)
    validation_dataset = get_validation_dataset()

# Visualize dataset

(training_digits, training_labels, training_bboxes,
 validation_digits, validation_labels, validation_bboxes) = dataset_to_numpy_util(training_dataset, validation_dataset,
                                                                                  10)

plt.rc('image', cmap='gray')
plt.rc('grid', linewidth=0)
plt.rc('xtick', top=False, bottom=False, labelsize='large')
plt.rc('ytick', left=False, right=False, labelsize='large')
plt.rc('axes', facecolor='F8F8F8', titlesize="large", edgecolor='white')
plt.rc('text', color='a8151a')
plt.rc('figure', facecolor='F0F0F0')  # Matplotlib fonts

display_digits_with_boxes(training_digits, training_labels, training_labels, np.array([]), training_bboxes,
                          np.array([]), "training digits and their labels", plt)
display_digits_with_boxes(validation_digits, validation_labels, validation_labels, np.array([]), validation_bboxes,
                          np.array([]), "validation digits and their labels", plt)

EPOCHS = 10
steps_per_epoch = 60000 // BATCH_SIZE
validation_step = 1

# model = define_and_compile_model(training_dataset)

with strategy.scope():
    model = build_model(strategy=strategy)

model.summary()

history = model.fit(training_dataset,
                    steps_per_epoch=steps_per_epoch,
                    validation_data=validation_dataset,
                    validation_steps=validation_step,
                    epochs=EPOCHS)

loss, classification_loss, bounding_box, classification_accuracy, bounding_box_mse = model.evaluate(validation_dataset,
                                                                                                    steps=1)
print("Validation accuracy: ", classification_accuracy)

# plot_metrics("classification_loss", "Classification Loss")
# plot_metrics("bounding_box_loss", "Bounding Box Loss")
