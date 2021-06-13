import tensorflow as tf
import tensorflow_datasets as tfds

IMG_SIZE = 28
INP_IMG_SIZE = 75


def read_image_tfds(image, label):
    xmin = tf.random.uniform((), 0, 48, dtype=tf.int32)
    ymin = tf.random.uniform((), 0, 48, dtype=tf.int32)
    image = tf.reshape(image, (IMG_SIZE, IMG_SIZE, 1))
    image = tf.image.pad_to_bounding_box(image, xmin, ymin, INP_IMG_SIZE, INP_IMG_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    xmin = tf.cast(xmin, tf.float32)
    ymin = tf.cast(ymin, tf.float32)

    # Normalize the bounding box co-ordinates
    xmax = (xmin + IMG_SIZE) / INP_IMG_SIZE
    ymax = (ymin + IMG_SIZE) / INP_IMG_SIZE
    xmin = xmin / INP_IMG_SIZE
    ymin = ymin / INP_IMG_SIZE

    return image, (tf.one_hot(label, 10), [xmin, ymin, xmax, ymax])


def get_training_dataset(strategy, batch_size):

    with strategy.scope():
        dataset = tfds.load("mnist", split="train", as_supervised=True, try_gcs=True)
        dataset = dataset.map(read_image_tfds, num_parallel_calls=16)
        dataset = dataset.shuffle(5000, reshuffle_each_iteration=True)
        dataset = dataset.repeat()
        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.prefetch(-1)
        print('Training dataset fetched')
    return dataset


def get_validation_dataset():
    dataset = tfds.load('mnist', split="test", as_supervised=True, try_gcs=True)
    dataset = dataset.map(read_image_tfds, num_parallel_calls=16)

    dataset = dataset.batch(10000, drop_remainder=True)
    dataset = dataset.repeat()
    print('Validation dataset fetched')
    return dataset
