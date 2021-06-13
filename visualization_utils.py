import numpy as np
import tensorflow as tf
from bounding_box import draw_boxes_on_img_array
import PIL.Image, PIL.ImageFont, PIL.ImageDraw
import os

MATPLOTLIB_FONT_DIR = '/home/richa/PycharmProjects/object-detection-MNIST'
iou_threshold = 0.5


# pull a batch from the datasets. This code is not very nice, it gets much better in eager mode (TODO)


def dataset_to_numpy_util(training_dataset, validation_dataset, N):
    # get one batch from each: 10000 validation digits, N training digits
    batch_train_ds = training_dataset.unbatch().batch(N)

    # eager execution: loop through datasets normally
    if tf.executing_eagerly():
        for validation_digits, (validation_labels, validation_bboxes) in validation_dataset:
            validation_digits = validation_digits.numpy()
            validation_labels = validation_labels.numpy()
            validation_bboxes = validation_bboxes.numpy()
            break
        for training_digits, (training_labels, training_bboxes) in batch_train_ds:
            training_digits = training_digits.numpy()
            training_labels = training_labels.numpy()
            training_bboxes = training_bboxes.numpy()
            break

    # these were one-hot encoded in the dataset
    validation_labels = np.argmax(validation_labels, axis=1)
    training_labels = np.argmax(training_labels, axis=1)

    return (training_digits, training_labels, training_bboxes,
            validation_digits, validation_labels, validation_bboxes)


''  # create digits from local fonts for testing


def create_digits_from_local_fonts(n):
    font_labels = []
    img = PIL.Image.new('LA', (75 * n, 75), color=(0, 255))  # format 'LA': black in channel 0, alpha in channel 1
    font1 = PIL.ImageFont.truetype(os.path.join(MATPLOTLIB_FONT_DIR, 'DejaVuSansMono-Oblique.ttf'), 25)
    font2 = PIL.ImageFont.truetype(os.path.join(MATPLOTLIB_FONT_DIR, 'STIXGeneral.ttf'), 25)
    d = PIL.ImageDraw.Draw(img)
    for i in range(n):
        font_labels.append(i % 10)
        d.text((7 + i * 75, 0 if i < 10 else -4), str(i % 10), fill=(255, 255), font=font1 if i < 10 else font2)
    font_digits = np.array(img.getdata(), np.float32)[:,
                  0] / 255.0  # black in channel 0, alpha in channel 1 (discarded)
    font_digits = np.reshape(np.stack(np.split(np.reshape(font_digits, [75, 75 * n]), n, axis=1), axis=0), [n, 75 * 75])
    return font_digits, font_labels


# utility to display a row of digits with their predictions

def display_digits_with_boxes(digits, predictions, labels, pred_bboxes, bboxes, iou, title, plt):
    n = 10

    indexes = np.random.choice(len(predictions), size=n)
    n_digits = digits[indexes]
    n_predictions = predictions[indexes]
    n_labels = labels[indexes]

    n_iou = []
    if len(iou) > 0:
        n_iou = iou[indexes]

    if len(pred_bboxes) > 0:
        n_pred_bboxes = pred_bboxes[indexes, :]

    if len(bboxes) > 0:
        n_bboxes = bboxes[indexes, :]

    n_digits = n_digits * 255.0
    n_digits = n_digits.reshape(n, 75, 75)
    fig = plt.figure(figsize=(20, 4))
    plt.title(title)
    plt.yticks([])
    plt.xticks([])

    for i in range(10):
        ax = fig.add_subplot(1, 10, i + 1)
        bboxes_to_plot = []
        if len(pred_bboxes) > i:
            bboxes_to_plot.append(n_pred_bboxes[i])

        if len(bboxes) > i:
            bboxes_to_plot.append(n_bboxes[i])

        img_to_draw = draw_boxes_on_img_array(image=n_digits[i], boxes=np.asarray(bboxes_to_plot),
                                              color=['red', 'green'])
        plt.xlabel(n_predictions[i])
        plt.xticks([])
        plt.yticks([])

        if n_predictions[i] != n_labels[i]:
            ax.xaxis.label.set_color('red')

        plt.imshow(img_to_draw)

        if len(iou) > i:
            color = "black"
            if (n_iou[i][0] < iou_threshold):
                color = "red"
            ax.text(0.2, -0.3, "iou: %s" % (n_iou[i][0]), color=color, transform=ax.transAxes)


'''# utility to display training and validation curves
def plot_metrics(metric_name, title, ylim=5, plt):
  plt.title(title)
  plt.ylim(0,ylim)
  plt.plot(history.history[metric_name],color='blue',label=metric_name)
  plt.plot(history.history['val_' + metric_name],color='green',label='val_' + metric_name)'''
