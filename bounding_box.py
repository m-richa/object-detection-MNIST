import PIL.Image, PIL.ImageFont, PIL.ImageDraw
import numpy as np

IMG_WIDTH = 75
IMG_HEIGHT = 75
use_normalized_coordinates = True


def draw_boxes_on_img_array(image, boxes, color=[], thickness=1):
    image_pil = PIL.Image.fromarray(image)
    rgbimg = PIL.Image.new("RGBA", image_pil.size)
    rgbimg.paste(image_pil)

    draw_boxes_on_image(rgbimg, boxes, color, thickness)

    return np.array(rgbimg)


def draw_boxes_on_image(image, boxes, color=[], thickness=1):

    boxes_shape = boxes.shape
    if not boxes_shape:
        return
    if len(boxes_shape) != 2 or boxes_shape[1] != 4:
        raise ValueError('Input must be (N, 4)')

    for i in range(boxes_shape[0]):

        draw_box_on_image(image, boxes[i,1], boxes[i,0], boxes[i,3], boxes[i,2], color[i],
                          thickness)


def draw_box_on_image(image,
                      ymin,
                      xmin,
                      ymax,
                      xmax,
                      color='red',
                      thickness=1):

    draw = PIL.ImageDraw.Draw(image)
    im_width, im_height = image.size

    (left, right, top, bottom) = (xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)

    draw.line([(left, top), (left, bottom), (right, top), (right, bottom)],
              width=thickness,
              fill=color)
