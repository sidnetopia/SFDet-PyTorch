import cv2
import os
import torch


def to_var(x, use_gpu, requires_grad=False):
    if torch.cuda.is_available() and use_gpu:
        x = x.cuda()
    x.requires_grad = requires_grad
    return x


def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def write_print(path, text):
    """Displays text in console and saves in text file

    Arguments:
        path {string} -- path to text file
        text {string} -- text to display and save
    """
    file = open(path, 'a')
    file.write(text + '\n')
    file.close()
    print(text)


def draw_bbox(image,
              start_point,
              end_point,
              color,
              thickness):

    image = cv2.rectangle(image,
                          start_point,
                          end_point,
                          color,
                          thickness)
    return image


def draw_labels(image,
                labels,
                dictionary,
                is_prediction=False):

    if is_prediction:
        thickness = 1
    else:
        thickness = 3

    for label in labels:
        start_point = (int(label[0]), int(label[1]))
        end_point = (int(label[2]), int(label[3]))

        image = cv2.rectangle(image,
                              start_point,
                              end_point,
                              (255, 255, 255),
                              thickness)

        text = label[4]

        if not is_prediction:
            text = dictionary[int(label[4])]

        image = cv2.putText(image,
                            text,
                            (int(label[0]), int(label[1] - 10)),
                            cv2.FONT_HERSHEY_DUPLEX,
                            0.7,
                            color=(255, 255, 255),
                            thickness=2)

    return image
