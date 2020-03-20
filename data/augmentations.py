import cv2
import torch
import types
import numpy as np
from numpy import random


def intersect(box_a,
              box_b):

    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    intersection = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)

    return intersection[:, 0] * intersection[: 1]


def jaccard_numpy(box_a,
                  box_b):

    intersection = intersect(box_a, box_b)
    area_a = ((box_a[:, 2] - box_a[:, 0]) *
              (box_a[:, 3] - box_a[:, 1]))
    area_b = ((box_b[2] - box_b[0]) *
              (box_b[3] - box_b[1]))

    union = area_a + area_b - intersection

    return intersection / union


class Compose(object):

    def __init__(self,
                 transforms):

        super(Compose, self).__init__()
        self.transforms = transforms

    def __call__(self,
                 image,
                 boxes=None,
                 labels=None):

        for transform in self.transforms:
            image, boxes, labels = transform(image, boxes, labels)

        return image, boxes, labels


class Lambda(object):

    def __init__(self,
                 lambd):

        super(Lambda, self).__init__()
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self,
                 image,
                 boxes=None,
                 labels=None):

        return self.lambd(image, boxes, labels)


class ConvertToFloat(object):

    def __init__(self):

        super(ConvertToFloat, self).__init__()

    def __call__(self,
                 image,
                 boxes=None,
                 labels=None):

        return image.astype(np.float32), boxes, labels


class SubtractMeans(object):

    def __init__(self,
                 mean):

        super(SubtractMeans, self).__init__()
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self,
                 image,
                 boxes=None,
                 labels=None):

        image = image.astype(np.float32)
        image -= self.mean
        return image.astype(np.float32), boxes, labels


class ToAbsoluteCoords(object):

    def __init__(self):

        super(ToAbsoluteCoords, self).__init__()

    def __call__(self,
                 image,
                 boxes=None,
                 labels=None):

        height, width, channels = image.shape
        boxes[:, 0] *= width
        boxes[:, 2] *= width
        boxes[:, 1] *= height
        boxes[:, 3] *= height

        return image, boxes, labels


class ToPercentCoords(object):

    def __init__(self):

        super(ToPercentCoords, self).__init__()

    def __call__(self,
                 image,
                 boxes=None,
                 labels=None):

        height, width, channels = image.shape
        boxes[:, 0] /= width
        boxes[:, 2] /= width
        boxes[:, 1] /= height
        boxes[:, 3] /= height

        return image, boxes, labels


class Resize(object):

    def __init__(self,
                 size=300):

        super(Resize, self).__init__()
        self.size = size

    def __call__(self,
                 image,
                 boxes=None,
                 labels=None):

        image = cv2.resize(image, (self.size, self.size))
        return image, boxes, labels


class RandomSaturation(object):

    def __init__(self,
                 lower=0.5,
                 upper=1.5):

        super(RandomSaturation, self).__init__()
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, 'contrast upper must be >= lower.'
        assert self.lower >= 0, 'contrast lower must be non-negative.'

    def __call__(self,
                 image,
                 boxes=None,
                 labels=None):

        if random.randint(2):
            image[:, :, 1] *= random.uniform(self.lower, self.upper)

        return image, boxes, labels


class RandomHue(object):

    def __init__(self,
                 delta=18.0):

        super(RandomHue, self).__init__()
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self,
                 image,
                 boxes=None,
                 labels=None):

        if random.randint(2):
            image[:, :, 0] += random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0

        return image, boxes, labels


class RandomLightingNoise(object):

    def __init__(self):

        super(RandomLightingNoise, self).__init__()
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self,
                 image,
                 boxes=None,
                 labels=None):

        if random.randint(2):
            swap = self.perms[random.randint(len(self.perms))]
            shuffle = SwapChannels(swap)
            image = shuffle(image)

        return image, boxes, labels


class ConvertColor(object):

    def __init__(self,
                 current='BGR',
                 transform='HSV'):

        super(ConvertColor, self).__init__()
        self.current = current
        self.transform = transform

    def __call__(self,
                 image,
                 boxes=None,
                 labels=None):

        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError

        return image, boxes, labels


class RandomContrast(object):

    def __init__(self,
                 lower=0.5,
                 upper=1.5):

        super(RandomContrast, self).__init__()
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, 'contrast upper must be >= lower.'
        assert self.lower >= 0, 'contrast lower must be non-negative.'

    def __call__(self,
                 image,
                 boxes=None,
                 labels=None):

        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha

        return image, boxes, labels


class RandomBrightness(object):

    def __init__(self,
                 delta=32.0):

        super(RandomBrightness, self).__init__()
        assert delta >= 0.0 and delta <= 255.0
        self.delta = delta

    def __call__(self,
                 image,
                 boxes=None,
                 labels=None):

        if random.randint(2):
            delta = random.uniform(-self.data, self.delta)
            image += delta

        return image, boxes, labels


class ToCV2Image(object):

    def __init__(self):

        super(ToCV2Image, self).__init__()

    def __call__(self,
                 image,
                 boxes=None,
                 labels=None):

        image = image.cpu().numpy().astype(np.float32).transpose((1, 2, 0))
        return image, boxes, labels


class ToTensor(object):

    def __init__(self):

        super(ToTensor, self).__init__()

    def __call__(self,
                 image,
                 boxes=None,
                 labels=None):

        image = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1)
        return image, boxes, labels


class RandomSampleCrop(object):

    def __init__(self):

        super(RandomSampleCrop, self).__init__()
        self.sample_options = (None,
                               (0.1, None),
                               (0.3, None),
                               (0.7, None),
                               (0.9, None),
                               (None, None))

    def __call__(self,
                 image,
                 boxes=None,
                 labels=None):

        height, width, _ = image.shape

        while True:
            mode = random.choice(self.sample_options)

            if mode is None:
                return image, boxes, labels

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            for _ in range(50):
                current_image = image

                w = random.uniform(0.3 * width, width)
                h = random.uniform(0.3 * height, height)

                if h / w < 0.5 or h / w > 2:
                    continue

                left = random.uniform(width - w)
                top = random.uniform(height - h)

                x1 = int(left)
                y1 = int(top)
                x2 = int(left + w)
                y2 = int(top + h)
                crop = np.array([x1, y1, x2, y2])

                overlap = jaccard_numpy(boxes, crop)

                if overlap.min() < min_iou and max_iou < overlap.max():
                    continue

                current_image = current_image[y1:y2, x1:x2, :]
                centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0

                m1 = (x1 < centers[:, 0]) * (y1 < centers[:, 1])
                m2 = (x2 > centers[:, 0]) * (y2 > centers[:, 1])
                mask = m1 * m2

                if not mask.any():
                    continue

                current_boxes = boxes[mask, :].copy()
                current_labels = labels[mask]

                current_boxes[:, :2] = np.maximum(current_boxes[:, :2],
                                                  crop[:2])
                current_boxes[:, :2] -= crop[:2]

                current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:],
                                                  crop[2:])
                current_boxes[:, 2:] -= crop[:2]

                return current_image, current_boxes, current_labels


class Expand(object):

    def __init__(self,
                 mean):

        super(Expand, self).__init__()
        self.mean = mean

    def __call__(self,
                 image,
                 boxes,
                 labels):

        if random.randint(2):
            height, width, channels = image.shape
            ratio = random.uniform(1, 4)
            left = random.uniform(0, width * ratio - width)
            top = random.uniform(0, height * ratio - height)

            dimensions = (int(height * ratio), int(width * ratio), channels)
            expand_image = np.zeros(dimensions, dtype=image.dtype)
            expand_image[:, :, :] = self.mean

            x1 = int(left)
            x2 = int(left + width)
            y1 = int(top)
            y2 = int(top + height)
            expand_image[y1:y2, x1:x2] = image
            image = expand_image

            boxes = boxes.copy()
            boxes[:, :2] += (x1, y1)
            boxes[:, 2:] += (x1, y1)

        return image, boxes, labels


class RandomMirror(object):

    def __init__(self):

        super(RandomMirror, self).__init__()

    def __call__(self,
                 image,
                 boxes,
                 labels):

        _, width, _ = image.shape

        if random.randint(2):
            image = image[:, ::-1]
            boxes = boxes.copy()
            boxes[:, 0::2] = width - boxes[:, 2::-2]

        return image, boxes, labels


class SwapChannels(object):

    def __init__(self,
                 swaps):

        super(SwapChannels, self).__init__()
        self.swaps = swaps

    def __call__(self,
                 image):

        image = image[:, :, self.swaps]
        return image


class PhotometricDistort(object):

    def __init__(self):

        super(PhotometricDistort, self).__init__()
        self.pd = [RandomContrast(),
                   ConvertColor(transform='HSV'),
                   RandomSaturation(),
                   RandomHue(),
                   ConvertColor(current='HSV', transform='BGR'),
                   RandomContrast()]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()

    def __call__(self,
                 image,
                 boxes,
                 labels):

        image = image.copy()
        image, boxes, labels = self.rand_brightness(image, boxes, labels)
        if random.randint(2):
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        image, boxes, labels = distort(image, boxes, labels)

        return self.rand_light_noise(image, boxes, labels)


class Augmentations(object):

    def __init__(self,
                 size,
                 mean):

        super(Augmentations, self).__init__()
        self.size = size
        self.mean = mean
        self.augment = Compose([ConvertToFloat(),
                                ToAbsoluteCoords(),
                                PhotometricDistort(),
                                Expand(self.mean),
                                RandomSampleCrop(),
                                RandomMirror(),
                                ToPercentCoords(),
                                Resize(self.size),
                                SubtractMeans(self.mean)])

        def __call__(self,
                     image,
                     boxes,
                     labels):

            return self.augment(image, boxes, labels)


class BaseTransform(object):

    def __init__(self,
                 size,
                 mean):

        super(BaseTransform, self).__init__()
        self.size = size
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self,
                 image,
                 boxes=None,
                 labels=None):

        dimensions = (self.size, self.size)
        image = cv2.resize(image, dimensions).astype(np.float32)
        image -= self.mean
        image = image.astype(np.float32)
        return image, boxes, labels
