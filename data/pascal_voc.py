import os
import cv2
import torch
import numpy as np
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset


VOC_CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
               'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
               'train', 'tvmonitor')


class VOCAnnotationTransform(object):
    """This class converts the data from Annotation XML files to a list
    of [xmin, ymin, xmax, ymax, class]"""

    def __init__(self,
                 keep_difficult=False):
        """Class constructor for VOCAnnotationTransform

        Keyword Arguments:
            keep_difficult {bool} -- determines if the dataset will contain
            difficult examples or not (default: {False})
        """

        super(VOCAnnotationTransform, self).__init__()
        self.class_to_index = dict(zip(VOC_CLASSES, range(len(VOC_CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self,
                 target,
                 width,
                 height):
        """Executed when the class is called as a function

        Arguments:
            target {xml.etree.ElementTree.Element} -- annotation file
            width {int} -- Width of the corresponding image; Used for scaling
            the coordinates of bounding boxes
            height {int} -- Height of the corresponding image; Used for
            scaling the coordinates of bounding boxes

        Returns:
            list -- list of bounding boxes formatted as
            [xmin, ymin, xmax, ymax, class]
        """

        labels = []

        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')
            bndbox = [(int(bbox.find('xmin').text) - 1) / width,
                      (int(bbox.find('ymin').text) - 1) / height,
                      (int(bbox.find('xmax').text) - 1) / width,
                      (int(bbox.find('ymax').text) - 1) / height,
                      self.class_to_index[name]]
            labels += [bndbox]

        return labels


class PascalVOC(Dataset):
    """PascalVOC dataset

    Extends:
        Dataset
    """

    def __init__(self,
                 data_path,
                 image_sets,
                 new_size,
                 mode,
                 image_transform,
                 target_transform=VOCAnnotationTransform(),
                 keep_difficult=False):
        """Class constructor for PascalVOC

        Arguments:
            data_path {string} -- path to the dataset
            image_sets {tuple} -- contains the year and the subset - either
            trainval or test
            new_size {int} -- new height and width of the image
            mode {string} -- experiment mode - either train or test
            image_transform {object} -- produces different dataset
            augmentation techniques

        Keyword Arguments:
            target_transform {object} -- transforms data from Annotation XML
            files to a list of bounding boxes formatted as
            [xmin, ymin, xmax, ymax, class]
            (default: {VOCAnnotationTransform()})
            keep_difficult {bool} -- determines if the dataset will contain
            difficult examples (default: {False})
        """

        super(PascalVOC, self).__init__()

        self.data_path = data_path
        self.image_sets = image_sets
        self.new_size = new_size
        self.mode = mode
        self.image_transform = image_transform
        self.target_transform = target_transform
        self.keep_difficult = keep_difficult

        self.annotation_path = os.path.join('{}',
                                            '{}',
                                            'Annotations',
                                            '{}.xml')
        self.image_path = os.path.join('{}',
                                       '{}',
                                       'JPegImages',
                                       '{}.jpg')
        self.text_path = os.path.join('{}',
                                      '{}',
                                      'ImageSets',
                                      'Main',
                                      '{}.txt')

        self.ids = []
        for(year, name) in self.image_sets:
            version = 'VOC{}'.format(year)
            path = os.path.join(self.data_path, version)
            with open(self.text_path.format(path, name, name)) as f:
                for line in f:
                    self.ids.append((path, name, line.strip()))

    def __len__(self):
        """Returns the number of images in the dataset

        Returns:
            int -- number of images in the dataset
        """

        return len(self.ids)

    def __getitem__(self,
                    index):
        """Gets the image and its corresponding annotation found in
        position index of the list of images in the dataset

        Arguments:
            index {int} -- index of the image in the list of images in the
            dataset

        Returns:
            Tensor, list -- tensor representation of the image, list of
            bounding boxes of objects in the image formatted as
            [xmin, ymin, xmax, ymax, class]
        """

        image, target, _, _ = self.pull_item(index)
        return image, target

    def pull_item(self,
                  index):
        """Gets the image found in position index of the list of images in the
        dataset together with its corresponding annotation, its height, and its
        width

        Arguments:
            index {int} -- index of the image in the list of images in the
            dataset

        Returns:
            Tensor, list, int, int -- tensor representation of the image,
            list of bounding boxes of objects in the image formatted as
            [xmin, ymin, xmax, ymax, class], height, width
        """

        image_id = self.ids[index]

        target_path = self.annotation_path.format(image_id)
        image_path = self.image_path.format(image_id)

        target = ET.parse(target_path).getroot()
        image = cv2.imread(image_path)
        height, width, _ = image.shape

        if self.target_transform is not None:
            target = self.target_transform(target, width, height)

        if self.image_transform is not None:
            target = np.array(target)
            boxes = target[:, :4]
            labels = target[:, 4]
            image, boxes, labels = self.image_transform(image, boxes, labels)
            # to rgb
            image = image[:, :, (2, 1, 0)]
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))

        return torch.from_numpy(image).permute(2, 0, 1), target, height, width

    def pull_image(self,
                   index):
        """Gets the image found in position index of the list of images in the
        dataset

        Arguments:
            index {int} -- index of the image in the list of images in the
            dataset

        Returns:
            np.ndarray -- image
        """

        image_id = self.ids[index]
        image_path = self.image_path.format(image_id)
        return cv2.imread(image_path, cv2.IMREAD_COLOR)

    def pull_annotation(self,
                        index):
        """Gets the annotation of the image found in position index of the
        list of images in the dataset. The coordinates of the annotation is
        not scaled by the height and width of the image

        Arguments:
            index {int} -- index of the image in the list of images in the
            dataset

        Returns:
            string, list -- id of the image, list of bounding boxes of objects
            in the image formatted as [xmin, ymin, xmax, ymax, class]
        """

        image_id = self.ids[index]
        target_path = self.annotation_path.format(image_id)
        target = ET.parse(target_path).getroot()
        target = self.target_transform(target, 1, 1)

        return image_id[1], target

    def pull_tensor(self,
                    index):
        """Gets a tensor of the image found in position index of the list of
        images in the dataset

        Arguments:
            index {int} -- index of the image in the list of images in the
            dataset

        Returns:
            Tensor -- tensor representation of the image
        """

        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)
