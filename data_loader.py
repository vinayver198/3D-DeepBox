from glob import glob
import os
import numpy as np
import cv2
import multiprocessing
from config import cfg
import copy


def read_image(filename):
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def compute_anchors(angle):
    anchors = []

    wedge = 2. * np.pi / cfg.BIN
    l_index = int(angle / wedge)
    r_index = l_index + 1

    if (angle - l_index * wedge) < wedge / 2 * (1 + cfg.OVERLAP / 2):
        anchors.append([l_index, angle - l_index * wedge])

    if (r_index * wedge - angle) < wedge / 2 * (1 + cfg.OVERLAP / 2):
        anchors.append([r_index % cfg.BIN, angle - r_index * wedge])

    return anchors


def parse_labels(labels):
    all_objs = []
    for label_file in labels:
        for line in open(label_file).readlines():
            line = line.strip().split(' ')
            truncated = np.abs(float(line[1]))
            occluded = np.abs(float(line[2]))

            if line[0] in cfg.CLASSES and truncated < 0.1 and occluded < 0.1:
                new_alpha = float(line[3]) + np.pi / 2.
                if new_alpha < 0:
                    new_alpha = new_alpha + 2. * np.pi
                new_alpha = new_alpha - int(new_alpha / (2. * np.pi)) * (2. * np.pi)
                image_file = label_file.replace('txt', 'png')
                image_file = image_file.replace('label_2', 'image_2')
                obj = {'name': line[0],
                       'image': image_file,
                       'xmin': int(float(line[4])),
                       'ymin': int(float(line[5])),
                       'xmax': int(float(line[6])),
                       'ymax': int(float(line[7])),
                       'dims': np.array([float(number) for number in line[8:11]]),
                       'new_alpha': new_alpha
                       }
                all_objs.append(obj)
    return all_objs


class Preprocessor:
    def __init__(self, all_objs, data_dir, isTest):
        self.all_objs = all_objs
        self.data_dir = data_dir
        self.isTest = isTest

    def __call__(self, indices):
        imgs = []
        dims = []
        orients = []
        confs = []
        for index in indices:
            flip = np.random.binomial(1, .5)
            if not self.isTest:
                obj = self.all_objs[index]
                obj['dims'] = obj['dims'] - cfg.CLASSES_AVG[obj['name']]
                # Fix orientation and confidence for no flip
                orientation = np.zeros((cfg.BIN, 2))
                confidence = np.zeros(cfg.BIN)

                anchors = compute_anchors(obj['new_alpha'])

                for anchor in anchors:
                    orientation[anchor[0]] = np.array([np.cos(anchor[1]), np.sin(anchor[1])])
                    confidence[anchor[0]] = 1.

                confidence = confidence / np.sum(confidence)
                # print(f'confidence : {len(confidence)}')

                obj['orient'] = orientation
                obj['conf'] = confidence[0]

                # Fix orientation and confidence for flip
                orientation = np.zeros((cfg.BIN, 2))
                confidence = np.zeros(cfg.BIN)
                anchors = compute_anchors(2. * np.pi - obj['new_alpha'])
                for anchor in anchors:
                    orientation[anchor[0]] = np.array([np.cos(anchor[1]), np.sin(anchor[1])])
                    confidence[anchor[0]] = 1

                confidence = confidence / np.sum(confidence)
                obj['orient_flipped'] = orientation
                obj['conf_flipped'] = confidence[0]

                # Prepare image patch
                xmin = obj['xmin']
                ymin = obj['ymin']
                xmax = obj['xmax']
                ymax = obj['ymax']
                img = read_image(obj['image'])
                img = copy.deepcopy(img[ymin:ymax + 1, xmin:xmax + 1]).astype(np.float32)
                if flip > 0.5:
                    img = cv2.flip(img, 1)
                img = cv2.resize(img, (cfg.IMAGE_WIDTH, cfg.IMAGE_HEIGHT))
                img = img - np.array([[[103.939, 116.779, 123.68]]])

                if flip > 0.5:
                    imgs.append(img)
                    dims.append(obj['dims'])
                    orients.append(obj['orient_flipped'])
                    confs.append(obj['conf_flipped'])

                    # return [img, obj['dims'], obj['orient_flipped'], obj['conf_flipped']]
                else:
                    imgs.append(img)
                    dims.append(obj['dims'])
                    orients.append(obj['orient_flipped'])
                    confs.append(obj['conf_flipped'])
        return imgs, dims, orients, confs


TRAIN_POOL = multiprocessing.Pool(4)
VAL_POOL = multiprocessing.Pool(2)


def data_generator(data_dir, shuffle=False, isTest=False, batch_size=1):
    labels = glob(os.path.join(data_dir, 'label_2', '*.txt'))
    labels.sort()

    all_objs = parse_labels(labels)
    nums = len(all_objs)

    indices = list(range(nums))
    if shuffle:
        np.random.shuffle(indices)

    num_batches = int(np.floor(nums / float(batch_size)))

    processor = Preprocessor(all_objs, data_dir, isTest)

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        index_section = indices[start_idx:start_idx + batch_size]
        img_batch, dim_batch, orientation_batch, confidence_batch = processor(index_section)

        yield np.array(img_batch).astype(np.float32), np.array(dim_batch).astype(np.float32), np.array(orientation_batch).astype(np.float32), np.array(confidence_batch).astype(np.float32)


"""
for batch in data_generator(cfg.TRAIN_DIR, shuffle=True, isTest=False, batch_size=2):
    img_batch, dim_batch, orientation_batch, confidence_batch = batch
    print(img_batch.shape)
    print(dim_batch.shape)
    print(orientation_batch.shape)
    print(confidence_batch)
    break
"""
