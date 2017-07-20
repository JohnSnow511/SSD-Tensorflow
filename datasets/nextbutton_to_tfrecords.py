# Copyright 2017 xhjia. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Converts Next Button data to TFRecords file format with Example protos.

The raw Next Button data set is expected to reside in PNG files located in the
directory 'Images_HEGIHT_WIDTH', HEIGHT, WIDTH refer to the Integer, and represent
image's height and width. Similarly, bounding box annotations are supposed to be
stored in the 'Annotation directory'

This TensorFlow script converts the training and evaluation data into
a sharded data set consisting of 1024 and 128 TFRecord files, respectively.

Each validation TFRecord file contains ~500 records. Each training TFRecord
file contains ~1000 records. Each record within the TFRecord file is a
serialized Example proto. The Example proto contains the following fields:

    image/encoded: string containing PNG encoded image in RGB colorspace
    image/height: integer, image height in pixels
    image/width: integer, image width in pixels
    image/channels: integer, specifying the number of channels, always 3
    image/format: string, specifying the format, always 'PNG'


    image/object/bbox/xmin: list of float specifying the 0+ human annotated
        bounding boxes
    image/object/bbox/xmax: list of float specifying the 0+ human annotated
        bounding boxes
    image/object/bbox/ymin: list of float specifying the 0+ human annotated
        bounding boxes
    image/object/bbox/ymax: list of float specifying the 0+ human annotated
        bounding boxes
    image/object/bbox/label: list of integer specifying the classification index.
    image/object/bbox/label_text: list of string descriptions.

Note that the length of xmin is identical to the length of xmax, ymin and ymax
for each example.
"""
import os
import sys
import random

import numpy as np
import tensorflow as tf

from datasets.dataset_utils import int64_feature, float_feature, bytes_feature
from datasets.nextbutton_common import NEXT_LABELS
from datasets.nextbutton_common import IMG_DIRS

_ORG_IMG_WIDTH = 1440
_ORG_IMG_HEIGHT = 2560
_ORG_DEPTH = 3
_RESIZE_RATIO = 0.1
_LABEL = 'next'

# Original dataset organisation.
_DIRECTORY_ANNOTATIONS = 'Annotations/'
_DIRECTORY_IMAGES = IMG_DIRS[str(_RESIZE_RATIO)]

# TFRecords convertion parameters.
_RANDOM_SEED = 4242
_SAMPLES_PER_FILES = 500


def _process_image(directory, name):
    """Process a image and annotation file.

    Args:
      filename: string, path to an image file e.g., '/path/to/example.JPG'.
      coder: instance of ImageCoder to provide TensorFlow image coding utils.
    Returns:
      image_buffer: string, JPEG encoding of RGB image.
      height: integer, image height in pixels.
      width: integer, image width in pixels.
    """
    # Read the image file.
    filename = os.path.join(directory, _DIRECTORY_IMAGES, name + '.png')
    image_data = tf.gfile.FastGFile(filename, 'rb').read()

    # Image shape.
    shape = [int(_ORG_IMG_HEIGHT * _RESIZE_RATIO),
             int(_ORG_IMG_WIDTH * _RESIZE_RATIO),
             int(_ORG_DEPTH)]

    bboxes = []
    labels = []
    labels_text = []
    # Read the TXT annotation file.
    filename = os.path.join(directory, _DIRECTORY_ANNOTATIONS, name + '.txt')
    # In current setting, bb return a np.array with shape (4,) x_min, y_min, x_max, y_max
    bb = np.loadtxt(filename, dtype=np.float32, delimiter= '\n')
    bb = bb * _RESIZE_RATIO
    labels.append(int(NEXT_LABELS[_LABEL][0]))
    labels_text.append(_LABEL.encode('ascii'))
    # ymin, xmin, ymax, xmax
    bboxes.append((float(bb[1]) / shape[0],
                   float(bb[0]) / shape[1],
                   float(bb[3]) / shape[0],
                   float(bb[2]) / shape[1]))
    return image_data, shape, bboxes, labels, labels_text


def _convert_to_example(image_data, labels, labels_text, bboxes, shape):
    """Build an Example proto for an image example.

    Args:
      image_data: string, PNG encoding of RGB image;
      labels: list of integers, identifier for the ground truth;
      labels_text: list of strings, human-readable labels;
      bboxes: list of bounding boxes; each box is a list of integers;
          specifying [xmin, ymin, xmax, ymax]. All boxes are assumed to belong
          to the same label as the image label.
      shape: 3 integers, image shapes in pixels.
    Returns:
      Example proto
    """
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    for b in bboxes:
        assert len(b) == 4
        # pylint: disable=expression-not-assigned
        [l.append(point) for l, point in zip([ymin, xmin, ymax, xmax], b)]
        # pylint: enable=expression-not-assigned

    image_format = b'PNG'
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': int64_feature(shape[0]),
        'image/width': int64_feature(shape[1]),
        'image/channels': int64_feature(shape[2]),
        'image/shape': int64_feature(shape),
        'image/object/bbox/xmin': float_feature(xmin),
        'image/object/bbox/xmax': float_feature(xmax),
        'image/object/bbox/ymin': float_feature(ymin),
        'image/object/bbox/ymax': float_feature(ymax),
        'image/object/bbox/label': int64_feature(labels),
        'image/object/bbox/label_text': bytes_feature(labels_text),
        'image/format': bytes_feature(image_format),
        'image/encoded': bytes_feature(image_data)}))
    return example


def _add_to_tfrecord(dataset_dir, name, tfrecord_writer):
    """Loads data from image and annotations files and add them to a TFRecord.

    Args:
      dataset_dir: Dataset directory;
      name: Image name to add to the TFRecord;
      tfrecord_writer: The TFRecord writer to use for writing.
    """
    image_data, shape, bboxes, labels, labels_text = \
        _process_image(dataset_dir, name)
    example = _convert_to_example(image_data, labels, labels_text,
                                  bboxes, shape)
    tfrecord_writer.write(example.SerializeToString())


def _get_output_filename(output_dir, name, idx):
    return '%s/%s_%04d.tfrecords' % (output_dir, name, idx)


def run(dataset_dir, output_dir, name='nextbutton_train', shuffling=False):
    """Runs the conversion operation.

    Args:
      dataset_dir: The dataset directory where the dataset is stored.
      output_dir: Output directory.
    """

    if not tf.gfile.Exists(dataset_dir):
        tf.gfile.MakeDirs(dataset_dir)

    # Dataset filenames, and shuffling.
    print('current path is: %s' % os.getcwd())
    path = os.path.join(dataset_dir, _DIRECTORY_ANNOTATIONS)
    filenames = sorted(os.listdir(path))
    if shuffling:
        random.seed(_RANDOM_SEED)
        random.shuffle(filenames)

    # Process dataset files.
    i = 0
    fidx = 0
    while i < len(filenames):
        # Open new TFRecord file.
        tf_filename = _get_output_filename(output_dir, name, fidx)
        with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
            j = 0
            while i < len(filenames) and j < _SAMPLES_PER_FILES:
                sys.stdout.write('\r>> Converting image %d/%d' % (i+1, len(filenames)))
                sys.stdout.flush()

                filename = filenames[i]
                if not filename.endswith('.txt'):
                    i += 1
                    continue
                img_name = filename[:-4]
                _add_to_tfrecord(dataset_dir, img_name, tfrecord_writer)
                i += 1
                j += 1
            fidx += 1

    print('\nFinished converting the nextbutton dataset with %d imgaes!' % i)
