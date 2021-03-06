import os
import numpy as np
import tensorflow as tf
import pdb

def read_labeled_image_list(img_dir):
  """Reads a .txt file containing pathes and labeles
  Args:
    img_dir: path of directory that contains images
  Returns:
    List with all filenames
  """
  files = os.listdir(img_dir)
  img_paths = []
  for file in files:
    if file[-4:] == '.png':
      img_paths.append(img_dir + file)

  return img_paths

def read_images_from_disk(input_queue):
  """Consumes a single filename and label as a ' '-delimited string
  Args:
    filename_and_label_tensor: A scalar string tensor
  Returns:
    Two tensors: the decoded image, and the string label
  """
  img_path = tf.read_file(input_queue)
  img = tf.image.decode_png(img_path, channels=1)

  return img

def get_loader(root, batch_size, shuffle=True):
  """ Get a data loader for tensorflow computation graph
  Args:
    root: Path/to/dataset/root/, a string
    batch_size: Batch size, a integer
    shuffle: If the data should be shuffled every epoch, a boolean
  Returns:
    img_batch: A (float) tensor containing a batch of images.
    lab_batch: A (int) tensor containing a batch of labels.
  """
  img_paths_np = read_labeled_image_list(root+'/imgs/')

  with tf.device('/cpu:0'):
    img_paths = tf.convert_to_tensor(img_paths_np, dtype=tf.string)

    input_queue = tf.train.slice_input_producer([img_paths], shuffle=shuffle, 
      capacity=10*batch_size)

    img = read_images_from_disk(input_queue[0])

    img.set_shape([64, 64, 1])
    img = tf.cast(img, tf.float32)

    img = tf.image.per_image_standardization(img) 

    img_batch= tf.train.batch([img], num_threads=1, batch_size=batch_size, capacity=10*batch_size)

  return img_batch
