import tensorflow as tf
from layers import *
import pdb

def quick_cnn(x, labels, c_num, batch_size, is_train, reuse):
  with tf.variable_scope('C', reuse=reuse) as vs:

    image = x
    print (x.shape)
    # conv1
    with tf.variable_scope('conv1', reuse=reuse):
      hidden_num = 4
      x = conv_factory(x, hidden_num, 3, 2, is_train, reuse)
      x = tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,1,1,1], padding='SAME')
      print (x.shape)

    # conv2
    with tf.variable_scope('conv2', reuse=reuse):
      hidden_num *= 2
      x = conv_factory(x, hidden_num, 3, 2, is_train, reuse)
      x = tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,1,1,1], padding='SAME')
      print (x.shape)

    with tf.variable_scope('conv3', reuse=reuse):
      hidden_num *= 2
      x = conv_factory(x, hidden_num, 3, 2, is_train, reuse)
      x = tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,1,1,1], padding='SAME')
      print (x.shape)

    with tf.variable_scope('conv4', reuse=reuse):
      hidden_num *= 2
      x = conv_factory(x, hidden_num, 3, 2, is_train, reuse)
      x = tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,1,1,1], padding='SAME')
      print (x.shape)

    with tf.variable_scope('conv5', reuse=reuse):
      hidden_num *= 2
      x = conv_factory(x, hidden_num, 3, 2, is_train, reuse)
      x = tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,1,1,1], padding='SAME')
      print (x.shape)

    with tf.variable_scope('flatten', reuse=reuse):
      x = tf.reshape(x, [batch_size,-1])
      hidden_num *= 2
      x = fc_factory(x, hidden_num, is_train, reuse)
      x = tf.reshape(x, [batch_size,1,1,128])
      print (x.shape)
    feat = x

    with tf.variable_scope('deconv1', reuse=reuse):
      hidden_num /= 2
      x = t_conv_factory(x, hidden_num,[batch_size,2,2,64] ,3, 1, is_train, reuse)
      print (x.shape)

    with tf.variable_scope('deconv2', reuse=reuse):
      hidden_num /= 2
      x = t_conv_factory(x, hidden_num,[batch_size,4,4,32] ,3, 1, is_train, reuse)
      print (x.shape)

    with tf.variable_scope('deconv3', reuse=reuse):
      hidden_num /= 2
      x = t_conv_factory(x, hidden_num,[batch_size,8,8,16] ,3, 1, is_train, reuse)
      print (x.shape)

    with tf.variable_scope('deconv4', reuse=reuse):
      hidden_num /= 2
      x = t_conv_factory(x, hidden_num,[batch_size,16,16,8] ,3, 1, is_train, reuse)
      print (x.shape)

    with tf.variable_scope('deconv5', reuse=reuse):
      hidden_num /= 2
      x = t_conv_factory(x, hidden_num,[batch_size,32,32,4] ,3, 1, is_train, reuse)
      print (x.shape)

    with tf.variable_scope('deconv6', reuse=reuse):
      hidden_num = 1
      x = t_conv_factory(x, hidden_num,[batch_size,64,64,1] ,3, 1, is_train, reuse)
      print (x.shape)

    # dropout
#    if is_train:
#      x = tf.nn.dropout(x, keep_prob=0.5)

    with tf.variable_scope('L2loss', reuse=reuse):
      loss = tf.losses.mean_squared_error(image,x)
      accuracy = tf.reduce_mean(loss)

    outputs = {"output_image":tf.reshape(x,[batch_size,64,64]),
    		   "input_image":tf.reshape(image,[batch_size,64,64])}

  variables = tf.contrib.framework.get_variables(vs)
  return loss, feat,accuracy, variables, outputs

def VAE(inputs, labels, c_num, batch_size, is_train, reuse):
  with tf.variable_scope('C', reuse=reuse) as vs:
    if type(inputs) is not list:
      x = inputs
    else:
      x = inputs[0]

    image = x
    print (x.shape)
    # conv1
    with tf.variable_scope('conv1', reuse=reuse):
      hidden_num = 4
      x = conv_factory(x, hidden_num, 3, 2, is_train, reuse)
      x = tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,1,1,1], padding='SAME')
      print (x.shape)

    # conv2
    with tf.variable_scope('conv2', reuse=reuse):
      hidden_num *= 2
      x = conv_factory(x, hidden_num, 3, 2, is_train, reuse)
      x = tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,1,1,1], padding='SAME')
      print (x.shape)

    with tf.variable_scope('conv3', reuse=reuse):
      hidden_num *= 2
      x = conv_factory(x, hidden_num, 3, 2, is_train, reuse)
      x = tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,1,1,1], padding='SAME')
      print (x.shape)

    with tf.variable_scope('conv4', reuse=reuse):
      hidden_num *= 2
      x = conv_factory(x, hidden_num, 3, 2, is_train, reuse)
      x = tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,1,1,1], padding='SAME')
      print (x.shape)

    with tf.variable_scope('conv5', reuse=reuse):
      hidden_num *= 2
      x = conv_factory(x, hidden_num, 3, 2, is_train, reuse)
      x = tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,1,1,1], padding='SAME')
      print (x.shape)

    x = tf.reshape(x, [batch_size,-1])

    with tf.variable_scope('gen_mean', reuse=reuse):
      x_mean = fc_factory_noact(x, hidden_num, is_train, reuse)

    with tf.variable_scope('gen_var', reuse=reuse):
      x_var = fc_factory_noact(x, hidden_num, is_train, reuse)

    # with tf.variable_scope('gen', reuse=reuse):
    #   mean_var = fc_factory(x, 128, is_train, reuse)
    #   x_mean = mean_var[:,0:64]
    #   x_var = mean_var[:,64:128]
    #   pdb.set_trace()

    with tf.variable_scope('gen_feat', reuse=reuse):
      z = tf.random_normal(shape=[64],mean=0,stddev=1.0)
      if not is_train:
        z = tf.zeros(1)
      z = (x_var**2)*z + x_mean
      KL_loss = tf.reduce_mean(x_mean**2 + x_var**2 - tf.log(x_var**2) - 1,axis=0)/(2.)

    feat = z

    if c_num == 0:
      z = inputs[1]

    x = tf.reshape(z,[batch_size,1,1,hidden_num])
    print(x.shape)

    with tf.variable_scope('deconv1', reuse=reuse):
      hidden_num /= 2
      x = t_conv_factory(x, hidden_num,[batch_size,2,2,32] ,3, 1, is_train, reuse)
      print (x.shape)

    with tf.variable_scope('deconv2', reuse=reuse):
      hidden_num /= 2
      x = t_conv_factory(x, hidden_num,[batch_size,4,4,16] ,3, 1, is_train, reuse)
      print (x.shape)

    with tf.variable_scope('deconv3', reuse=reuse):
      hidden_num /= 2
      x = t_conv_factory(x, hidden_num,[batch_size,8,8,8] ,3, 1, is_train, reuse)
      print (x.shape)

    with tf.variable_scope('deconv4', reuse=reuse):
      hidden_num /= 2
      x = t_conv_factory(x, hidden_num,[batch_size,16,16,4] ,3, 1, is_train, reuse)
      print (x.shape)

    with tf.variable_scope('deconv5', reuse=reuse):
      hidden_num /= 2
      x = t_conv_factory(x, hidden_num,[batch_size,32,32,2] ,3, 1, is_train, reuse)
      print (x.shape)

    with tf.variable_scope('deconv6', reuse=reuse):
      hidden_num = 1
      x = t_conv_factory_sig(x, hidden_num,[batch_size,64,64,1] ,3, 1, is_train, reuse)
      print (x.shape)

    # dropout
#    if is_train:
#      x = tf.nn.dropout(x, keep_prob=0.5)

    with tf.variable_scope('L2loss', reuse=reuse):
      ML_loss = -tf.reduce_mean(image*tf.log(x) + (1-image)*tf.log(1-x),0)
    
    loss = tf.reduce_mean(ML_loss) + tf.reduce_mean(KL_loss)
    accuracy = tf.reduce_mean(ML_loss)

    outputs = {"output_image":tf.reshape(x,[batch_size,64,64]),
    		   "input_image":tf.reshape(image,[batch_size,64,64]),
           "x_mean": x_mean,
           "x_var": x_var}

  variables = tf.contrib.framework.get_variables(vs)
  return loss, feat,accuracy, variables, outputs

