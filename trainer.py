from __future__ import print_function                                                                 

import sys
import os
import numpy as np
from tqdm import trange
import pdb
import time
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt



from models import *


def norm_img(img):
  return img / 127.5 - 1.

def denorm_img(img):
  return (img + 1.) * 127.5

class Trainer(object):
  def __init__(self, config, data_loader, label_loader, test_data_loader, test_label_loader):
    self.config = config
    self.data_loader = data_loader
    self.label_loader = label_loader
    self.test_data_loader = test_data_loader
    self.test_label_loader = test_label_loader

    self.optimizer = config.optimizer
    self.batch_size = config.batch_size
    self.batch_size_test = config.batch_size_test
    self.batch_size_gen = config.batch_size_gen

    self.step = tf.Variable(0, name='step', trainable=False)
    self.start_step = 0
    self.log_step = config.log_step
    self.epoch_step = config.epoch_step
    self.max_step = config.max_step
    self.save_step = config.save_step
    self.test_iter = config.test_iter
    self.wd_ratio = config.wd_ratio

    if config.question == 'Q1_1':
      self.model = quick_cnn
    elif config.question == 'Q1_2':
      self.model = VAE
    else:
      self.model = VAE

    self.lr = tf.Variable(config.lr, name='lr')

    # Exponential learning rate decay
    self.epoch_num = config.max_step / config.epoch_step
    decay_factor = (config.min_lr / config.lr)**(1./(self.epoch_num-1.))
    self.lr_update = tf.assign(self.lr, self.lr*decay_factor, name='lr_update')

    self.c_num = config.c_num

    self.model_dir = config.model_dir
    self.load_path = config.load_path

    self.build_model()

    self.build_test_model()

    if config.is_gen:
      self.build_gen_model()


    self.saver = tf.train.Saver()

    self.summary_writer = tf.summary.FileWriter(self.model_dir)
    sv = tf.train.Supervisor(logdir=self.model_dir,
                             is_chief=True,
                             saver=self.saver,
                             summary_op=None,
                             summary_writer=self.summary_writer,
                             save_model_secs=60,
                             global_step=self.step,
                             ready_for_local_init_op=None)

    gpu_options = tf.GPUOptions(allow_growth=True)
    sess_config = tf.ConfigProto(allow_soft_placement=True,
                                 gpu_options=gpu_options)

    self.sess = sv.prepare_or_wait_for_session(config=sess_config)

  def train(self):
    acc = np.zeros(self.max_step)
    it = range(self.start_step,self.max_step)
    start_time = time.time()

    for step in trange(self.start_step, self.max_step):
      fetch_dict = {
        'c_optim': self.c_optim,
        'wd_optim': self.wd_optim,
        'c_loss': self.c_loss,
        'accuracy': self.accuracy,
        'outputs':self.out }

      if step % self.log_step == self.log_step - 1:
        fetch_dict.update({
          'lr': self.lr,
          'summary': self.summary_op,
          'test_outputs': self.test_out })

      result = self.sess.run(fetch_dict)
      acc[step] = result['accuracy']

      if step % self.log_step == self.log_step - 1:
        self.summary_writer.add_summary(result['summary'], step)
        self.summary_writer.flush()
        lr = result['lr']
        c_loss = result['c_loss']
        accuracy = result['accuracy']
        in_images = result['outputs']['input_image']
        out_images = result['outputs']['output_image']
        print("\n[{}/{}:{:.6f}] Loss_C: {:.6f} Accuracy: {:.4f}" . \
        	format(step, self.max_step, lr, c_loss, accuracy))
        sys.stdout.flush()

      if step % self.save_step == self.save_step - 1:
        self.saver.save(self.sess, self.model_dir + '/model')
        plt.close('all')
        plt.figure()
        plt.subplot(1,2,1)
        plt.title('org_img')
        plt.imshow(in_images[1], cmap='gray')

        plt.subplot(1,2,2)
        plt.title('rec_img')
        plt.imshow(out_images[1], cmap='gray')

        plt.savefig(self.config.model_dir + '/iter_%d_image_1'%(step) +'.png' )
        plt.close('all')
        time.sleep(0.1)


        plt.figure()
        plt.subplot(1,2,1)
        plt.title('org_img')
        plt.imshow(in_images[2], cmap='gray')

        plt.subplot(1,2,2)
        plt.title('rec_img')
        plt.imshow(out_images[2], cmap='gray')

        plt.savefig(self.config.model_dir + '/iter_%d_image_2'%(step) +'.png' )
        plt.close('all')

        time.sleep(0.1)
        test_accuracy = 0
        for iter in xrange(self.test_iter):
          fetch_dict = { "test_accuracy":self.test_accuracy,
          				"test_outputs":self.test_out }
          result = self.sess.run(fetch_dict)
          test_accuracy += result['test_accuracy']
          t_in_images = result['test_outputs']['input_image']
          t_out_images = result['test_outputs']['output_image']

        test_accuracy /= self.test_iter
          

        print("\n[{}/{}:{:.6f}] Test Accuracy: {:.4f}" . \
              format(step, self.max_step, lr, test_accuracy))
        sys.stdout.flush()

      if step % self.epoch_step == self.epoch_step - 1:
        self.sess.run([self.lr_update])

    tr_time = time.time() - start_time
    plt.figure(1)
    plt.plot(it,acc)
    plt.title('training accuracy')
    plt.savefig(self.config.model_dir+'/question_' + self.config.question + 
      '_test_accuracy=%.4f_train_time=%.4fsecs'%(test_accuracy,tr_time)+ '.png' )
    plt.close(1)
    time.sleep(0.1)


    plt.figure()
    plt.subplot(121)
    plt.title('org_img')
    plt.imshow(t_in_images[1], cmap='gray')
    plt.subplot(122)
    plt.title('rec_img')
    plt.imshow(t_out_images[1], cmap='gray')  
    plt.savefig(self.config.model_dir + '/test_image_1' +'.png' )
    plt.close('all')
    time.sleep(0.1)


    plt.figure()
    plt.subplot(121)
    plt.title('org_img')
    plt.imshow(t_in_images[2], cmap='gray')

    plt.subplot(122)
    plt.title('rec_img')
    plt.imshow(t_out_images[2], cmap='gray')
    
    plt.savefig(self.config.model_dir + '/test_image_2' +'.png' )
    plt.close('all')
    time.sleep(0.1)



  def build_model(self):
    self.x = self.data_loader
    self.labels = self.label_loader
    x = self.x

    self.c_loss, feat, self.accuracy, self.c_var, self.out = self.model(
      x, self.labels, self.c_num, self.batch_size, is_train=True, reuse=False)

    # self.c_loss = tf.reduce_mean(self.c_loss, 0)

    wd_optimizer = tf.train.GradientDescentOptimizer(self.lr)
    if self.optimizer == 'sgd':
      c_optimizer = tf.train.MomentumOptimizer(self.lr, 0.9)
    elif self.optimizer == 'adam':
      c_optimizer = tf.train.AdamOptimizer(self.lr)
    else:
      raise Exception("[!] Caution! Don't use {} opimizer.".format(self.optimizer))

    for var in tf.trainable_variables():
      weight_decay = tf.multiply(tf.nn.l2_loss(var), self.wd_ratio)
      tf.add_to_collection('losses', weight_decay)
    wd_loss = tf.add_n(tf.get_collection('losses'))

    self.c_optim = c_optimizer.minimize(self.c_loss, var_list=self.c_var)
    self.wd_optim = wd_optimizer.minimize(wd_loss)

    self.summary_op = tf.summary.merge([
      tf.summary.scalar("c_loss", self.c_loss),
      tf.summary.scalar("accuracy", self.accuracy),
      tf.summary.scalar("lr", self.lr),
      # tf.summary.image("inputs", self.x),
      tf.summary.histogram("feature", feat)
    ])



  def test(self):
    self.saver.restore(self.sess, self.model_dir + '/model.ckpt-0')
    test_accuracy = 0
    for iter in trange(self.test_iter):
      fetch_dict = {"test_accuracy":self.test_accuracy,
                    "test_outputs" :self.test_out}
      result = self.sess.run(fetch_dict)
      test_accuracy += result['test_accuracy']
      t_out_images = result['test_outputs']['output_image']
      t_in_images = result['test_outputs']['input_image']

    test_accuracy /= self.test_iter

    print (test_accuracy)

    plt.figure()
    plt.subplot(121)
    plt.title('org_img')
    plt.imshow(t_in_images[1], cmap='gray')
    plt.subplot(122)
    plt.title('rec_img')
    plt.imshow(t_out_images[1], cmap='gray')  
    plt.savefig(self.config.model_dir + '/test_image_1' +'.png' )
    plt.close('all')
    time.sleep(0.1)


    plt.figure()
    plt.subplot(121)
    plt.title('org_img')
    plt.imshow(t_in_images[2], cmap='gray')

    plt.subplot(122)
    plt.title('rec_img')
    plt.imshow(t_out_images[2], cmap='gray')
    
    plt.savefig(self.config.model_dir + '/test_image_2' +'.png' )
    plt.close('all')
    time.sleep(0.1)

  def build_test_model(self):
    self.test_x = self.test_data_loader
    self.test_labels = self.test_label_loader
    test_x = self.test_x

    loss, self.test_feat, self.test_accuracy, var,self.test_out = self.model(
      test_x, self.test_labels, self.c_num, self.batch_size_test, is_train=False, reuse=True)
  
  def build_gen_model(self):
    self.test_x = self.test_data_loader
    self.test_labels = self.test_label_loader
    test_x = self.test_x
    self.noise_in = tf.placeholder(tf.float32,shape = (self.batch_size_gen,64))

    inputs = [test_x, self.noise_in]

    loss, self.gen_feat, self.gen_accuracy, var,self.gen_out = self.model(
      inputs, self.test_labels, 0, self.batch_size_gen, is_train=False, reuse=True)

  def gen_images(self):
    self.saver.restore(self.sess, self.model_dir + '/model.ckpt-0')

    noise_in= np.random.multivariate_normal(mean=np.zeros(64), cov=np.eye(64), size = (self.batch_size_gen))


    fetch_dict = {"outputs":self.gen_out}
    feed_dict = {self.noise_in : noise_in}
    result = self.sess.run(fetch_dict,feed_dict = feed_dict)
    t_out_images = result['outputs']['output_image']


    plt.figure()
    plt.title('random_img')
    plt.imshow(t_out_images[0], cmap='gray')  
    plt.savefig(self.config.model_dir + '/random_image_1' +'.png' )
    plt.close('all')
    time.sleep(0.1)

    noise_in= np.random.multivariate_normal(mean=np.zeros(64), cov=np.eye(64), size = (self.batch_size_gen))


    fetch_dict = {"outputs":self.gen_out}
    feed_dict = {self.noise_in : noise_in}
    result = self.sess.run(fetch_dict,feed_dict = feed_dict)
    t_out_images = result['outputs']['output_image']

    plt.figure()
    plt.title('random_img')
    plt.imshow(t_out_images[0], cmap='gray')
    plt.savefig(self.config.model_dir + '/random_image_2' +'.png' )
    plt.close('all')
    time.sleep(0.1)


  def gen_trans_images(self):
    self.saver.restore(self.sess, self.model_dir + '/model.ckpt-0')

    fetch_dict = {"test_accuracy":self.test_accuracy,
                  "test_outputs" :self.test_out}

    result = self.sess.run(fetch_dict)
    means = result['test_outputs']['x_mean']
    variances = result['test_outputs']['x_var']**2

    t_in_images = result['test_outputs']['input_image']


    plt.figure()
    plt.title('image_1')
    plt.imshow(t_in_images[0], cmap='gray')  
    plt.savefig(self.config.model_dir + '/org1' +'.png' )
    plt.close('all')
    time.sleep(0.1)

    plt.figure()
    plt.title('image_2')
    plt.imshow(t_in_images[1], cmap='gray')  
    plt.savefig(self.config.model_dir + '/org2' +'.png' )
    plt.close('all')
    time.sleep(0.1)

    m1 = np.tile(means[0],(self.batch_size_test, 1))
    m2 = np.tile(means[1],(self.batch_size_test, 1))

    # m1= np.random.multivariate_normal(mean=means[0], cov=np.diag(variances[0])
    #   , size = (self.batch_size_gen))

    # m2= np.random.multivariate_normal(mean=means[1], cov=np.diag(variances[1])
    #   , size = (self.batch_size_gen))


    for idx,i in enumerate(np.arange(0,1.1,0.1)):
      noise_in = (m1*i + m2*(1-i)).reshape(self.batch_size_test,64)
      fetch_dict = {"outputs":self.gen_out}
      feed_dict = {self.noise_in : noise_in}
      result = self.sess.run(fetch_dict,feed_dict = feed_dict)
      t_out_images = result['outputs']['output_image']

      plt.figure()
      plt.title('image_%d'%(idx))
      plt.imshow(t_out_images[0], cmap='gray')  
      plt.savefig(self.config.model_dir + '/trans_%d'%(idx) +'.png' )
      plt.close('all')
      time.sleep(0.1)


    # plt.figure()
    # plt.title('random_img')
    # plt.imshow(t_out_images[2], cmap='gray')
    # plt.savefig(self.config.model_dir + '/random_image_2' +'.png' )
    # plt.close('all')
    # time.sleep(0.1)

    # mean = result['outputs']['x_mean'][0]
    # var = np.diag(result['outputs']['x_var'][0])


    # noise_in = np.random.randn(self.batch_size_gen,64)
    # fetch_dict = {"outputs":self.test_out}
    # feed_dict = {self.noise_in : noise_in}
    # result = self.sess.run(fetch_dict,feed_dict = feed_dict)

      

