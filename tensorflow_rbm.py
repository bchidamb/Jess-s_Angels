#https://github.com/whyjay/RBM.tensorflow/blob/master/rbm.py

from __future__ import print_function
import PIL.Image as Image
import time

import numpy as np
import tensorflow as tf
import os


class RBM(object):
    def __init__(self, sess, dim_visible=784, dim_hidden=500, y_dim=10, image_width=28,
                 W=None, hbias=None, vbias=None, k=15, dataset_name='mnist', batch_size=64):

        self.sess = sess

        self.dim_visible = dim_visible
        self.dim_hidden = dim_hidden
        self.y_dim = y_dim
        self.image_width = image_width

        self.dataset_name = dataset_name
        self.batch_size = batch_size

        abs_val = -4*np.sqrt(6./(self.dim_hidden + self.dim_visible))
        self.W = tf.get_variable("W", [self.dim_visible, self.dim_hidden], tf.float32,
                            tf.random_uniform_initializer(minval=-abs_val, maxval=abs_val))
        self.h_bias = tf.get_variable("h_bias", [self.dim_hidden], tf.float32,
                                 tf.constant_initializer(0.0))
        self.v_bias = tf.get_variable("v_bias", [self.dim_visible], tf.float32,
                                 tf.constant_initializer(0.0))

        self.k = k
        self.chain = tf.get_variable("chain", [batch_size, self.dim_hidden],
                                     tf.float32, tf.constant_initializer(0.0))
        self.monitoring_loss = None
        self.build_model()

    def build_model(self):
        self.images = tf.placeholder(tf.float32, [self.batch_size] +
                                     [self.image_width, self.image_width],
                                     name='real_images')

        pre_sigmoid_h, h_mean, h_sample = self.sample_h_given_v(self.images)


        # k Gibbs step
        nh_sample = h_sample
        for k in xrange(self.k):
            [ pre_sigmoid_nvs, nv_mean, nv_sample,
            pre_sigmoid_nh, nh_mean, nh_sample ] = self.gibbs_hvh(nh_sample)

        chain_end = nv_sample

        # CD-k loss
        self.loss = tf.reduce_mean(self.free_energy(self.images)) \
            - tf.reduce_mean(self.free_energy(chain_end))

        self.saver = tf.train.Saver()

    def train(self, config):
        data_X, data_y = self.load_mnist()
        optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1)\
            .minimize(self.loss)
        tf.initialize_all_variables().run()

        if self.load(config.checkpoint_dir):
            print(" [*] Load Success")
        else:
            print(" [!] Load Failed")

        counter = 1
        start_time = time.time()

        for epoch in xrange(config.epoch):
            batch_idxs = len(data_X) // config.batch_size

            for idx in xrange(0, batch_idxs):
                batch_images = data_X[idx*self.batch_size:(idx+1)*self.batch_size]
                batch_labels = data_y[idx*self.batch_size:(idx+1)*self.batch_size]

                _ = self.sess.run([optim], feed_dict={self.images: batch_images})
                loss = self.loss.eval({self.images: batch_images})

                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.8f" \
                    % (epoch, idx, batch_idxs, time.time() - start_time, loss))

                if np.mod(counter, 500) == 2:
                    self.save(config.checkpoint_dir, counter)

    def free_energy(self, v_sample):
        flat_v = tf.reshape(v_sample, [self.batch_size, -1])
        wx_b = tf.matmul(flat_v, self.W) + self.h_bias
        v_bias_term = tf.matmul(flat_v, tf.reshape(self.v_bias, [-1,1]))
        h_term = tf.reduce_sum(tf.log(1 + tf.exp(wx_b)))
        return - h_term - v_bias_term

    def sample_prob(self, probs):
        return tf.nn.relu(tf.sign(probs - tf.random_uniform(tf.shape(probs))))

    def propup(self, vis):
        pre_sigmoid = tf.matmul(tf.reshape(vis, [self.batch_size, -1]), self.W) + self.h_bias
        return [pre_sigmoid, tf.nn.sigmoid(pre_sigmoid)]

    def sample_h_given_v(self, v0_sample):
        pre_sigmoid_h1, h1_mean = self.propup(v0_sample)
        h1_sample = self.sample_prob(h1_mean)
        return [pre_sigmoid_h1, h1_mean, h1_sample]

    def propdown(self, hid):
        pre_sigmoid = tf.matmul(hid, tf.transpose(self.W)) + self.v_bias
        return [pre_sigmoid, tf.nn.sigmoid(pre_sigmoid)]

    def sample_v_given_h(self, h0_sample):
        pre_sigmoid_v1, v1_mean = self.propdown(h0_sample)
        v1_sample = self.sample_prob(v1_mean)
        return [pre_sigmoid_v1, v1_mean, v1_sample]

    def gibbs_hvh(self, h0_sample):
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v1_sample)
        return [pre_sigmoid_v1, v1_mean, v1_sample,
                pre_sigmoid_h1, h1_mean, h1_sample]

    def gibbs_vhv(self, v0_sample):
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v0_sample)
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h1_sample)
        return [pre_sigmoid_h1, h1_mean, h1_sample,
                pre_sigmoid_v1, v1_mean, v1_sample]

    def save(self, checkpoint_dir, step):
        model_name = "RBM.model"
        model_dir = "%s_%s" % (self.dataset_name, self.batch_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print (" [*] Reading checkpoints...")

        model_dir = "%s_%s" % (self.dataset_name, self.batch_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    def load_mnist(self):
        data_dir = os.path.join("./data", self.dataset_name)

        with open(os.path.join(data_dir,'train-images-idx3-ubyte')) as f:
            loaded = np.fromfile(file=f,dtype=np.uint8)
            tr_X = loaded[16:].reshape((60000,28,28)).astype(np.float)

        with open(os.path.join(data_dir,'train-labels-idx1-ubyte')) as f:
            loaded = np.fromfile(file=f,dtype=np.uint8)
            tr_y = loaded[8:].reshape((60000)).astype(np.float)

        with open(os.path.join(data_dir,'t10k-images-idx3-ubyte')) as f:
            loaded = np.fromfile(file=f,dtype=np.uint8)
            te_X = loaded[16:].reshape((10000,28,28)).astype(np.float)

        with open(os.path.join(data_dir,'t10k-labels-idx1-ubyte')) as f:
            loaded = np.fromfile(file=f,dtype=np.uint8)
            te_y = loaded[8:].reshape((10000)).astype(np.float)

        X = np.concatenate((tr_X, te_X), axis=0)
        y = np.concatenate((tr_y, te_y), axis=0)

        seed = 831
        np.random.seed(seed)
        np.random.shuffle(X)
        np.random.seed(seed)
        np.random.shuffle(y)

        y_onehot = np.zeros((len(y), self.y_dim), dtype=np.float)
        for i, label in enumerate(y):
            y_onehot[i, int(label)] = 1.0

        return X/255., y_onehot
