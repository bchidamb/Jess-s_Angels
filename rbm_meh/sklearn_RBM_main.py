#https://github.com/whyjay/RBM.tensorflow/blob/master/main.py

import os
import scipy.misc
import numpy as np
import argparse
from IPython import embed

from rbm import RBM
from utils import pp
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", default=25, help="Epoch to train [25]")
parser.add_argument("--learning_rate", default=0.01, help="Learning rate of adam [0.0002]")
parser.add_argument("--beta1", default=0.5, help="Momentum term of adam [0.5]")
parser.add_argument("--batch_size", default=20, help="Size of batch images [64]")
parser.add_argument("--n_hidden", default=500, help="Size of Hidden layer")
parser.add_argument("--output_dir", default="rbm_plots")
parser.add_argument("--checkpoint_dir", default="checkpoint", help="Directory name to save the checkpoint [checkpoint]")
parser.add_argument("--sample_dir", default="samples", help="Directory name to save image samples [samples]")
parser.add_argument("--is_train", default=False, help="True for training, False for testing [False]")

def main():
    args = parser.parse_args()
    pp.pprint(args)

    if not os.path.exists(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)
    if not os.path.exists(args.sample_dir):
        os.mkdir(args.sample_dir)

    with tf.Session() as sess:
        rbm = RBM(sess)

        if args.is_train:
            rbm.train(args)
        else:
            rbm.load(args.checkpoint_dir)



if __name__ == "__main__":
main()
