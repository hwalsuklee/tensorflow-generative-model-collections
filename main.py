import os

## GAN Variants
from GAN import GAN
from CGAN import CGAN
from infoGAN import infoGAN
from ACGAN import ACGAN
from EBGAN import EBGAN
from WGAN import WGAN
from WGAN_GP import WGAN_GP
from DRAGAN import DRAGAN
from LSGAN import LSGAN
from BEGAN import BEGAN

## VAE Variants
from VAE import VAE
from CVAE import CVAE

from utils import show_all_variables

import tensorflow as tf
import argparse

"""parsing and configuration"""
def parse_args():
    desc = "Tensorflow implementation of GAN collections"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--gan_type', type=str, default='GAN',
                        choices=['GAN', 'CGAN', 'infoGAN', 'ACGAN', 'EBGAN', 'BEGAN', 'WGAN', 'WGAN_GP', 'DRAGAN', 'LSGAN', 'VAE', 'CVAE'],
                        help='The type of GAN', required=True)
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'fashion-mnist', 'celebA'],
                        help='The name of dataset')
    parser.add_argument('--epoch', type=int, default=20, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=100, help='The size of batch')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--result_dir', type=str, default='results',
                        help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory name to save training logs')

    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):
    # --checkpoint_dir
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    # --result_dir
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    # --result_dir
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    # --epoch
    try:
        assert args.epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')

    return args

"""main"""
def main():
    # parse arguments
    args = parse_args()
    if args is None:
      exit()

    # open session
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        # declare instance for GAN
        if args.gan_type == 'GAN':
            gan = GAN(sess, epoch=args.epoch, batch_size=args.batch_size, dataset_name=args.dataset,
                      checkpoint_dir=args.checkpoint_dir, result_dir=args.result_dir, log_dir=args.log_dir)
        elif args.gan_type == 'CGAN':
            gan = CGAN(sess, epoch=args.epoch, batch_size=args.batch_size, dataset_name=args.dataset,
                       checkpoint_dir=args.checkpoint_dir, result_dir=args.result_dir, log_dir=args.log_dir)
        elif args.gan_type == 'ACGAN':
            gan = ACGAN(sess, epoch=args.epoch, batch_size=args.batch_size, dataset_name=args.dataset,
                        checkpoint_dir=args.checkpoint_dir, result_dir=args.result_dir, log_dir=args.log_dir)
        elif args.gan_type == 'infoGAN':
            gan = infoGAN(sess, epoch=args.epoch, batch_size=args.batch_size, dataset_name=args.dataset,
                          checkpoint_dir=args.checkpoint_dir, result_dir=args.result_dir, log_dir=args.log_dir)
        elif args.gan_type == 'EBGAN':
            gan = EBGAN(sess, epoch=args.epoch, batch_size=args.batch_size, dataset_name=args.dataset,
                        checkpoint_dir=args.checkpoint_dir, result_dir=args.result_dir, log_dir=args.log_dir)
        elif args.gan_type == 'WGAN':
            gan = WGAN(sess, epoch=args.epoch, batch_size=args.batch_size, dataset_name=args.dataset,
                       checkpoint_dir=args.checkpoint_dir, result_dir=args.result_dir, log_dir=args.log_dir)
        elif args.gan_type == 'WGAN_GP':
            gan = WGAN_GP(sess, epoch=args.epoch, batch_size=args.batch_size, dataset_name=args.dataset,
                       checkpoint_dir=args.checkpoint_dir, result_dir=args.result_dir, log_dir=args.log_dir)
        elif args.gan_type == 'DRAGAN':
            gan = DRAGAN(sess, epoch=args.epoch, batch_size=args.batch_size, dataset_name=args.dataset,
                       checkpoint_dir=args.checkpoint_dir, result_dir=args.result_dir, log_dir=args.log_dir)
        elif args.gan_type == 'LSGAN':
            gan = LSGAN(sess, epoch=args.epoch, batch_size=args.batch_size, dataset_name=args.dataset,
                         checkpoint_dir=args.checkpoint_dir, result_dir=args.result_dir, log_dir=args.log_dir)
        elif args.gan_type == 'BEGAN':
            gan = BEGAN(sess, epoch=args.epoch, batch_size=args.batch_size, dataset_name=args.dataset,
                        checkpoint_dir=args.checkpoint_dir, result_dir=args.result_dir, log_dir=args.log_dir)
        elif args.gan_type == 'VAE':
            gan = VAE(sess, epoch=args.epoch, batch_size=args.batch_size, dataset_name=args.dataset,
                        checkpoint_dir=args.checkpoint_dir, result_dir=args.result_dir, log_dir=args.log_dir)
        elif args.gan_type == 'CVAE':
            gan = CVAE(sess, epoch=args.epoch, batch_size=args.batch_size, dataset_name=args.dataset,
                        checkpoint_dir=args.checkpoint_dir, result_dir=args.result_dir, log_dir=args.log_dir)
        else:
            raise Exception("[!] There is no option for " + args.gan_type)

        # build graph
        gan.build_model()

        # show network architecture
        show_all_variables()

        # launch the graph in a session
        gan.train()
        print(" [*] Training finished!")

        # visualize learned generator
        gan.visualize_results(args.epoch-1)
        print(" [*] Testing finished!")

if __name__ == '__main__':
    main()