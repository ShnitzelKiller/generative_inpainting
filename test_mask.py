import argparse
from util import str2bool
import cv2
import numpy as np
import tensorflow as tf
import neuralgym as ng
import h5py
import logging
from inpaint_model import InpaintCAModel
import os

parser = argparse.ArgumentParser()
parser.add_argument('--image', default=None, type=str,
                    help='The filename of image to be completed.')
parser.add_argument('--mask', default='', type=str,
                    help='The filename of mask, value 255 indicates mask.')
parser.add_argument('--output', default='output.png', type=str,
                    help='Where to write output.')
parser.add_argument('--checkpoint_dir', default='', type=str,
                    help='The directory of tensorflow checkpoint.')
parser.add_argument('--invert_mask', default='false', type=str, help='Whether to invert mask (0 -> 255, 255 -> 0)')
parser.add_argument('--config', default='inpaint.yml', type=str, help='config file to use')

if __name__ == "__main__":

    ng.get_gpus(1, dedicated=False)
    args = parser.parse_args()
    config = ng.Config(args.config)
    logger = logging.getLogger()
    invert_mask = str2bool(args.invert_mask)
    model = InpaintCAModel()
    mask = cv2.imread(args.mask)
    if args.image is not None:
        image = cv2.imread(args.image)
        assert image.shape == mask.shape
    else:
        with open(config.DATA_FLIST[config.DATASET][1]) as f:
            val_fnames = [(l.split(' ')[0], l.split(' ')[1]) for l in f.read().splitlines()]

        static_fnames = val_fnames[:config.STATIC_VIEW_SIZE]
        static_image = ng.data.DataFromFNames(
                static_fnames, config.IMG_SHAPES, nthreads=1,
                random_crop=config.RANDOM_CROP,gamma=config.GAMMA, exposure=config.EXPOSURE).data_pipeline(1)



    if invert_mask:
        print('inverting mask')
        mask = 255 - mask

    h, w, _ = mask.shape
    grid = 8
    mask = mask[:h//grid*grid, :w//grid*grid, :]
    mask = np.expand_dims(mask, 0)
    
    if args.image is not None:
        image = image[:h//grid*grid, :w//grid*grid, :]
        print('Shape of image: {}'.format(image.shape))
        image = np.expand_dims(image, 0)
        input_image = np.concatenate([image, mask], axis=2)
    

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    with tf.Session(config=sess_config) as sess:
        if args.image is not None:
            input_image = tf.constant(input_image, dtype=tf.float32)
            output = model.build_server_graph(input_image, config=config)
            output = (output + 1.) * 127.5
            output = tf.reverse(output, [-1])
            output = tf.saturate_cast(output, tf.uint8)
        else:
            logger.info('input image size: ' + str(static_image[0].shape))
            input_image = tf.concat([static_image[0], mask], axis=2)
            logger.info('concat image size: ' + str(input_image.shape))
            logger.info('mask size: ' + str(mask.shape))
            output = model.build_server_graph(input_image, config=config, exclusionmask=static_image[1])
            output = (output + 1.) * 127.5
            output = tf.reverse(output, [-1])
            output = tf.saturate_cast(output, tf.uint8)


        # load pretrained model
        vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        assign_ops = []
        for var in vars_list:
            vname = var.name
            from_name = vname
            var_value = tf.contrib.framework.load_variable(args.checkpoint_dir, from_name)
            assign_ops.append(tf.assign(var, var_value))
        sess.run(assign_ops)
        print('Model loaded.')
        if args.image is not None:
            result = sess.run(output)
            
            cv2.imwrite(args.output, result[0][:, :, ::-1])
            #f = h5py.File(args.output + '.h5', 'w')
            #f['output'] = result[0][:,:,::-1]
        else:
            tf.train.start_queue_runners(sess)
            for i in range(config.STATIC_VIEW_SIZE):
                logger.info('saving image ' + str(i))
                result = sess.run(output)
                cv2.imwrite(os.path.join(args.output, str(i) + '.png'), result[0][:,:,::-1])
        
