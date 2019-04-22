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
parser.add_argument('--mask', default=None, type=str,
                    help='The filename of mask, value 255 indicates mask.')
parser.add_argument('--output', default='output.png', type=str,
                    help='Where to write output.')
parser.add_argument('--checkpoint_dir', default='', type=str,
                    help='The directory of tensorflow checkpoint.')
parser.add_argument('--invert_mask', default='false', type=str, help='Whether to invert mask (0 -> 255, 255 -> 0)')
parser.add_argument('--config', default='inpaint.yml', type=str, help='config file to use')
parser.add_argument('--flist', default=None, type=str, help='filename list to use as dataset')
parser.add_argument('--height', default=256, type=int, help='height of images in data flist')
parser.add_argument('--width', default=256, type=int, help='width of images in data flist')

if __name__ == "__main__":

    ng.get_gpus(1, dedicated=False)
    args = parser.parse_args()
    config = ng.Config(args.config)
    logger = logging.getLogger()
    invert_mask = str2bool(args.invert_mask)
    model = InpaintCAModel()
        
    if args.image is not None:
        image = cv2.imread(args.image)
        h, w, _ = image.shape
        assert image.shape == mask.shape
    else:
        if args.flist is None:
            flist = config.DATA_FLIST[config.DATASET][1]
            shapes = config.IMG_SHAPES
        else:
            flist = args.flist 
            shapes = [args.height, args.width, 3]
        with open(flist) as f:
            count = 1
            mask_index = -1
            exclusionmask_index = -1
            if not config.GEN_MASKS:
                mask_index = count
                count += 1
            if config.EXC_MASKS:
                exclusionmask_index = count
                count += 1
            if count == 1:
                val_fnames = f.read().splitlines()
            elif count == 2:
                val_fnames = [(l.split(' ')[0], l.split(' ')[1]) for l in f.read().splitlines()]
            elif count == 3:
                val_fnames = [(l.split(' ')[0], l.split(' ')[1], l.split(' ')[2]) for l in f.read().splitlines()]

        dataset = ng.data.DataFromFNames(
                val_fnames, shapes, nthreads=1,
                random_crop=config.RANDOM_CROP,gamma=config.GAMMA, exposure=config.EXPOSURE)
        static_image = dataset.data_pipeline(1)

    if args.mask is not None:
        mask = cv2.imread(args.mask)
        h, w, _ = mask.shape
        if invert_mask:
            print('inverting mask')
            mask = 255 - mask
        grid = 8
        mask = mask[:h//grid*grid, :w//grid*grid, :]
        mask = np.expand_dims(mask, 0)
    elif not config.GEN_MASKS:
        mask = static_image[mask_index]
    else:
        print('Mask not specified, and no masks in dataset, aborting')
        exit()
    
    if args.image is not None:
        image = image[:h//grid*grid, :w//grid*grid, :]
        print('Shape of image: {}'.format(image.shape))
        image = np.expand_dims(image, 0)
        input_image = np.concatenate([image, mask], axis=2)
    

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True

    if args.image is not None:
        input_image = tf.constant(input_image, dtype=tf.float32)
        output = model.build_server_graph(input_image, config=config)
        output = (output + 1.) * 127.5
        output = tf.reverse(output, [-1])
        output = tf.saturate_cast(output, tf.uint8)
    else:
        logger.info('input image size: ' + str(static_image[0].shape))
        input_image = tf.concat([tf.expand_dims(static_image[0], 0), mask], axis=2)
        logger.info('concat image size: ' + str(input_image.shape))
        logger.info('mask size: ' + str(mask.shape))
        output = model.build_server_graph(input_image, config=config, exclusionmask=static_image[exclusionmask_index] if config.EXC_MASKS else None)
        output = (output + 1.) * 127.5
        output = tf.reverse(output, [-1])
        output = tf.saturate_cast(output, tf.uint8)

    with tf.Session(config=sess_config) as sess:
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
            for i in range(dataset.file_length):
                logger.info('saving image ' + str(i))
                result = sess.run(output)
                cv2.imwrite(os.path.join(args.output, str(i) + '.png'), result[0][:,:,::-1])
        
