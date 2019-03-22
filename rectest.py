from rectification import *
import cv2
import argparse
import glob
import os
from util import str2bool

parser = argparse.ArgumentParser()
parser.add_argument('--image', default=None, type=str,help='image to rectify')
parser.add_argument('--theta', default=0, type=float, help='theta of camera')
parser.add_argument('--phi', default=0, type=float, help='phi of camera')
parser.add_argument('--output', default='rectification_test.exr', type=str, help='output path')
parser.add_argument('--resolution', default=512, type=int, help='output resolution')
parser.add_argument('--imagepath', default='', type=str, help='directory of images to load if image is None')
parser.add_argument('--suffix', default='_N_T.exr', type=str, help='suffix of images')
parser.add_argument('--maskpath', default=None, type=str, help='where to look for masks')
parser.add_argument('--masksuffix', default='objectmask.png', type=str, help='suffix of masks')


if __name__ == "__main__":
    args = parser.parse_args()
    if args.image is None:
        print('loading data')
        files = [f.name for f in os.scandir(args.imagepath) if f.name.endswith(args.suffix)]
        masks = [f.name for f in os.scandir(args.maskpath) if f.name.endswith(args.masksuffix)]
        ids = {file.split('_')[0] for file in files}.intersection({mask.split('_')[0] for mask in masks})
        images = [d.name for d in os.scandir(args.imagepath) if d.name.endswith(args.suffix) and d.name.split('_')[0] in ids]
        masks = [d.name for d in os.scandir(args.maskpath) if d.name.endswith(args.masksuffix) and d.name.split('_')[0] in ids]
        images.sort()
        masks.sort()
        for file, mask in zip(images, masks):
            head, tail = os.path.split(file)
            mhead, mtail = os.path.split(mask)
            parts = tail.split('_')

            theta = float(parts[3][5:])
            phi = float(parts[4][3:])
            print(tail)
            print(mtail)
            print('theta',theta)
            print('phi:',phi)
            image = cv2.imread(os.path.join(args.imagepath, file), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
            if image is not None:
                imagerec = computeRectification(image.astype(np.float64), theta, phi, texres=args.resolution)
                cv2.imwrite(os.path.join(args.output, tail), imagerec.astype(np.float32))
            else:
                print(file, 'not found')
            #maskimg = cv2.imread(os.path.join(args.maskpath, mask))
            #if maskimg is not None:
            #    maskrec = computeRectification(maskimg, theta, phi, texres=args.resolution)
            #    maskrec = (maskrec == 1) * 255
            #    cv2.imwrite(os.path.join(args.output, mtail), maskrec)
            #else:
            #    print('mask', mask, 'not found')
            
    else:
        image = cv2.imread(args.image, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
        if image is not None:
            print('image dimensions:', image.shape)
            print('image type:', image.dtype)
            imagerec = computeRectification(image, args.theta, args.phi, texres=args.resolution)
            print('rectified type:', imagerec.dtype)
            cv2.imwrite(args.output, imagerec.astype(np.float32))
        else:
            print('image not found')
