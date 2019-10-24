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
parser.add_argument('--resolution', default=256, type=int, help='output resolution')
parser.add_argument('--imagepath', default=None, type=str, help='directory of images to load if image is None')
parser.add_argument('--suffix', default='_N_T.exr', type=str, help='suffix of images')
parser.add_argument('--maskpath', default=None, type=str, help='where to look for masks')
parser.add_argument('--masksuffix', default='objectmask.png', type=str, help='suffix of masks')
parser.add_argument('--invert_mask', action='store_true', help='invert the mask')
parser.add_argument('--skip', default=None, choices=['mask','image'],type=str, help='--skip (mask | image): skip outputting the mask or image')
parser.add_argument('--start', default=0, type=int, help='--number to start from')
parser.add_argument('--end', default=100000, type=int, help='--max id')
parser.add_argument('--rectmask', action='store_true', help='only render a mask of the visible region of the plane')
parser.add_argument('--mode', choices=['hdrimage','hdrdepth', 'ldrimage'], default='ldrimage', help='whether to scale images as if they were HDR color images or depth maps, or do nothing (ldr image)')
parser.add_argument('--maxdepth', type=float, default=50, help='maximum depth value for hdrdepth mode')

if __name__ == "__main__":
    args = parser.parse_args()
    if args.image is None:
        print('loading data')
        if args.imagepath is not None:
            files = [f.name for f in os.scandir(args.imagepath) if f.name.endswith(args.suffix)]
        if args.maskpath is not None and args.imagepath is not None:
            masks = [f.name for f in os.scandir(args.maskpath) if f.name.endswith(args.masksuffix)]
            ids = {file.split('_')[0] for file in files}.intersection({mask.split('_')[0] for mask in masks})
            ids = {i for i in ids if int(i[:5]) >= args.start and int(i[:5]) < args.end}
            images = [d.name for d in os.scandir(args.imagepath) if d.name.endswith(args.suffix) and d.name.split('_')[0] in ids]
            masks = [d.name for d in os.scandir(args.maskpath) if d.name.endswith(args.masksuffix) and d.name.split('_')[0] in ids]
            masks.sort()
        else:
            if args.imagepath is not None:
                images = [f for f in files if int(f.split('_')[0][:5]) >= args.start and int(f.split('_')[0][:5]) < args.end and f.endswith(args.suffix)]
                images.sort()
            if args.maskpath is not None:
                masks = [f.name for f in os.scandir(args.maskpath) if f.name.endswith(args.masksuffix)]
                masks.sort()
        
        iterator = range(len(images)) if args.imagepath is not None else range(len(masks))
        for i in iterator:
            file = images[i] if args.imagepath is not None else masks[i]
            head, tail = os.path.split(file)
            theta_index = tail.find('Theta')+5
            theta_end = tail[theta_index:].find('_')
            theta = float(tail[theta_index:theta_index+theta_end])
            phi_index = tail.find('Phi')+3
            phi_end = tail[phi_index:].find('_')
            phi = float(tail[phi_index:phi_index+phi_end])
            print(tail)
            print('theta',theta)
            print('phi:',phi)
            if args.rectmask:
                image = np.full([args.resolution*2,args.resolution*2,3], 255, np.uint8)
                imagerec = computeRectification(image, theta, phi, texres=args.resolution)
                imagerec = (imagerec == 1).astype(np.uint8)*255
                cv2.imwrite(os.path.join(args.output, tail.split('_')[0]+'_rectmask.png'), imagerec)
            else:
                if not args.skip == 'image':
                    image = cv2.imread(os.path.join(args.imagepath, file), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
                    if image is not None:
                        imagerec = computeRectification(image.astype(np.float64), theta, phi, texres=args.resolution)
                        #cv2.imwrite(os.path.join(args.output, tail), imagerec.astype(np.float32))
                        if args.mode == 'hdrimage':
                            imagerec = np.clip((imagerec**0.4545)*255, 0, 255).astype(np.uint8)
                        elif args.mode == 'hdrdepth':
                            imagerec2 = np.clip(imagerec/args.maxdepth*255.0,0,255).astype(np.uint8)
                            imagerec2[imagerec == 0.0] = 255
                            imagerec = imagerec2.astype(np.uint8)
                        elif args.mode == 'ldrimage':
                            pass
                        else:
                            print('unimplemented mode')
                            exit()

                        cv2.imwrite(os.path.join(args.output, tail + '.png'), imagerec)
                            
                    else:
                        print(file, 'not found')
                if not args.skip == 'mask':
                    if args.maskpath is not None:
                        mask = masks[i]
                        mhead, mtail = os.path.split(mask)
                        print(mtail)
                        maskimg = cv2.imread(os.path.join(args.maskpath, mask))
                        if maskimg is not None:
                            if args.invert_mask:
                                maskimg = 255 - maskimg
                            maskrec = computeRectification(maskimg, theta, phi, texres=args.resolution)
                            maskrec = (maskrec == 1) * 255
                            cv2.imwrite(os.path.join(args.output, mtail), maskrec)
                        else:
                            print('mask', mask, 'not found')
            
            
    else:
        image = cv2.imread(args.image, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
        if image is not None:
            print('image dimensions:', image.shape)
            print('image type:', image.dtype)
            imagerec = computeRectification(image, args.theta, args.phi, texres=args.resolution)
            print('rectified type:', imagerec.dtype)
            imagerec = np.clip((imagerec**0.4545)*255, 0, 255).astype(np.uint8)
            cv2.imwrite(args.output, imagerec)
        else:
            print('image not found')
