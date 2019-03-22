import argparse
import cv2
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--image', default=None, type=str, help='image to be gamma corrected') #644, 354
parser.add_argument('--output', default='gammatest.png', type=str, help='output image')
parser.add_argument('--gamma', default=2.2, type=float, help='gamma')
parser.add_argument('--exposure', default=1, type=float, help='exposure')

if __name__ == '__main__':
    args = parser.parse_args()
    img = cv2.imread(args.image, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    if img is not None:
        img2 = np.clip(255 * (img*args.exposure) ** args.gamma, 0, 255).astype(np.uint8)
        cv2.imwrite(args.output, img2)
    else:
        print('image not found')
        
