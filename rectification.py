import numpy as np

import math
from skimage.transform import warp
import numpy as np

def normalize_np(v):
    return v/np.linalg.norm(v)

def unproject(pt,f,w,h,p,towards,right,up):
    v = pt - p
    d = np.dot(v,towards)
    v = f*v/d
    x = np.dot(v,right) + (w-1)/2
    y = (h-1)/2 - np.dot(v,up)
    return np.array([x,y])

def computeHomography(imI, projected, texresX=400, texresY=400):
    points = np.array([[0,0],[0,texresX],[texresY,texresX],[texresY,0]])

    # Solve
    M = np.zeros((8,8),dtype=np.dtype('f8'))
    M[0:4,0:2] = points
    M[4:8,3:5] = points
    M[0:4,2] = 1
    M[4:8,5] = 1
    M[0:4,6:8] = -points*np.expand_dims(projected[:,0],1)
    M[4:8,6:8] = -points*np.expand_dims(projected[:,1],1)

    b = np.expand_dims(np.concatenate([projected[:,0], projected[:,1]]),1)
    x = np.linalg.solve(M,b)
    x = np.reshape(np.concatenate([x,np.array([[1]])]),(3,3))

    return np.transpose(warp(imI, x, output_shape=(texresX,texresY), mode='constant', cval=0.0), (1, 0, 2))

def computeRectification(imI, camtheta, camphi, texres=400):
    r = 26.
    h, w, _ = imI.shape
    hfov = 45. * math.pi/180
    theta = camtheta * math.pi/180
    phi = camphi * math.pi/180
    f = w / (2*math.tan(hfov/2))

    # Camera params         
    p = np.array([r*math.cos(theta)*math.cos(phi),                                  
         r*math.sin(phi),
         r*math.sin(theta)*math.cos(phi)], dtype=np.dtype('f8'))                                          
    towards = normalize_np(np.array([0,1,0], dtype=np.dtype('f8')) - p)
    right = normalize_np(np.cross(towards,np.array([0,1,0], dtype=np.dtype('f8'))))
    up = np.cross(right, towards)

    worldpoints = np.array([[-8,0,8],[-8,0,-8],[8,0,-8],[8,0,8]],dtype=np.dtype('f8'))
    projected = np.stack([unproject(worldpoints[i],f,w,h,p,towards,right,up) for i in range(4)])
    return computeHomography(imI, projected, texres, texres)
    
    
if __name__ == '__main__':
    import argparse
    import cv2
    parser = argparse.ArgumentParser()
    parser.add_argument('image')
    parser.add_argument('points', metavar=('X1', 'Y1', 'X2', 'Y2', 'X3', 'Y3', 'X4', 'Y4'),nargs=8, type=float, help='control points in input image for rectification')
    parser.add_argument('--resolution', type=int, metavar=('RESOLUTIONX','RESOLUTIONY'), nargs=2, default=[512, 512], help='output resolution')
    parser.add_argument('--output', metavar='FILENAME',default='rectification_test.png')
    parser.add_argument('--scales', metavar=('SCALEX','SCALEY'),nargs=2, type=float, help='amount to scale the x and y coordinates of input points', default=[1,1])
    args = parser.parse_args()
    im = cv2.imread(args.image)
    projected = np.zeros([4,2])
    points = args.points
    points.reverse()
    projected[:,1] = points[::2]
    projected[:,0] = points[1::2]
    projected[:,0] *= args.scales[0]
    projected[:,1] *= args.scales[1]
    imr = computeHomography(im, projected, texresX=args.resolution[0], texresY=args.resolution[1])
    cv2.imwrite(args.output, np.clip(imr*255, 0, 255).astype(np.uint8))
    
