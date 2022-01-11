import sys
import cv2 as cv # ! pip install opencv-python
import numpy as np # ! pip install numpy
import matplotlib.pyplot as plt # ! pip install matplotlib
from skimage import util # ! pip install scikit-image

def main():
    path = sys.argv[1]
    src = cv.imread(path)
    
    # sharpen image
    kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float32)
    imgLaplacian = cv.filter2D(src, cv.CV_32F, kernel)
    sharp = np.float32(src)
    imgResult = sharp - imgLaplacian
    imgResult = np.clip(imgResult, 0, 255)
    imgResult = imgResult.astype('uint8')
    imgLaplacian = np.clip(imgLaplacian, 0, 255)
    imgLaplacian = np.uint8(imgLaplacian)
    
    # convert to binary image
    bw = cv.cvtColor(imgResult, cv.COLOR_BGR2GRAY)
    _, bw = cv.threshold(bw, 40, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    
    # clean noise
    bw = util.invert(bw)
    kernel = np.ones((5,5),np.uint8)
    bw = cv.morphologyEx(bw, cv.MORPH_OPEN, kernel, iterations = 2)
    bw = cv.morphologyEx(bw, cv.MORPH_CLOSE, kernel, iterations = 5)
    kernel = np.ones((3,3),np.uint8)
    bw = cv.morphologyEx(bw, cv.MORPH_CLOSE, kernel, iterations = 10)
    kernel = np.ones((5,5),np.uint8)
    bw = cv.morphologyEx(bw, cv.MORPH_OPEN, kernel, iterations = 2)
    
    # fill in holes caused by starch granules
    inv_bw = util.invert(bw)
    h, w = bw.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv.floodFill(inv_bw, mask, (0,0), 0);
    im_out = bw | inv_bw
    
    # distance transform
    dist = cv.distanceTransform(im_out, cv.DIST_L2, 3)
    cv.normalize(dist, dist, 0, 1.0, cv.NORM_MINMAX)
    
    # extract peaks from distance transform
    _, dist = cv.threshold(dist, 0.1, 1.0, cv.THRESH_BINARY)
    kernel1 = np.ones((3,3), dtype=np.uint8)
    dist = cv.dilate(dist, kernel1)
    
    # clean up noise from peaks
    kernel = np.ones((4,4),np.uint8)
    dist = cv.morphologyEx(dist, cv.MORPH_OPEN, kernel, iterations = 2)
    kernel = np.ones((3,3),np.uint8)
    dist = cv.erode(dist, kernel, 2)
    dist = cv.dilate(dist, kernel, 2)
    
    # count sufficiently large regions
    dist_8u = dist.astype('uint8')
    contours, _ = cv.findContours(dist_8u, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    count = 0
    for cont in contours:
        if (len(cont) > 4):
            count += 1
    print(count)

if __name__ == '__main__':
    main()
