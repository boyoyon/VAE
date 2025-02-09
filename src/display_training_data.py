import cv2, sys
import numpy as np

nrRows = 8
nrCols = 8

argv = sys.argv
argc = len(argv)

print('%s displays training data' % argv[0])
print('[usage] python %s <training data file>' % argv[0])

if argc < 2:
    quit()

images = np.load(argv[1])

nrData, H, W, C = images.shape
print(nrData, H, W)

screen = np.empty((nrRows * H, nrCols * W, C), np.uint8)

key = -1
idx = 0

while key != 27: # ESC

    flags = np.ones((nrRows, nrCols), np.int32)
    flags *= -1

    print(idx)

    for y in range(nrRows):
        top = y * H

        for x in range(nrCols):
            left = x * W

            screen[top:top+H, left:left+W] = images[idx]

            idx = (idx + 1) % nrData

    cv2.imshow('scereen', screen)

    key = cv2.waitKey(0)

cv2.destroyAllWindows()
 
