import cv2, os, sys
import numpy as np

SIZEX = 128
SIZEY = 128

margin = 0

argv = sys.argv
argc = len(argv)

print('%s extracts a image within a image table' % argv[0])
print('[usage] python %s <image table> [<nrRows> <nrCols> <margin>]' % argv[0])

if argc < 2:
    quit()

screen = cv2.imread(argv[1])
H, W = screen.shape[:2]

cv2.imshow('screen', screen)

if argc > 2:
    nrRows = int(argv[2])
    SIZEY = H // nrRows
else:
    nrRows = H // SIZEY

if argc > 3:
    nrCols = int(argv[3])
    SIZEX = W // nrCols
else:
    nrCols = W // SIZEX

if argc > 4:
    margin = int(argv[4])

LEFT  = 2424832
UP    = 2490368
RIGHT = 2555904
DOWN  = 2621440

key = -1
no = 1
r = 0
c = 0
prevR = -1
prevC = -1

while key != 27:

    key = cv2.waitKeyEx(100)
    
    if key == UP:
        r -= 1
        if r < 0:
            r = nrRows - 1
    
    elif key == DOWN:
        r += 1
        if r >= nrRows:
            r = 0
    
    elif key == LEFT:
        c -= 1
        if c < 0:
            c = nrCols - 1
    
    elif key == RIGHT:
        c += 1
        if c >= nrCols:
            c = 0
    
    elif key == ord('s'):
        selected = screen[top:bottom, left:right]
        dst_path = '%04d.png' % no
    
        while os.path.isfile(dst_path):
            no += 1
            dst_path = '%04d.png' % no
    
        cv2.imwrite(dst_path, selected)
        print('save %s' % dst_path)
        no += 1
   
    if r != prevR or c != prevC:
        left = c * SIZEX + (c + 1) * margin
        right = left + SIZEX
        top = r * SIZEY + (r + 1) * margin
        bottom = top + SIZEY
     
        dst = screen.copy()
        dst = cv2.rectangle(dst, (left, top), (right, bottom), (255, 0, 255), 2)
        cv2.imshow('screen', dst) 

        prevR = r
        prevC = c

cv2.destroyAllWindows()
