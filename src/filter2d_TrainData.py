import cv2, glob, os, sys
import numpy as np

CENTER_COEFF = 28.0    

argv = sys.argv
argc = len(argv)

print('%s blurres training data' % argv[0])
print('[usage] python %s <training_data(.npy)> [<center coeff>]' % argv[0])

if argc < 2:
    quit()

imgs = np.load(argv[1])
nrImages = imgs.shape[0]

if argc > 2:
    CENTER_COEFF = float(argv[2])

kernel = np.array([[-2.0, -4.0, -2.0], [-4.0, CENTER_COEFF, -4.0], [-2.0, -4.0, -2.0]])
kernel /= np.sum(kernel)

filter2Ds = []

for i, img in enumerate(imgs):

    print('processing: %d/%d' % ((i+1), nrImages))
    dst = cv2.filter2D(img, -1, kernel)
    filter2Ds.append(dst)

base = os.path.basename(argv[1])
filename = os.path.splitext(base)[0]
dst_path = '%s_filter2d.npy' % filename
np.save(dst_path, filter2Ds)
print('save %s' % dst_path)

