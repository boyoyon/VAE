import cv2, glob, sys
import numpy as np 

SIZE = 128

prefix = 'cat'

argv = sys.argv
argc = len(argv)

print('%s creates training data(.npy) from images' % argv[0]) 
print('[usage] python %s <wildcard for images>' % argv[0])

if argc < 2:
    quit()

paths = glob.glob(argv[1])
nrData = len(paths)

imgs = []

for i, path in enumerate(paths):

    print('%d/%d' % (i+1, nrData))

    img = cv2.imread(path)
    H, W = img.shape[:2]
    if H != SIZE or W != SIZE:
        img = cv2.resize(img, (SIZE, SIZE))
    imgs.append(img)

imgs = np.array(imgs)
dst_path = '%s_128x128_%d.npy' % (prefix, nrData)
np.save(dst_path, imgs)
print('save %s' % dst_path)
