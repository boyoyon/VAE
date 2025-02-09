import cv2, os, sys
import numpy as np

SIZE = 7

argv = sys.argv
argc = len(argv)

print('%s adjusts contrast of training data' % argv[0])
print('[usage] python %s <training_data(.npy)>' % argv[0])

if argc < 2:
    quit()

imgs = np.load(argv[1])
nrImages = imgs.shape[0]

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2,2))

clahes = []

for i, img in enumerate(imgs):

    print('processing: %d/%d' % ((i+1), nrImages))

    b, g, r = cv2.split(img)
    dst_b = clahe.apply(b)
    dst_g = clahe.apply(g)
    dst_r = clahe.apply(r)

    dst = cv2.merge((dst_b, dst_g, dst_r))
    clahes.append(dst)

base = os.path.basename(argv[1])
filename = os.path.splitext(base)[0]
dst_path = '%s_clahed.npy' % filename
np.save(dst_path, clahes)
print('save %s' % dst_path)

