import cv2, os, sys
import numpy as np

SIZE = 7

argv = sys.argv
argc = len(argv)

print('%s blurres training data' % argv[0])
print('[usage] python %s <training_data(.npy)> [<SIZE>]' % argv[0])

if argc < 2:
    quit()

if argc > 2:
    SIZE = int(argv[2])
    if SIZE % 2 == 0:
        SIZE += 1

imgs = np.load(argv[1])
nrImages = imgs.shape[0]

blurs = []

for i, img in enumerate(imgs):

    print('processing: %d/%d' % ((i+1), nrImages))
    blurred = cv2.GaussianBlur(img, (SIZE, SIZE), 0)
    blurs.append(blurred)

base = os.path.basename(argv[1])
filename = os.path.splitext(base)[0]
dst_path = '%s_blurred.npy' % filename
np.save(dst_path, blurs)
print('save %s' % dst_path)

