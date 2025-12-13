import cv2, os, sys
import numpy as np

CENTER_COEFF = 24.1
kernel = np.array([[-2.0, -4.0, -2.0], [-4.0, CENTER_COEFF, -4.0], [-2.0, -4.0, -2.0]])
kernel /= np.sum(kernel)

argv = sys.argv
argc = len(argv)

print('%s executes HPF on training data' % argv[0])
print('[usage] python %s <training data(.npy)>' % argv[0])

if argc < 2:
    quit()

data = np.load(argv[1])

data_out = np.zeros(data.shape, np.float32)

for i in range(data.shape[0]):

    print('processing %d/%d' % (i+1, data.shape[0]))

    img = data[i].astype(np.float32) / 255.0

    img = cv2.filter2D(img, -1, kernel)/256 + 0.5

    data_out[i] = img

base = os.path.basename(argv[1])
filename = os.path.splitext(base)[0]
dst_path = '%s_HPF.npy' % filename
np.save(dst_path, data_out)
