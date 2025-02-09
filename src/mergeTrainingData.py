import cv2, glob, os, sys
import numpy as np

argv = sys.argv
argc = len(argv)

print('%s merges training data' % argv[0])
print('[usage] python %s <training data1(.npy)> <training data2(.npy)>' % argv[0])

if argc < 3:
    quit()

data1 = np.load(argv[1])
data2 = np.load(argv[2])

print('%s shape:' % argv[1], data1.shape)
print('%s shape:' % argv[2], data2.shape)

c1 = data1.shape[-1]
c2 = data2.shape[-1]

w1 = data1.shape[-2]
w2 = data2.shape[-2]

h1 = data1.shape[-3]
h2 = data2.shape[-3]

if h1 != h2 or w1 != w2 or c1 != c2:
    print('dimension mismatch: (%d x %d x %d) - (%d x %d x %d)' % (h1, w1, c1, h2, w2, c2))
    quit()

if len(data1.shape) == 3:
   data1 = np.expand_dims(data1, axis=0)

if len(data2.shape) == 3:
   data2 = np.expand_dims(data2, axis=0)

merged = np.append(data1, data2, axis=0)
print('merged shape:', merged.shape)

base1 = os.path.basename(argv[1])
filename1 = os.path.splitext(base1)[0]

base2 = os.path.basename(argv[2])
filename2 = os.path.splitext(base2)[0]

dst_path = 'merge_%s_%s.npy' % (filename1, filename2)
np.save(dst_path, merged)
print('save %s' % dst_path)




