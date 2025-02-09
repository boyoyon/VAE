import cv2, glob, sys
import numpy as np 

fKEEP_ASPECT = True

SIZE = 128

def padding_resize(img):
    
    H, W = img.shape[:2]

    if H > W:
    
        left = (H - W) // 2
        right = left + W

        tmp = np.empty((H, H, 3), np.uint8)

        for x in range(left):
            tmp[:,x,:] = img[:,0,:]

        for x in range(right, W):
            tmp[:,x,:] = img[:,W-1,:]

        tmp[:,left:right,:] = img

        img = cv2.resize(tmp, (SIZE, SIZE))

    elif H < W:
        
        top = (W - H) // 2
        bottom = top + H

        tmp = np.empty((W, W, 3), np.uint8)

        for y in range(top):
            tmp[y,:,:] = img[0,:,:]
 
        for y in range(bottom, H):
            tmp[y,:,:] = img[H-1,:,:]

        tmp[top:bottom,:,:] = img

        img = cv2.resize(tmp, (SIZE, SIZE))

    else:
        
        img = cv2.resize(img, (SIZE, SIZE))

    return img

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
    
        if fKEEP_ASPECT:
           img = padding_resize(img) 

        else:
            img = cv2.resize(img, (SIZE, SIZE))

    imgs.append(img)

dst_path = 'face128_%d.npy' % nrData
np.save(dst_path, imgs)
print('save %s' % dst_path)
