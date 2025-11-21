import cv2, glob, itertools, os, sys
os.environ['TF_ENABLE_ONEDNN_OPTS']='0' # 警告の抑制
import mediapipe as mp
import numpy as np
import skimage.transform

DECIMATION = 5 # landmark間引き

# Mediapipeの準備
mp_face_mesh = mp.solutions.face_mesh
# refine_landmarks 指定あり
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

# 顔のランドマークを抽出する関数
def get_landmarks(image):
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.multi_face_landmarks:
        return None
    
    landmarks = []

    points = results.multi_face_landmarks[0].landmark

    for i, pt in enumerate(points):
        if i % DECIMATION == 0:
            landmarks.append((int(pt.x * image.shape[1]), int(pt.y * image.shape[0])))

    return np.array(landmarks)

def main():

    argv = sys.argv
    argc = len(argv)

    print('%s warps image using Thin Plate Spline' % argv[0])
    print('[usage] %s <wildcard for images> <alpha(0-100)>' % argv[0])

    if argc < 2:
        quit()

    ALPHA = 50
    if argc > 2:
        ALPHA = int(argv[2])

    alpha = ALPHA / 100

    OUTPUT_FOLDER = 'moprhed'
    no = 2
    while os.path.exists(OUTPUT_FOLDER):
        OUTPUT_FOLDER = 'morphed_%d' % no
        no += 1

    os.mkdir(OUTPUT_FOLDER)

    paths = glob.glob(argv[1])

    nrImages = len(paths)

    landmarks = []
    imgs = []

    for i, path in enumerate(paths):

        print('pre-processing %d/%d' % ((i+1), nrImages))
        img = cv2.imread(path)
        imgs.append(img)

        landmark = get_landmarks(img)
        landmarks.append(landmark)

    pairs = itertools.combinations(np.arange(nrImages), 2)

    tps = skimage.transform.ThinPlateSplineTransform()
        
    for pair in pairs:

        idx0 = pair[0]
        idx1 = pair[1]

        # 画像の読み込み
        img1 = imgs[idx0]
        H1, W1 = img1.shape[:2]
        img2 = imgs[idx1]
        H2, W2 = img2.shape[:2]
        img1 = cv2.resize(img1, ((W1+W2)//2, (H1+H2)//2))
        img2 = cv2.resize(img2, ((W1+W2)//2, (H1+H2)//2))
        
        # ランドマークを取得
        landmarks1 = landmarks[idx0]
        landmarks2 = landmarks[idx1]
    
        blended_landmarks = landmarks1 * alpha + landmarks2 * (1.0 - alpha)
        blended_landmarks = blended_landmarks.astype(np.int32)
    
        tps.estimate(blended_landmarks, landmarks1)
        #tps.estimate(landmarks2, landmarks1)
        img1 = img1.astype(np.float32) / 255.0
        warped1 = skimage.transform.warp(img1, tps)
    
        tps.estimate(blended_landmarks, landmarks2)
        img2 = img2.astype(np.float32) / 255.0
        warped2 = skimage.transform.warp(img2, tps)
    
        morphed = warped1 * alpha + warped2 * (1.0 - alpha)
        
        base1 = os.path.basename(paths[idx0])
        filename1 = os.path.splitext(base1)[0]

        base2 = os.path.basename(paths[idx1])
        filename2 = os.path.splitext(base2)[0]

        dst_path = os.path.join(OUTPUT_FOLDER, '%s_%s.png' % (filename1, filename2))
    
        morphed = np.clip(morphed * 255, 0, 255)
        morphed = morphed.astype(np.uint8)
        cv2.imwrite(dst_path, morphed)
        print('save %s' % dst_path)

if __name__ == '__main__':
    main()
