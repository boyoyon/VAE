"""
画像から顔を検出→顔の傾き補正→顔周辺を切り出し、します。
・python correctFaceRoll.py (顔の写った画像群へのワイルドカード)
・mediapipeを使っているのでインストールしていない場合は、
　installしてください。
   pip install mediapipe
・カレントディレクトリに(連番).pngを出力するので
　出力してかまわないディレクトで実行してください。

"""
import cv2, glob, os, sys
import numpy as np

os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

# mediapipeでfacemeshを取得する
def get_face_landmarks(image):

    positions = []

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5) as face_mesh:
        image = image

        height, width = image.shape[:2]

        # BGR-->RGB変換してMediapipe Facemsh処理に渡す
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # face meshが取得されたら
        if results.multi_face_landmarks:

            # 1個目だけを処理する
            face_landmarks = results.multi_face_landmarks[0]

            for landmark in face_landmarks.landmark:
                x = landmark.x
                y = landmark.y
                #z = landmark.z
                positions.append((int(x* width), int(y * height)))

    return np.array(positions)

def main():

    MARGIN = 10 # percent
    fFlip = False

    no = 1

    argv = sys.argv
    argc = len(argv)

    print('%s crops face area' % argv[0])
    print('[usage] python %s <wildcard for images>' % argv[0])
    
    if argc < 2:
        quit()

    paths = glob.glob(argv[1])
    
    for path in paths:

        src = cv2.imread(path, cv2.IMREAD_COLOR)
        H, W = src.shape[:2]
    
        pos1 = get_face_landmarks(src)

        try:
            center = (np.mean(pos1[:,0]), np.mean(pos1[:,1]))
        except IndexError:
            print(path, pos1)
            continue

        xR, yR = pos1[468]
        xL, yL = pos1[473]
        angle = np.rad2deg(np.arctan2(xL - xR, yL - yR)) * -1 + 90

        for a in [0]:
        #for a in [-20, -15, -10, -5, 0, 5, 10, 15, 20]:

            #アフィン変換行列を作成する
            rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=angle + a, scale=1)
    
            #アフィン変換行列を画像に適用する
            dst1 = cv2.warpAffine(src=src, M=rotate_matrix, dsize=(W, H))
    
            pos2 = get_face_landmarks(dst1)
        
            try:
                minX = np.min(pos2[:,0])
            except IndexError:
                print('Error: ',path)
                print(pos2)
                continue

            try:
                maxX = np.max(pos2[:,0])
            except IndexError:
                print('Error: ',path)
                print(pos2)
                continue
            
            try:
                minY = np.min(pos2[:,1])
            except IndexError:
                print('Error: ',path)
                print(pos2)
                continue
            
            try:
                maxY = np.max(pos2[:,1])
            except IndexError:
                print('Error: ',path)
                print(pos2)
                continue
    
            marginX = (maxX - minX) * MARGIN // 100
            marginY = (maxY - minY) * MARGIN // 100
    
            left = np.max((0, minX - marginX))
            right = np.min((W - 1, maxX + marginX))
            top = np.max((0, minY - marginY))
            bottom = np.min((H - 1, maxY + marginY))
    
            dst = dst1[top:bottom, left:right]
    
            dst_path = '%04d.png' % no
            while os.path.exists(dst_path):
                no += 1
                dst_path = '%04d.png' % no

            cv2.imwrite(dst_path, dst)
            print('save %s' % dst_path)
            no += 1

            if fFlip:

                dst_path = '%04d.png' % no
                while os.path.exists(dst_path):
                    no += 1
                    dst_path = '%04d.png' % no

                dst2 = cv2.flip(dst, 1)

                cv2.imwrite(dst_path, dst2)
                print('save %s' % dst_path)
                no += 1

if __name__ == "__main__":
    main()
