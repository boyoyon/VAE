"""
ワイルドカードで指定した顔画像群からモーフィングした画像群を生成する。
mediapipeを使用するので、installしていない場合は、pip install mediapipe を実行する必要がある。
カレントディレクトリに画像群を出力するので、出力して構わない場所で実行すること。

python FaceMorph_wildcard.py (顔画像群へのワイルドカード)　[(分割数：default 2)]

例) python FaceMorph_wildcard.py *.png

"""

import cv2, glob, itertools, os, sys
import numpy as np

os.environ['TF_ENABLE_ONEDNN_OPTS']='0'

import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

SIZE = 256

ESC = 27

# 何分割でモーフィングを実行するかを指定する
# 2 ⇒ 50%
nrDivs = 2

# 背景もモーフィングする場合はこちら
DEF_TRIANGLES = 'def_triangles2.txt'
NUM_DIVS = 20

# 顔だけをモーフィングする場合はこちら
#DEF_TRIANGLES = 'def_triangles.txt'

# 顔の特徴点で三角形で構成した、頂点インデックスのリストを読み込む
def get_triangles_list():

    triangles_list = []

    with open(os.path.join(os.path.dirname(__file__), DEF_TRIANGLES), mode='r') as f:
        lines = f.read().split('\n')
        for line in lines:
            data = line.split(' ')
            if len(data) == 3:
                triangles_list.append((int(data[0]), int(data[1]), int(data[2])))
    
    return triangles_list

# mediapipeでfacemeshを取得する
def get_facemesh(image):

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5) as face_mesh:
        image = image

        height, width = image.shape[:2]

        # Convert the BGR image to RGB before processing.
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        point2d = []

        # face meshが取得されたら
        if results.multi_face_landmarks:

            # 1個目だけを処理する
            face_landmarks = results.multi_face_landmarks[0]

            for landmark in face_landmarks.landmark:
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                point2d.append((x, y))

            # 画像の周囲にkeypointを追加する

            for i in range(NUM_DIVS+1):
                x = int(i * (width - 1) / NUM_DIVS)
                point2d.append((x, 0))
                point2d.append((x, height - 1))

            for i in range(1, NUM_DIVS):
                y = int(i * (height - 1) / NUM_DIVS)
                point2d.append((0, y))
                point2d.append((width - 1, y))

        return point2d

# アフィン変換行列を求め、パッチ画像にアフィン変換を適用する
def applyAffineTransform(src, srcTri, dstTri, size) :
    
    # 2つの三角形座標を与えてアフィン変換行列を求める
    warpMat = cv2.getAffineTransform( np.float32(srcTri), np.float32(dstTri) )
    
    # パッチ画像にアフィン変換を適用する
    if src.shape[0] == 0 or src.shape[1] == 0:
        return None

    dst = cv2.warpAffine( src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101 )

    return dst

# 1つの三角形対に対してモーフィングを実行する
def morphTriangle(img1, img2, img, t1, t2, alpha) :

    # 補間した三角形を求める
    t = []
    for i in range(len(t1)):
        x = ( 1 - alpha ) * t1[i][0] + alpha * t2[i][0]
        y = ( 1 - alpha ) * t1[i][1] + alpha * t2[i][1]
        t.append((x,y))

    t1 = np.array(t1)
    t2 = np.array(t2)
    t  = np.array(t)

    # 各三角形の外接矩形を求める

    r1 = cv2.boundingRect(np.float32([t1]))
    left1   = r1[0]
    top1    = r1[1]
    right1  = r1[0] + r1[2]
    bottom1 = r1[1] + r1[3]

    r2 = cv2.boundingRect(np.float32([t2]))
    left2   = r2[0]
    top2    = r2[1]
    right2  = r2[0] + r2[2]
    bottom2 = r2[1] + r2[3]

    r = cv2.boundingRect(np.float32([t]))
    left    = r[0]
    top     = r[1]
    right   = r[0] + r[2]
    bottom  = r[1] + r[3]
    width   = r[2]
    height  = r[3]

    # 外接矩形の左上を原点とした三角形頂点座標を求める
    t1Rect = []
    t2Rect = []
    tRect = []

    for i in range(3):
        tRect.append(((t[i][0] - left), (t[i][1] - top)))
        t1Rect.append(((t1[i][0] - left1), (t1[i][1] - top1)))
        t2Rect.append(((t2[i][0] - left2), (t2[i][1] - top2)))

    # 三角形を塗りつぶしてマスクを作成する
    mask = np.zeros((height, width, 3), dtype = np.float32)
    cv2.fillConvexPoly(mask, np.int32(tRect), (1.0, 1.0, 1.0), 16, 0);

    # 外接矩形を画像として抽出する
    img1Rect = img1[top1:bottom1, left1:right1]
    img2Rect = img2[top2:bottom2, left2:right2]

    size = (width, height)
    warpImage1 = applyAffineTransform(img1Rect, t1Rect, tRect, size)
    warpImage2 = applyAffineTransform(img2Rect, t2Rect, tRect, size)

    #print(img.shape)
    #print(bottom, right, top, left)

    if warpImage1 is None or warpImage2 is None:
        #print('warpImage1 and/or warpImage2 is None')
        return #img = (img1 + img2) / 2


    else:
        # 外接矩形パッチにαブレンディングを適用
        imgRect = (1.0 - alpha) * warpImage1 + alpha * warpImage2

        # 外接矩形パッチにαブレンディングした三角形領域をコピーする
        if bottom > img.shape[0] or right > img.shape[1] or top < 0 or left < 0:
            img = imgRect.copy()
            return

        img[top:bottom, left:right] = img[top:bottom, left:right] * ( 1 - mask ) + imgRect * mask


def main():

    argv = sys.argv
    argc = len(argv)

    if argc < 2:
        print('%s morphs between two images' % argv[0])
        print('[usage] python %s <wildcard for images> [<No. of divisions>]' % argv[0])
        quit()

    paths = glob.glob(argv[1])
    nrData = len(paths)

    if argc > 2:
        nrDIvs = int(argv[2])

    facemeshs = []
    images = []
    for i, path in enumerate(paths):
        print('pre-processing %d/%d' % (i+1, nrData))
        img = cv2.imread(path)
        img = cv2.resize(img, (SIZE, SIZE))
        facemeshs.append(get_facemesh(img))
        img = img.astype(np.float32)
        images.append(img)

    triangles_list = get_triangles_list()
    
    alphas = np.linspace(0, 1, nrDivs + 1)[1:-1]

    pairs = itertools.combinations(np.arange(nrData), 2)

    for pair in pairs:

        idxA = pair[0]
        idxB = pair[1]

        baseA = os.path.basename(paths[idxA])
        filenameA = os.path.splitext(baseA)[0]
    
        baseB = os.path.basename(paths[idxB])
        filenameB = os.path.splitext(baseB)[0]
    
        # Read images
        imgA = images[idxA]
        imgB = images[idxB]

        facemeshA = facemeshs[idxA]
        facemeshB = facemeshs[idxB]
    
        # Allocate space for final output
        #imgMorph = np.zeros(imgA.shape, dtype = imgA.dtype)
        imgMorph = (imgA + imgB) / 2
   
        for alpha in alphas:
       
            ALPHA = int(alpha * 100)
            print('Morphing %s and %s with alpha:%d' % (paths[idxA], paths[idxB], ALPHA))

            for triangle in triangles_list:
        
                idx0 = triangle[0]
                idx1 = triangle[1]
                idx2 = triangle[2]
        
                tA = [facemeshA[idx0], facemeshA[idx1], facemeshA[idx2]]
                tB = [facemeshB[idx0], facemeshB[idx1], facemeshB[idx2]]
        
                # Morph one triangle at a time.
                morphTriangle(imgA, imgB, imgMorph, tA, tB, alpha)
    
            cv2.imwrite('%s_%s_%d.png' % (filenameA, filenameB, ALPHA), np.uint8(imgMorph))
    
    cv2.destroyAllWindows()

if __name__ == '__main__' :
    main()
