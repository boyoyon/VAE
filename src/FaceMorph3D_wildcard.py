import glfw
import numpy as np
from PIL import Image
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
import mediapipe as mp
import cv2, glob, itertools, os, sys

mp_face_mesh = mp.solutions.face_mesh

#奥行き補正用インデックス
RightEyeCover = 470
RightEyePupilTop = 159
RightEyePupilLeft = 469
RightEyePupilCenter = 468
RightEyePupilRight = 471
RightEyePupilBottom = 145
RightEyelidOver = 27
RightEyelidUpperLeft = 158
RightEyelidLowerLeft = 153
RightEyelidUpperCenter = 159
RightEyelidLowerCenter = 472
RightEyelidUpperRight = 160
RightEyelidLowerRight = 144

LeftEyeCover = 475
LeftEyePupilTop = 385
LeftEyePupilLeft = 474
LeftEyePupilCenter = 473
LeftEyePupilRight = 476
LeftEyePupilBottom = 374
LeftEyelidOver = 258
LeftEyelidUpperLeft = 386
LeftEyelidLowerLeft = 373
LeftEyelidUpperCenter = 385
LeftEyelidLowerCenter = 477
LeftEyelidUpperRight = 384
LeftEyelidLowerRight = 381

"""
RightContour1 = 162
RightContour2 = 127
RightContour3 = 234
RightContour4 = 227
RightContour5 =  93
RightContour6 = 137
RightContour7 = 132
RightContour8 = 177
RightContour9 = 215
RightContour10 =  58
RightContour11 = 172
RightContour12 = 136
"""

#画像幅をALIGNピクセルの倍数にcropする
ALIGN = 4

# マウスドラッグ中かどうか
isDragging = False

# マウスのクリック位置
oldPos = [0, 0]
newPos = [0, 0]

# 操作の種類
MODE_NONE = 0x00
MODE_TRANSLATE = 0x01
MODE_ROTATE = 0x02
MODE_SCALE = 0x04

# マウス移動量と回転、平行移動の倍率
ROTATE_SCALE = 10.0
TRANSLATE_SCALE = 500.0

# 座標変換のための変数
Mode = MODE_NONE
RotMat = TransMat = ScaleMat = None
Scale = 1.0

# スキャンコード定義
SCANCODE_LEFT  = 331
SCANCODE_RIGHT = 333
SCANCODE_UP    = 328
SCANCODE_DOWN  = 336

# キーコード定義
KEY_I = 73
KEY_S = 83

# 方位角、仰角
AZIMUTH = dAZIMUTH = 0.0
ELEVATION = dELEVATION = 0.0

# FaceMeshをドロネー分解した三角形の頂点インデックスリスト
DEF_TRIANGLES = 'def_triangles2.txt'
NUM_DIVS = 20

# モデル位置
ModelPos = [0.0, 0.0]

# テクスチャー画像
textureImage = None

# Facemesh
positions = []
texcoords = []
faces = []

WIN_WIDTH = 600  # ウィンドウの幅 / Window width
WIN_HEIGHT = 800  # ウィンドウの高さ / Window height
WIN_TITLE = "3D Face"  # ウィンドウのタイトル / Window title

TEX_FILE = None
textureId = 0

idxModel = None

frameNo = 1

fInertia = False

zScale = 1.0

# 顔の特徴点で三角形で構成した、頂点インデックスのリストを読み込む
def setup_faces():

    global faces

    with open(os.path.join(os.path.dirname(__file__), DEF_TRIANGLES), mode='r') as f:
        lines = f.read().split('\n')
        for line in lines:
            data = line.split(' ')
            if len(data) == 3:
                faces.append((int(data[0]), int(data[2]), int(data[1])))
    
# mediapipeでfacemeshを取得する
def getPos(image):

    positions = []
    texcoords = []

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5) as face_mesh:
        image = image

        height, width = image.shape[:2]

        aspect = width / height

        # BGR-->RGB変換してMediapipe Facemsh処理に渡す
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # face meshが取得されたら
        if results.multi_face_landmarks:

            # 1個目だけを処理する
            face_landmarks = results.multi_face_landmarks[0]

            for landmark in face_landmarks.landmark:
                x = landmark.x
                y = landmark.y
                z = landmark.z * zScale
                texcoords.append((x, y))
                positions.append([(x - 0.5) * aspect, y - 0.5, z])

            # 画像の周囲にkeypointを追加する

            p = np.array(positions)
            maxZ = np.max(p[:,2])

            for i in range(NUM_DIVS+1):
                x = i / NUM_DIVS
                texcoords.append((x, 0.0))
                texcoords.append((x, 1.0))
                positions.append([(x - 0.5) * aspect, -0.5, 0.0])
                positions.append([(x - 0.5) * aspect, 0.5, maxZ])

            for i in range(1, NUM_DIVS):
                y = i / NUM_DIVS
                z = i / NUM_DIVS * maxZ
                texcoords.append((0.0, y))
                texcoords.append((1.0, y))
                positions.append([-0.5 * aspect, y - 0.5, z])
                positions.append([0.5 * aspect, y - 0.5, z])

    #奥行き補正(まぶたの引っ込み）
    dep1 = positions[RightEyePupilTop][2]
    dep2 = positions[RightEyelidOver][2]
    positions[RightEyeCover][2] = np.mean((dep1, dep2))
    dep1 = positions[LeftEyePupilTop][2]
    dep2 = positions[LeftEyelidOver][2]
    positions[LeftEyeCover][2] = np.mean((dep1, dep2))


    # 奥行き補正(瞳孔の下のひっこみ)
    positions[RightEyePupilBottom][2] = positions[RightEyelidLowerCenter][2]
    positions[LeftEyePupilBottom][2] = positions[LeftEyelidLowerCenter][2]

    #奥行き補正(瞳孔のひっこみ）
    dep1 = positions[RightEyelidUpperLeft][2]
    dep2 = positions[RightEyelidLowerLeft][2]

    positions[RightEyePupilLeft][2] = np.min((dep1, dep2)) - np.abs(dep1 - dep2) / 16

    dep1 = positions[RightEyelidUpperCenter][2]
    dep2 = positions[RightEyelidLowerCenter][2]

    positions[RightEyePupilCenter][2] = np.min((dep1, dep2)) - np.abs(dep1 - dep2) / 8

    dep1 = positions[RightEyelidUpperRight][2]
    dep2 = positions[RightEyelidLowerRight][2]

    positions[RightEyePupilRight][2] = np.min((dep1, dep2)) - np.abs(dep1 - dep2) / 16

    dep1 = positions[LeftEyelidUpperLeft][2]
    dep2 = positions[LeftEyelidLowerLeft][2]

    positions[LeftEyePupilLeft][2] = np.min((dep1, dep2)) - np.abs(dep1 - dep2) / 16

    dep1 = positions[LeftEyelidUpperCenter][2]
    dep2 = positions[LeftEyelidLowerCenter][2]

    positions[LeftEyePupilCenter][2] = np.min((dep1, dep2)) - np.abs(dep1 - dep2) / 8

    dep1 = positions[LeftEyelidUpperRight][2]
    dep2 = positions[LeftEyelidLowerRight][2]

    positions[LeftEyePupilRight][2] = np.min((dep1, dep2)) - np.abs(dep1 - dep2) / 16

    #奥行き補正(頬輪郭ギザギザ）

    """
    dep1 = positions[RightContour1][2]
    dep2 = positions[RightContour3][2]
    positions[RightContour2][2] = np.mean((dep1, dep2))

    dep1 = positions[RightContour5][2]
    positions[RightContour4][2] = np.mean((dep1, dep2))

    dep1 = positions[RightContour5][2]
    dep2 = positions[RightContour7][2]
    positions[RightContour6][2] = np.mean((dep1, dep2))

    positions[RightContour9][2] = positions[RightContour8][2]
    """

    return positions, texcoords

def bindTexture(image):

    global idxModel

    H, W = image.shape[:2]
    glBindTexture(GL_TEXTURE_2D, textureId)
    gluBuild2DMipmaps(GL_TEXTURE_2D, GL_RGB8, W, H, GL_RGB, GL_UNSIGNED_BYTE, image.tobytes())
    glBindTexture(GL_TEXTURE_2D, 0)

    glDeleteLists(idxModel, 1)
    idxModel = glGenLists(1)

    glNewList(idxModel, GL_COMPILE)

    glBegin(GL_TRIANGLES)

    for i in range(len(faces)):

        idx0 = faces[i][0]
        idx1 = faces[i][1]
        idx2 = faces[i][2]

        glTexCoord2fv(texcoords[idx0])
        glVertex3fv(positions[idx0])

        glTexCoord2fv(texcoords[idx1])
        glVertex3fv(positions[idx1])

        glTexCoord2fv(texcoords[idx2])
        glVertex3fv(positions[idx2])

    glEnd()

    glEndList()

# OpenGLの初期化関数
def initializeGL():
    global textureId, idxModel

    # 背景色の設定 (黒)
    glClearColor(0.0, 0.0, 0.0, 1.0)

    # 深度テストの有効化
    glEnable(GL_DEPTH_TEST)

    glEnable(GL_CULL_FACE)

    # テクスチャの有効化
    glEnable(GL_TEXTURE_2D)

    # テクスチャの設定

    image = textureImage

    texHeight, texWidth, _ = image.shape

    # テクスチャの生成と有効化
    textureId = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, textureId)

    gluBuild2DMipmaps(GL_TEXTURE_2D, GL_RGB8, texWidth, texHeight, GL_RGB, GL_UNSIGNED_BYTE, image.tobytes())

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)

    # テクスチャ境界の折り返し設定
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)

    # テクスチャの無効化
    glBindTexture(GL_TEXTURE_2D, 0)

    idxModel = glGenLists(1)

    glNewList(idxModel, GL_COMPILE)

    glBegin(GL_TRIANGLES)

    for i in range(len(faces)):

        idx0 = faces[i][0]
        idx1 = faces[i][1]
        idx2 = faces[i][2]

        glTexCoord2fv(texcoords[idx0])
        glVertex3fv(positions[idx0])

        glTexCoord2fv(texcoords[idx1])
        glVertex3fv(positions[idx1])

        glTexCoord2fv(texcoords[idx2])
        glVertex3fv(positions[idx2])

    glEnd()

    glEndList()

# OpenGLの描画関数
def paintGL():

    if WIN_HEIGHT:

        # 背景色と深度値のクリア
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    
        # 投影変換行列
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(10.0, WIN_WIDTH / WIN_HEIGHT, 1.0, 100.0)
    
        # モデルビュー行列
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        gluLookAt(0.0, 0.0, -5.0,   # 視点の位置
            0.0, 0.0, 0.0,   # 見ている先
            0.0, -1.0, 0.0)  # 視界の上方向
    
        # 平面の描画
        glBindTexture(GL_TEXTURE_2D, textureId)  # テクスチャの有効化
    
        glPushMatrix()
        glScalef(Scale, Scale, Scale)
        glTranslatef(ModelPos[0], ModelPos[1], 0.0)
        glRotatef(ELEVATION, 1.0, 0.0, 0.0)
        glRotatef(AZIMUTH, 0.0, 1.0, 0.0)
        glCallList(idxModel)
        glPopMatrix()
    
        glBindTexture(GL_TEXTURE_2D, 0)  # テクスチャの無効化


# ウィンドウサイズ変更のコールバック関数
def resizeGL(window, width, height):
    global WIN_WIDTH, WIN_HEIGHT

    # ユーザ管理のウィンドウサイズを変更
    WIN_WIDTH = width
    WIN_HEIGHT = height

    # GLFW管理のウィンドウサイズを変更
    glfw.set_window_size(window, WIN_WIDTH, WIN_HEIGHT)

    # 実際のウィンドウサイズ (ピクセル数) を取得
    renderBufferWidth, renderBufferHeight = glfw.get_framebuffer_size(window)

    # ビューポート変換の更新
    glViewport(0, 0, renderBufferWidth, renderBufferHeight)

# アニメーションのためのアップデート
def animate():
    global AZIMUTH, ELEVATION

    if fInertia and not isDragging:
        AZIMUTH -= dAZIMUTH
        ELEVATION += dELEVATION

def save_screen(window):
    global frameNo

    width = WIN_WIDTH
    height = WIN_HEIGHT

    if width % 4 != 0:
        width = width // 4 * 4
        resizeGL(window, width, height)

    glReadBuffer(GL_FRONT)
    screen_shot = np.zeros((height, width, 3), np.uint8)
    glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, screen_shot.data)
    screen_shot = cv2.flip(screen_shot, 0) 
    screen_shot = cv2.cvtColor(screen_shot, cv2.COLOR_RGB2BGR)

    return screen_shot

# キーボードの押し離しを扱うコールバック関数
def keyboardEvent(window, key, scancode, action, mods):
    global AZIMUTH, ELEVATION, fInertia, dAZIMUTH, dELEVATION

    # 矢印キー操作

    if scancode == SCANCODE_LEFT:
        dAZIMUTH = -0.1
        AZIMUTH -= dAZIMUTH * 10

    if scancode == SCANCODE_RIGHT:
        dAZIMUTH = 0.1
        AZIMUTH -= dAZIMUTH * 10

    if scancode == SCANCODE_DOWN:
        dELEVATION = 0.1
        ELEVATION += dELEVATION * 10

    if scancode == SCANCODE_UP:
        dELEVATION = -0.1
        ELEVATION += dELEVATION * 10

    if action == 1: #press key
    
        if key == KEY_S:
            save_screen(window)

        if key == KEY_I:
            fInertia = not fInertia

# マウスのクリックを処理するコールバック関数
def mouseEvent(window, button, action, mods):
    global isDragging, newPos, oldPos, Mode, fInertia

    # クリックしたボタンで処理を切り替える
    if button == glfw.MOUSE_BUTTON_LEFT:
        Mode = MODE_ROTATE
    elif button == glfw.MOUSE_BUTTON_MIDDLE:
        Mode = MODE_SCALE
        if action == 1:
            fInertia = not fInertia
    elif button == glfw.MOUSE_BUTTON_RIGHT:
        Mode = MODE_TRANSLATE

    # クリックされた位置を取得
    px, py = glfw.get_cursor_pos(window)

    # マウスドラッグの状態を更新
    if action == glfw.PRESS:
        if not isDragging:
            isDragging = True
            oldPos = [px, py]
            newPos = [px, py]
    else:
        isDragging = False
        oldPos = [0, 0]
        newPos = [0, 0]

# マウスの動きを処理するコールバック関数
def motionEvent(window, xpos, ypos):
    global isDragging, newPos, oldPos, AZIMUTH, ELEVATION, ModelPos, dAZIMUTH, dELEVATION

    if isDragging:
        # マウスの現在位置を更新
        newPos = [xpos, ypos]

        # マウスがあまり動いていない時は処理をしない
        dx = newPos[0] - oldPos[0]
        dy = newPos[1] - oldPos[1]
        length = dx * dx + dy * dy
        if length < 2.0 * 2.0:
            return
        else:
            if Mode == MODE_ROTATE:
                dAZIMUTH = (xpos - oldPos[0]) / ROTATE_SCALE
                AZIMUTH -= dAZIMUTH
                dELEVATION = (ypos - oldPos[1]) / ROTATE_SCALE
                ELEVATION += dELEVATION
            elif Mode == MODE_TRANSLATE:
                ModelPos[0] += (xpos - oldPos[0]) / TRANSLATE_SCALE
                ModelPos[1] += (ypos - oldPos[1]) / TRANSLATE_SCALE

            oldPos = [xpos, ypos]

# マウスホイールを処理するコールバック関数
def wheelEvent(window, xoffset, yoffset):
    global Scale
    Scale += yoffset / 10.0

# 画像の横幅がALIGNピクセルの倍数になるようにクロップする
def prescale(image):
    height, width = image.shape[:2]

    if width % ALIGN != 0:
        WIDTH = width // ALIGN * ALIGN
        startX = (width - WIDTH) // 2
        endX = startX + WIDTH

        dst = np.empty((height, WIDTH, 3), np.uint8)
        dst = image[:, startX:endX]
        return dst

    else:
        return image

def save_ply(ply_filename):

    base = os.path.basename(ply_filename)
    filename, _ = os.path.splitext(base)
    #np.save('%s_pos.npy' % filename, np.array(positions))
    #np.save('%s_tex.npy' % filename, np.array(texcoords))
    #np.save('%s_faces.npy' % filename, np.array(faces))

    with open(ply_filename, mode='w') as f:

        line = 'ply\n'
        f.write(line)

        line = 'format ascii 1.0\n'
        f.write(line)

        line = 'element vertex %d\n' % len(positions)
        f.write(line)

        line = 'property float x\n'
        f.write(line)

        line = 'property float y\n'
        f.write(line)

        line = 'property float z\n'
        f.write(line)

        line = 'property float s\n'
        f.write(line)

        line = 'property float t\n'
        f.write(line)

        line = 'element face %d\n' % len(faces)
        f.write(line)

        line = 'property list uchar int vertex_indices\n'
        f.write(line)

        line = 'end_header\n'
        f.write(line)

        for i in range(len(positions)):
            line = '%f %f %f %f %f\n' % (
                    positions[i][0], positions[i][1], positions[i][2],
                    texcoords[i][0], 1.0 - texcoords[i][1])
            f.write(line)

        for i in range(len(faces)):
            idx0 = faces[i][0]
            idx1 = faces[i][1]
            idx2 = faces[i][2]
            
            line = '3 %d %d %d\n' % (idx0, idx1, idx2)
            #line = '3 %d %d %d\n' % (idx0, idx2, idx1)
            f.write(line)

def main():

    global positions, texcoords, textureImage, WIN_WIDTH, WIN_HEIGHT, Scale

    argv = sys.argv
    argc = len(argv)

    print('%s creates face model from the image' % argv[0])
    print('%s <wildcard for images> [<alpha1(float)> <alpha2(float)> ...]' % argv[0])
    
    if argc < 2:
        quit()

    Scale = 0.85
    
    setup_faces()

    # OpenGLを初期化する
    if glfw.init() == glfw.FALSE:
        raise Exception("Failed to initialize OpenGL")

    # Windowの作成
    window = glfw.create_window(WIN_WIDTH, WIN_HEIGHT, WIN_TITLE, None, None)
    if window is None:
        glfw.terminate()
        raise Exception("Failed to create Window")

    # OpenGLの描画対象にWindowを追加
    glfw.make_context_current(window)

    # ウィンドウのリサイズを扱う関数の登録
    glfw.set_window_size_callback(window, resizeGL)

    # キーボードのイベントを処理する関数を登録
    glfw.set_key_callback(window, keyboardEvent)

    # マウスのイベントを処理する関数を登録
    glfw.set_mouse_button_callback(window, mouseEvent)

    # マウスの動きを処理する関数を登録
    glfw.set_cursor_pos_callback(window, motionEvent)

    # マウスホイールを処理する関数を登録
    glfw.set_scroll_callback(window, wheelEvent)

    paths = glob.glob(argv[1])
    nrData = len(paths)

    alphas = [0.5]

    if argc > 2:
        alphas = []
        for i in range(2, argc):
            alphas.append(float(argv[i]))

    # pre-processing 

    filenames = []
    images = []
    pos = []
    tex = []

    for path in paths:
        print('preprocessing %s' % path)

        base = os.path.basename(path)
        filenames.append(os.path.splitext(base)[0])
        img = cv2.imread(path)
        p, t = getPos(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
        pos.append(p)
        tex.append(t)

    # morphing 

    pairs = itertools.combinations(np.arange(nrData), 2)

    fInitialized = False

    for pair in pairs:

        idx0 = pair[0]
        idx1 = pair[1]

        img0 = images[idx0]
        H0, W0 = img0.shape[:2]

        img1 = images[idx1]
        H1, W1 = img1.shape[:2]
    
        WIN_HEIGHT = (H0 + H1) // 2
        WIN_WIDTH = (W0 + W1) // 2
        WIN_WIDTH = WIN_WIDTH // 4 * 4

        resizeGL(window, WIN_WIDTH, WIN_HEIGHT)
        
        img0 = cv2.resize(img0, (WIN_WIDTH, WIN_HEIGHT))
        img1 = cv2.resize(img1, (WIN_WIDTH, WIN_HEIGHT))
        
        pos0 = pos[idx0]
        pos1 = pos[idx1]

        tex0 = tex[idx0]
        tex1 = tex[idx1]

        nrPos0 = len(pos0)
        nrPos1 = len(pos1)

        nrPos = np.min((nrPos0, nrPos1))

        for alpha in alphas:

            beta = 1.0 - alpha
            positions = []

            for i in range(nrPos):
                x = pos0[i][0] * alpha + pos1[i][0] * beta
                y = pos0[i][1] * alpha + pos1[i][1] * beta
                z = pos0[i][2] * alpha + pos1[i][2] * beta
        
                positions.append((x, y, z))
        
            for i in range(2):
        
                if i == 0:
                    textureImage = img0
                    texcoords = tex0
                    if not fInitialized:
                        initializeGL()
                        fInitialized = True
                    else:
                        bindTexture(img0)
                else:
                    textureImage = img1
                    texcoords = tex1
                    bindTexture(img1)

                # 描画
                paintGL()
        
                # アニメーション
                animate()
        
                # 描画用バッファの切り替え
                glfw.swap_buffers(window)
                glfw.poll_events()
        
                if i == 0:
                    screen0 = save_screen(window)
                else: 
                    screen1 = save_screen(window)
        
            screen0 = screen0.astype(np.float32)
            screen1 = screen1.astype(np.float32)
        
            dst = screen0 * alpha + screen1 * beta
            dst = np.clip(dst, 0, 255)
            dst = dst.astype(np.uint8)
        
            ALPHA = int(alpha * 100)
        
            dst_path = ('%s_%s_%d.png' % (filenames[idx0], filenames[idx1], ALPHA))
            cv2.imwrite(dst_path, dst)
            print('save %s' % dst_path)

    # 後処理
    glfw.destroy_window(window)
    glfw.terminate()


if __name__ == "__main__":
    main()
