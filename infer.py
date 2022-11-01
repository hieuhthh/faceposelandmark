import tensorflow as tf
import cv2
import numpy as np
import os
import shutil
import math

import sys 
sys.path.append('datatool')
from fmd.mark_dataset.util import draw_marks

try:
    shutil.rmtree('infer')
except:
    pass

try:
    os.mkdir('infer')
except:
    pass

weight = '/home/lap14880/hieunmt/face_landmark/save_model/best_model_face_pose_esti_192_68_noaug.h5'

with tf.device('/cpu:0'):
    model = tf.keras.models.load_model(weight)
    model.summary()

def preprocess_img(path, target_size):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = tf.image.resize(img, target_size)
    img = tf.cast(img, tf.float32) / 255.0
    return img

im_size = 192
img_size = (im_size, im_size)

img_path = '/home/lap14880/hieunmt/face_landmark/sample/4.jpg'

img = preprocess_img(img_path, img_size)

img_batch = tf.expand_dims(img, 0)
pred = model.predict(img_batch)

model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Mouth left corner
            (150.0, -150.0, -125.0)      # Mouth right corner
        ])
focal_length = img_size[1]
camera_center = (img_size[1] / 2, img_size[0] / 2)
camera_matrix = np.array(
            [[focal_length, 0, camera_center[0]],
             [0, focal_length, camera_center[1]],
             [0, 0, 1]], dtype="double")
dist_coeffs = np.zeros((4, 1))

marks = np.array(pred[0]) * im_size

img_draw = np.array(img) * 255
draw_marks(img_draw, marks)
cv2.imwrite(f"/home/lap14880/hieunmt/face_landmark/infer/infer_landmark_full.jpg", img_draw)  

marks = marks[[30,      # Nose tip
                8,      # Chin
                36,     # Left eye left corner
                45,     # Right eye right corne
                48,     # Left Mouth corner
                54]     # Right mouth corner
                ]
# print(marks)

img_draw = np.array(img) * 255
draw_marks(img_draw, marks)
cv2.imwrite(f"/home/lap14880/hieunmt/face_landmark/infer/infer_landmark_small.jpg", img_draw)  

img_draw = np.array(img) * 255

(success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, marks, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_UPNP)

(nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)

# draw direction where the person’s face is pointing
p1 = (int(marks[0][0]), int(marks[0][1]))
p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
cv2.line(img_draw, p1, p2, (255,255,255), 2)

def draw_annotation_box(img, rotation_vector, translation_vector, camera_matrix, color=(255, 255, 0), line_width=2):
    """Draw a 3D box as annotation of pose"""
    point_3d = []
    dist_coeffs = np.zeros((4,1))
    rear_size = 1
    rear_depth = 0
    point_3d.append((-rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, -rear_size, rear_depth))

    front_size = img.shape[1]
    front_depth = front_size*2
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d.append((-front_size, front_size, front_depth))
    point_3d.append((front_size, front_size, front_depth))
    point_3d.append((front_size, -front_size, front_depth))
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d = np.array(point_3d, dtype=np.float).reshape(-1, 3)

    # Map to 2d img points
    (point_2d, _) = cv2.projectPoints(point_3d,
                                      rotation_vector,
                                      translation_vector,
                                      camera_matrix,
                                      dist_coeffs)
    point_2d = np.int32(point_2d.reshape(-1, 2))
    

    # # Draw all the lines
    # cv2.polylines(img, [point_2d], True, color, line_width, cv2.LINE_AA)
    k = (point_2d[5] + point_2d[8])//2
    # cv2.line(img, tuple(point_2d[1]), tuple(
    #     point_2d[6]), color, line_width, cv2.LINE_AA)
    # cv2.line(img, tuple(point_2d[2]), tuple(
    #     point_2d[7]), color, line_width, cv2.LINE_AA)
    # cv2.line(img, tuple(point_2d[3]), tuple(
    #     point_2d[8]), color, line_width, cv2.LINE_AA)
    
    return(point_2d[2], k)

x1, x2 = draw_annotation_box(img_draw, rotation_vector, translation_vector, camera_matrix)

try:
    m = (p2[1] - p1[1])/(p2[0] - p1[0])
    ang1 = int(math.degrees(math.atan(m)))
except:
    ang1 = 90
    
try:
    m = (x2[1] - x1[1])/(x2[0] - x1[0])
    ang2 = int(math.degrees(math.atan(-1/m)))
except:
    ang2 = 90

# ang1: vertical
# ang2: horizon

draw_marks(img_draw, marks)
font = cv2.FONT_HERSHEY_SIMPLEX
print(marks)
print(ang1)
print(ang2)
cv2.putText(img_draw, str(ang1), (0,30), font, 1, (128, 255, 255), 3)
cv2.putText(img_draw, str(ang2), (0,60), font, 1, (255, 255, 128), 3)

cv2.imwrite(f"/home/lap14880/hieunmt/face_landmark/infer/infer_landmark.jpg", img_draw)  