import tensorflow as tf
import cv2
import numpy as np
import os
import shutil
import math

class InferFacePose:
    def __init__(self, im_size, weight):
        self.im_size = im_size
        self.img_size = (im_size, im_size)
        self.weight = weight
        self.model = self.get_model()
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Mouth left corner
            (150.0, -150.0, -125.0)      # Mouth right corner
        ])
        self.focal_length = self.img_size[1]
        self.camera_center = (self.img_size[1] / 2, self.img_size[0] / 2)
        self.camera_matrix = np.array(
                    [[self.focal_length, 0, self.camera_center[0]],
                    [0, self.focal_length, self.camera_center[1]],
                    [0, 0, 1]], dtype="float")
        self.dist_coeffs = np.zeros((4, 1))
        self.landmarks_68_3D = np.array( [
                    [-73.393523  , -29.801432   , 47.667532   ],
                    [-72.775014  , -10.949766   , 45.909403   ],
                    [-70.533638  , 7.929818     , 44.842580   ],
                    [-66.850058  , 26.074280    , 43.141114   ],
                    [-59.790187  , 42.564390    , 38.635298   ],
                    [-48.368973  , 56.481080    , 30.750622   ],
                    [-34.121101  , 67.246992    , 18.456453   ],
                    [-17.875411  , 75.056892    , 3.609035    ],
                    [0.098749    , 77.061286    , -0.881698   ],
                    [17.477031   , 74.758448    , 5.181201    ],
                    [32.648966   , 66.929021    , 19.176563   ],
                    [46.372358   , 56.311389    , 30.770570   ],
                    [57.343480   , 42.419126    , 37.628629   ],
                    [64.388482   , 25.455880    , 40.886309   ],
                    [68.212038   , 6.990805     , 42.281449   ],
                    [70.486405   , -11.666193   , 44.142567   ],
                    [71.375822   , -30.365191   , 47.140426   ],
                    [-61.119406  , -49.361602   , 14.254422   ],
                    [-51.287588  , -58.769795   , 7.268147    ],
                    [-37.804800  , -61.996155   , 0.442051    ],
                    [-24.022754  , -61.033399   , -6.606501   ],
                    [-11.635713  , -56.686759   , -11.967398  ],
                    [12.056636   , -57.391033   , -12.051204  ],
                    [25.106256   , -61.902186   , -7.315098   ],
                    [38.338588   , -62.777713   , -1.022953   ],
                    [51.191007   , -59.302347   , 5.349435    ],
                    [60.053851   , -50.190255   , 11.615746   ],
                    [0.653940    , -42.193790   , -13.380835  ],
                    [0.804809    , -30.993721   , -21.150853  ],
                    [0.992204    , -19.944596   , -29.284036  ],
                    [1.226783    , -8.414541    , -36.948060  ],
                    [-14.772472  , 2.598255     , -20.132003  ],
                    [-7.180239   , 4.751589     , -23.536684  ],
                    [0.555920    , 6.562900     , -25.944448  ],
                    [8.272499    , 4.661005     , -23.695741  ],
                    [15.214351   , 2.643046     , -20.858157  ],
                    [-46.047290  , -37.471411   , 7.037989    ],
                    [-37.674688  , -42.730510   , 3.021217    ],
                    [-27.883856  , -42.711517   , 1.353629    ],
                    [-19.648268  , -36.754742   , -0.111088   ],
                    [-28.272965  , -35.134493   , -0.147273   ],
                    [-38.082418  , -34.919043   , 1.476612    ],
                    [19.265868   , -37.032306   , -0.665746   ],
                    [27.894191   , -43.342445   , 0.247660    ],
                    [37.437529   , -43.110822   , 1.696435    ],
                    [45.170805   , -38.086515   , 4.894163    ],
                    [38.196454   , -35.532024   , 0.282961    ],
                    [28.764989   , -35.484289   , -1.172675   ],
                    [-28.916267  , 28.612716    , -2.240310   ],
                    [-17.533194  , 22.172187    , -15.934335  ],
                    [-6.684590   , 19.029051    , -22.611355  ],
                    [0.381001    , 20.721118    , -23.748437  ],
                    [8.375443    , 19.035460    , -22.721995  ],
                    [18.876618   , 22.394109    , -15.610679  ],
                    [28.794412   , 28.079924    , -3.217393   ],
                    [19.057574   , 36.298248    , -14.987997  ],
                    [8.956375    , 39.634575    , -22.554245  ],
                    [0.381549    , 40.395647    , -23.591626  ],
                    [-7.428895   , 39.836405    , -22.406106  ],
                    [-18.160634  , 36.677899    , -15.121907  ],
                    [-24.377490  , 28.677771    , -4.785684   ],
                    [-6.897633   , 25.475976    , -20.893742  ],
                    [0.340663    , 26.014269    , -22.220479  ],
                    [8.444722    , 25.326198    , -21.025520  ],
                    [24.474473   , 28.323008    , -5.712776   ],
                    [8.449166    , 30.596216    , -20.671489  ],
                    [0.205322    , 31.408738    , -21.903670  ],
                    [-7.198266   , 30.844876    , -20.328022  ] ], dtype=np.float32)

    def preprocess_img(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = tf.image.resize(img, self.img_size)
        img = tf.cast(img, tf.float32) / 255.0
        return img

    def draw_marks(self, image, marks, mark_size=3, color=(0, 255, 0), line_width=-1):
        """Draw the marks in image.
        Args:
            image: the image on which to be drawn.
            marks: points coordinates in a numpy array.
            mark_size: the size of the marks.
            color: the color of the marks, in BGR format, ranges 0~255.
            line_width: the width of the mark's outline. Set to -1 to fill it.
        """
        # We are drawing in an image, this is a 2D situation.
        image_copy = image.copy()
        for point in marks:
            cv2.circle(image_copy, (int(point[0]), int(point[1])),
                    mark_size, color, line_width, cv2.LINE_AA)
        return image_copy

    def get_model(self):
        """
        can change strategy to gpu, gpus
        strategy = tf.distribute.MirroredStrategy()
        strategy = tf.distribute.get_strategy()
        """
        with tf.device('/cpu:0'):
            model = tf.keras.models.load_model(self.weight)
            model.summary()
        return model

    def predict_marks(self, img):
        img_batch = tf.expand_dims(img, 0)
        pred = self.model.predict(img_batch)
        marks = np.array(pred[0]) * self.im_size
        """
        return array (68, 2)
            68 keypoints (x, y)
        """
        return marks

    def get_3d_box(self, img, rotation_vector, translation_vector, camera_matrix, color=(255, 255, 0), line_width=2):
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

    def predict_angle(self, img):
        """
        return ang_vertical, ang_horizon
        angle (degree)
        ang_vertical: vertical angle
        ang_horizon: horizon angle
        """
        marks = self.predict_marks(img)
        
        marks = marks[[30,      # Nose tip
                        8,      # Chin
                        36,     # Left eye left corner
                        45,     # Right eye right corner
                        48,     # Left Mouth corner
                        54]     # Right mouth corner
                    ]
      
        eye_center = ((marks[2][0] + marks[3][0]) / 2, (marks[2][1] + marks[3][1]) / 2)
        dx = abs(marks[2][0] - marks[3][0])
        dy = abs(marks[2][1] - marks[3][1])
        ang_rot = np.arctan2(dy, dx)
        ang_rot = ang_rot * 180 / math.pi  

        (_, rotation_vector, translation_vector) = cv2.solvePnP(self.model_points, 
                                                                marks, 
                                                                self.camera_matrix, 
                                                                self.dist_coeffs, 
                                                                flags=cv2.SOLVEPNP_UPNP)

        (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), 
                                                         rotation_vector, 
                                                         translation_vector, 
                                                         self.camera_matrix, 
                                                         self.dist_coeffs)
        
        x1, x2 = self.get_3d_box(img, rotation_vector, translation_vector, self.camera_matrix)

        p1 = (int(marks[0][0]), int(marks[0][1]))
        p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

        try:
            m = (p2[1] - p1[1])/(p2[0] - p1[0])
            ang_vertical = int(math.degrees(math.atan(m)))
        except:
            ang_vertical = 90
            
        try:
            m = (x2[1] - x1[1])/(x2[0] - x1[0])
            ang_horizon = int(math.degrees(math.atan(-1/m)))
        except:
            ang_horizon = 90

        ang_vertical /= -2
        ang_horizon /= 2

        return ang_vertical, ang_horizon, ang_rot

    def predict_marks_from_path(self, path):
        img = self.preprocess_img(path)
        marks = self.predict_marks(img)
        return marks

    def predict_angle_from_path(self, path):
        img = self.preprocess_img(path)
        ang_vertical, ang_horizon, ang_rot = self.predict_angle(img)
        return ang_vertical, ang_horizon, ang_rot

if __name__ == '__main__':
    im_size = 192

    # weight = '/home/lap14880/hieunmt/face_landmark/save_model/best_model_face_pose_esti_192_68_noaug.h5'
    # weight = '/home/lap14880/hieunmt/face_landmark/save_model/best_model_face_pose_esti_192_68_aug.h5'
    # weight = '/home/lap14880/hieunmt/face_landmark/save_model/best_model_face_pose_esti_192_68_noaug_mid512.h5'
    # weight = '/home/lap14880/hieunmt/face_landmark/save_model/best_model_face_pose_esti_EfficientNetV1B1_192_68_4epoch.h5'
    weight = '/home/lap14880/hieunmt/face_landmark/save_model/best_model_face_pose_esti_EfficientNetV1B1_192_68_100epoch.h5'

    infer_face_pose = InferFacePose(im_size, weight)

    # img_path = '/home/lap14880/hieunmt/face_landmark/sample/1.jpg'
    img_path = '/home/lap14880/hieunmt/face_landmark/nhutsample/hh3.png'

    img = infer_face_pose.preprocess_img(img_path)
    marks = infer_face_pose.predict_marks(img)
    # print(marks)

    import time

    start = time.time()
    ang_vertical, ang_horizon, ang_rot = infer_face_pose.predict_angle(img)
    dur = time.time() - start
    print('time:', dur)
    
    # ang_vertical, ang_horizon, ang_rot = infer_face_pose.predict_angle_from_path(img_path)

    print('ang_vertical, ang_horizon, ang_rot:', ang_vertical, ang_horizon, ang_rot)

    try:
        shutil.rmtree('infer_e2e')
    except:
        pass

    try:
        os.mkdir('infer_e2e')
    except:
        pass

    img = np.array(img) * 255
    img_write = infer_face_pose.draw_marks(img, marks)
    cv2.imwrite(f"/home/lap14880/hieunmt/face_landmark/infer_e2e/infer_e2e_ori.jpg", img)  
    cv2.imwrite(f"/home/lap14880/hieunmt/face_landmark/infer_e2e/infer_e2e_landmark.jpg", img_write) 




