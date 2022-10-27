import numpy as np
import cv2

class MarkOperator(object):
    """Operator instances are used to transform the marks."""

    def __init__(self):
        pass

    def get_distance(self, mark1, mark2):
        """Calculate the distance between two marks."""
        return np.linalg.norm(mark2 - mark1)

    def get_angle(self, vector1, vector2, in_radian=False):
        """Return the angel between two vectors."""
        d = np.dot(vector1, vector2)
        cos_angle = d / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
        if cos_angle > 1.0:
            radian = 0
        elif cos_angle < -1.0:
            radian = np.pi
        else:
            radian = np.arccos(cos_angle)

        c = np.cross(vector1, vector2)
        if (c.ndim == 0 and c < 0) or (c.ndim == 1 and c[2] < 0):
            radian = 2*np.pi - radian

        return radian if in_radian is True else np.rad2deg(radian)

    def pad_to_3d(self, marks_2d, pad_value=-1):
        """Pad the 2D marks with zeros in z axis."""
        marks_3d = np.pad(marks_2d, ((0, 0), (0, 1)),
                          mode='constant', constant_values=pad_value)

        return marks_3d

    def get_center(self, marks):
        """Return the center point of the mark group."""
        x, y, z = (np.amax(marks, 0) + np.amin(marks, 0)) / 2

        return np.array([x, y, z])

    def get_height_width_depth(self, marks):
        """Return the height and width of the marks bounding box."""
        height, width, depth = np.amax(marks, 0) - np.amin(marks, 0)

        return height, width, depth

    def rotate(self, marks, radian, center=(0, 0)):
        """Rotate the marks around center by angle"""
        _points = marks[:, :2] - np.array(center, np.float)
        cos_angle = np.cos(-radian)
        sin_angle = np.sin(-radian)
        rotaion_matrix = np.array([[cos_angle, sin_angle],
                                   [-sin_angle, cos_angle]])
        marks[:, :2] = np.dot(_points, rotaion_matrix) + center

        return marks

    def flip_lr(self, marks, width):
        """Flip the marks in horizontal direction."""
        marks[:, 0] = width - marks[:, 0]

        # Reset the order of the marks. The HRNet authors had provided this
        # information in the official repository.
        num_marks = marks.shape[0]
        if num_marks == 98:     # WFLW
            mirrored_pairs = np.array([
                [0,  32], [1,  31], [2,  30], [3,  29], [4,  28], [5,  27],
                [6,  26], [7,  25], [8,  24], [9,  23], [10, 22], [11, 21],
                [12, 20], [13, 19], [14, 18], [15, 17], [33, 46], [34, 45],
                [35, 44], [36, 43], [37, 42], [38, 50], [39, 49], [40, 48],
                [41, 47], [60, 72], [61, 71], [62, 70], [63, 69], [64, 68],
                [65, 75], [66, 74], [67, 73], [55, 59], [56, 58], [76, 82],
                [77, 81], [78, 80], [87, 83], [86, 84], [88, 92], [89, 91],
                [95, 93], [96, 97]
            ])
        elif num_marks == 68:   # IBUG, etc.
            mirrored_pairs = np.array([
                [1,  17], [2,  16], [3,  15], [4,  14], [5,  13], [6,  12],
                [7,  11], [8,  10], [18, 27], [19, 26], [20, 25], [21, 24],
                [22, 23], [32, 36], [33, 35], [37, 46], [38, 45], [39, 44],
                [40, 43], [41, 48], [42, 47], [49, 55], [50, 54], [51, 53],
                [62, 64], [61, 65], [68, 66], [59, 57], [60, 56]]) - 1
        else:
            raise ValueError(
                "Number of points {} not supported, please check the dataset.".format(num_marks))

        tmp = marks[mirrored_pairs[:, 0]]
        marks[mirrored_pairs[:, 0]] = marks[mirrored_pairs[:, 1]]
        marks[mirrored_pairs[:, 1]] = tmp

        return marks

    def _generate_heatmap(self, heatmap_size, center_point, sigma):
        """Generating a heatmap with Gaussian distribution.
        Args:
            heatmap_size: a tuple containing the size of the heatmap.
            center_point: a tuple containing the center point of the distribution.
            sigma: how large area the distribution covers.
        Returns:
            a heatmap
        """
        def _generate_gaussian_map(sigma):
            """Generate gaussian distribution with center value equals to 1."""
            heat_range = 2 * sigma * 3 + 1
            xs = np.arange(0, heat_range, 1, np.float32)
            ys = xs[:, np.newaxis]
            x_core = y_core = heat_range // 2
            gaussian = np.exp(-((xs - x_core) ** 2 + (ys - y_core)
                                ** 2) / (2 * sigma ** 2))

            return gaussian

        # Check that any part of the gaussian is in-bounds
        map_height, map_width = heatmap_size
        x, y = int(center_point[0]), int(center_point[1])

        radius = sigma * 3
        x0, y0 = x - radius, y - radius
        x1, y1 = x + radius + 1, y + radius + 1

        # If the distribution is out of the map, return an empty map.
        if (x0 >= map_width or y0 >= map_height or x1 < 0 or y1 < 0):
            return np.zeros(heatmap_size)

        # Generate a Gaussian map.
        gaussian = _generate_gaussian_map(sigma)

        # Get the intersection area of the Gaussian map.
        x_gauss = max(0, -x0), min(x1, map_width) - x0
        y_gauss = max(0, -y0), min(y1, map_height) - y0
        gaussian = gaussian[y_gauss[0]: y_gauss[1], x_gauss[0]: x_gauss[1]]

        # Pad the Gaussian with zeros to get the heatmap.
        pad_width = np.max(
            [[0, 0, 0, 0], [y0, map_height-y1, x0, map_width-x1, ]], axis=0).reshape([2, 2])
        heatmap = np.pad(gaussian, pad_width, mode='constant')

        return heatmap

    def generate_heatmaps(self, norm_marks, map_size=(64, 64), sigma=3):
        """Generate heatmaps for all the marks."""
        maps = []
        width, height = map_size
        for norm_mark in norm_marks:
            x = width * norm_mark[0]
            y = height * norm_mark[1]
            heatmap = self._generate_heatmap(map_size, (x, y), sigma)
            maps.append(heatmap)

        return np.array(maps, dtype=np.float32)

def move_face_to_center(image, marks, mo):
    """This function will move the marked face to the image center.
    Args:
        image: image containing a marked face.
        marks: the face marks.
        mo: the mark operater.
    Returns:
        a same size image with marked face at center.
    """
    img_height, img_width, _ = image.shape
    face_center = mo.get_center(marks)[:2]
    translation_mat = np.array([[1, 0, img_width / 2 - face_center[0]],
                                [0, 1, img_height / 2 - face_center[1]]])
    image_translated = cv2.warpAffine(
        image, translation_mat, (img_width, img_height))

    translation_vector = np.array(
        [img_width / 2 - face_center[0],  img_height / 2 - face_center[1]])

    return image_translated, translation_vector


def rotate_to_vertical(image, sample, mo):
    """Rotate the image to make the face vertically aligned.
    Args:
        image: an image with face to be processed.
        sample: the dataset sample of the input image.
        mo: the mark operator.
    Returns:
        a same size image with aligned face.
    """
    img_height, img_width, _ = image.shape
    key_marks = sample.get_key_marks()[:, :2]
    vector_eye = (key_marks[3] - key_marks[0])
    degrees = mo.get_angle(vector_eye, np.array([100, 0]))
    rotation_mat = cv2.getRotationMatrix2D(
        ((img_width-1)/2.0, (img_height-1)/2.0), -degrees, 1)
    image_rotated = cv2.warpAffine(
        image, rotation_mat, (img_width, img_height))

    return image_rotated, degrees


def crop_face(image, marks, scale=1.8, shift_ratios=(0, 0)):
    """Crop the face area from the input image.
    Args:
        image: input image.
        marks: the facial marks of the face to be cropped.
        scale: how much to scale the face box.
        shift_ratios: shift the face box to (right, down) by facebox size * ratios
    Returns:
        cropped face image.
    """
    # How large the bounding box is?
    x_min, y_min, _ = np.amin(marks, 0)
    x_max, y_max, _ = np.amax(marks, 0)
    side_length = max((x_max - x_min, y_max - y_min)) * scale

    # Face box is scaled, get the new corners, shifted.
    img_height, img_width, _ = image.shape
    x_shift, y_shift = np.array(shift_ratios) * side_length

    x_start = int(img_width / 2 - side_length / 2 + x_shift)
    y_start = int(img_height / 2 - side_length / 2 + y_shift)
    x_end = int(img_width / 2 + side_length / 2 + x_shift)
    y_end = int(img_height / 2 + side_length / 2 + y_shift)

    # In case the new bbox is out of image bounding.
    border_width = 0
    border_x = min(x_start, y_start)
    border_y = max(x_end - img_width, y_end - img_height)
    if border_x < 0 or border_y > 0:
        border_width = max(abs(border_x), abs(border_y))
        x_start += border_width
        y_start += border_width
        x_end += border_width
        y_end += border_width
        image_with_border = cv2.copyMakeBorder(image, border_width,
                                               border_width,
                                               border_width,
                                               border_width,
                                               cv2.BORDER_CONSTANT,
                                               value=[0, 0, 0])
        image_cropped = image_with_border[y_start:y_end,
                                          x_start:x_end]
    else:
        image_cropped = image[y_start:y_end, x_start:x_end]

    return image_cropped, border_width, (x_start, y_start, x_end, y_end)

def list98_to_list68(points):
    info_68 = []
    for j in range(17):
        x = points[j*2][0]
        y = points[j*2][1]
        info_68.append([x,y,0])
    for j in range(33, 38):
        x = points[j][0]
        y = points[j][1]
        info_68.append([x,y,0])
    for j in range(42, 47):
        x = points[j][0]
        y = points[j][1]
        info_68.append([x,y,0])
    for j in range(51, 61):
        x = points[j][0]
        y = points[j][1]
        info_68.append([x,y,0])
    point_38_x = (float(points[60][0]) + float(points[62][0])) / 2.0
    point_38_y = (float(points[60][1]) + float(points[62][1])) / 2.0
    point_39_x = (float(points[62][0]) + float(points[64][0])) / 2.0
    point_39_y = (float(points[62][1]) + float(points[64][1])) / 2.0
    point_41_x = (float(points[64][0]) + float(points[66][0])) / 2.0
    point_41_y = (float(points[64][1]) + float(points[66][1])) / 2.0
    point_42_x = (float(points[60][0]) + float(points[66][0])) / 2.0
    point_42_y = (float(points[60][1]) + float(points[66][1])) / 2.0
    point_44_x = (float(points[68][0]) + float(points[70][0])) / 2.0
    point_44_y = (float(points[68][1]) + float(points[70][1])) / 2.0
    point_45_x = (float(points[70][0]) + float(points[72][0])) / 2.0
    point_45_y = (float(points[70][1]) + float(points[72][1])) / 2.0
    point_47_x = (float(points[72][0]) + float(points[74][0])) / 2.0
    point_47_y = (float(points[72][1]) + float(points[74][1])) / 2.0
    point_48_x = (float(points[68][0]) + float(points[74][0])) / 2.0
    point_48_y = (float(points[68][1]) + float(points[74][1])) / 2.0
    info_68.append([float(point_38_x), float(point_38_y), 0])
    info_68.append([float(point_39_x), float(point_39_y), 0])
    info_68.append([points[64][0], points[64][1], 0])
    info_68.append([float(point_41_x), float(point_41_y), 0])
    info_68.append([float(point_42_x), float(point_42_y), 0])
    info_68.append([points[68][0], points[68][1], 0])
    info_68.append([float(point_44_x), float(point_44_y), 0])
    info_68.append([float(point_45_x), float(point_45_y), 0])
    info_68.append([points[72][0], points[72][1], 0])
    info_68.append([float(point_47_x), float(point_47_y), 0])
    info_68.append([float(point_48_x), float(point_48_y), 0])
    
    for j in range(76, 96):
        x = points[j][0]
        y = points[j][1]
        info_68.append([x,y,0])
    return info_68