from fmd.ds300w import DS300W
from fmd.mark_dataset.util import draw_marks

# Set the path to the dataset directory.
DS300W_DIR = "/home/lap14880/hieunmt/face_landmark/download/300W"

# Construct a dataset.
ds = DS300W("300w")

# Populate the dataset with essential data
ds.populate_dataset(DS300W_DIR)

# See what we have got.
print(ds)

sample = ds.pick_one()

# for sample in ds:
#     # do whatever you want, like
#     print(sample.marks)

image = sample.read_image()

import cv2
cv2.imwrite("Preview.jpg", image)

facial_marks = sample.marks

key_marks = sample.get_key_marks()

print(facial_marks)

draw_marks(image, facial_marks)

cv2.imwrite("Preview_landmark.jpg", image)