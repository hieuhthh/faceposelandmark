import sys 
sys.path.append('datatool')

import tensorflow as tf

from datatool.make_dataset import *

def build_decoder(with_labels=True, target_size=(256, 256)):
    def decode_img(img):
        # img = tf.image.resize(img, target_size)
        img = tf.cast(img, tf.float32) / 255.0
        return img

    def decode_label(label):
        label = tf.cast(label, tf.float32) / target_size[0]
        return label

    def decode_with_labels(img, label):
        return decode_img(img), decode_label(label)

    return decode_with_labels if with_labels else decode_img

def build_dataset(imgs, labels=None, labelsID=None, bsize=32,
                  decode_fn=None, augment=False, 
                  repeat=True, shuffle=1024,
                  ):
    """
    imgs: images
    labels: list of 68 pair (x, y)
    """              
    
    AUTO = tf.data.experimental.AUTOTUNE
    dataset_input = tf.data.Dataset.from_tensor_slices((imgs))
    dataset_label = tf.data.Dataset.from_tensor_slices((labels))

    dset = tf.data.Dataset.zip((dataset_input, dataset_label))
    dset = dset.repeat() if repeat else dset
    dset = dset.shuffle(shuffle) if shuffle else dset
    dset = dset.map(decode_fn, num_parallel_calls=AUTO)
    dset = dset.batch(bsize)
    dset = dset.prefetch(AUTO)
    
    return dset

"""
image_list: list of images (160x160x3) not normalized
masks_list: list of 68 pair (x, y) not normalized
"""

if __name__ == '__main__':
    im_size = 160
    img_size = (im_size, im_size)
    BATCH_SIZE = 128
    repeat = True
    shuffle = 4096

    image_list, masks_list = get_images_marks(im_size)

    # print(image_list[0])
    # print(masks_list[0])

    print(np.shape(image_list))
    print(np.shape(masks_list))

    decoder = build_decoder(with_labels=True, target_size=img_size)
    dataset = build_dataset(image_list, masks_list, bsize=BATCH_SIZE, decode_fn=decoder,
                            repeat=repeat, shuffle=shuffle)

    for x, y in dataset:
        print(x)
        print(y)
        break