from layer import *

# !pip install -U git+https://github.com/leondgarse/keras_cv_attention_models -q
from keras_cv_attention_models import efficientnet

def get_base_model(name, input_shape):
    if name == 'EfficientNetV2S':
        return efficientnet.EfficientNetV2S(num_classes=0, input_shape=input_shape, pretrained="imagenet")

    if name == 'EfficientNetV1B1':
        return efficientnet.EfficientNetV1B1(num_classes=0, input_shape=input_shape, pretrained="imagenet")

    if name == 'EfficientNetV1B2':
        return efficientnet.EfficientNetV1B2(num_classes=0, input_shape=input_shape, pretrained="imagenet")

    if name == 'EfficientNetV1B3':
        return efficientnet.EfficientNetV1B3(num_classes=0, input_shape=input_shape, pretrained="imagenet")

    if name == 'ResNet50':
        return tf.keras.applications.resnet50.ResNet50(include_top=False, input_shape=input_shape, weights='imagenet')

    raise Exception("Cannot find this base model:", name)

def create_model(input_shape, base, n_landmark, final_dropout):
    inp = Input(shape=input_shape, name="input_1")
    
    x = base(inp)
    
    x = GlobalAveragePooling2D()(x)

    x = Dropout(final_dropout)(x)

    x = Dense(512, activation="swish")(x)
    
    x = Dropout(final_dropout)(x)

    x = Dense(n_landmark * 2, activation="sigmoid")(x)

    x = Reshape((n_landmark, 2))(x)

    model = Model([inp], [x])
    
    return model

if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"]="2, 3"

    im_size = 192
    img_size = (im_size, im_size)
    input_shape = (im_size, im_size, 3)
    base_name = 'EfficientNetV1B2'
    final_dropout = 0.1
    n_landmark = 68

    base = get_base_model(base_name, input_shape)
    model = create_model(input_shape, base, n_landmark, final_dropout)
    model.summary()