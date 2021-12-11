from keras import backend as K
from keras.applications.mobilenet import MobileNet, preprocess_input
from keras.models import load_model, Model, Sequential
from keras.optimizers import RMSprop, Adam, SGD
from keras.engine.topology import Layer
from keras.engine import InputSpec
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.activations import relu

from keras.layers import Input, Activation, Conv2D
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import GlobalAveragePooling2D, MaxPooling2D
from keras.layers.convolutional import Conv2DTranspose, UpSampling2D
from keras.layers.core import Reshape, Dropout
from keras.layers.core import Lambda

from PIL import Image, ImageDraw, ImageFont, ImageOps
import tensorflow as tf

import cv2
from ImageProcessing import get_viewport
import numpy as np

from CameraOperations import show_grid

from fcnprocessing import ImageDataGenerator
import keras


def run_prediction(x):
    prediction = model.predict(x, 1, True)
    return prediction


def read_config(config_file="./imagenet1000_clsid_to_human_array.json"):
    '''
      Read Configuration parameters from external file
    '''
    import json
    
    with open(config_file) as config_data:
        config = json.load(config_data)
    return config


def predict_with_mobilenet(image, model=None):
    if model is None:
        model = load_model('./mobilenet_1_0_224_tf_local.h5') 
    image_input = image/255.
    X = np.expand_dims(image_input, 0)
    prediction = model.predict(X, len(X), False)
    idx = np.argmax(prediction, 1)[0]
    return idx, prediction, model

def save_channels(prediction, classes=1000):
    max_prob = np.max(prediction)
    print("\n{: >3.0f} ".format(0), end="")
    for r in range(classes):
        prediction_img = prediction[0][:,:,r]*255/max_prob
        cv2.imwrite("./deconv/img{}.jpg".format(r), prediction_img)
        '''
        if r>0 and not r%10:
            term = "\n{: >3.0f} ".format(int(r/10))
        else:
            term = ","
        #print("{1: >6.2f}/{0: >4.2f}".format(np.max(prediction[0][:,:,r]),np.sum(prediction[0][:,:,r])), end=term)
        print("{0: >6.2f}".format(np.max(prediction[0][:,:,r])/np.sum(prediction[0][:,:,r])), end=term)
        '''

def deconv_layer(inputs, 
                 filters, 
                 alpha=1.0, 
                 kernel=(3, 3), 
                 strides=(2, 2), 
                 block_id=1, 
                 activation=None, 
                 use_bias=True):
    '''
        Deconvolutional layers
    '''
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    filters = int(filters * alpha)
    x = Conv2DTranspose(filters, kernel,
               padding='valid',
               use_bias=use_bias,
               strides=strides,
               name='deconv_%d' % block_id,
               kernel_initializer='he_normal',
               bias_initializer='zeros')(inputs)
    #x = BatchNormalization(axis=channel_axis, name='deconv_bn_%d' % block_id)(x)
    if activation is None:
        return Activation(relu, name='deconv_relu_%d' % block_id)(x)
    else:
        return activation(x)

    
def conv_layer(inputs, 
               filters, 
               alpha=1.0, 
               kernel=(3, 3), 
               strides=(1, 1), 
               block_id=1, 
               activation=None, 
               use_bias=False,
               padding='valid'):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    filters = int(filters * alpha)
    x = Conv2D(filters, kernel,
               padding=padding,
               use_bias=use_bias,
               strides=strides,
               name='t_conv_%d' % block_id,
               kernel_initializer='he_normal',
               bias_initializer='zeros')(inputs)
    x = BatchNormalization(axis=channel_axis, name='t_conv_bn_%d' % block_id)(x)
    if activation is None:
        return Activation(relu, name='t_conv_relu_%d' % block_id)(x)
    else:
        return activation(x)

class CustomReshape(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(CustomReshape, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        super(CustomReshape, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return tf.reshape(x, [-1, self.output_dim]) #np.reshape(x, (-1, 14)) 

    def compute_output_shape(self, input_shape):
        print("CustomReshape", input_shape)
        return (input_shape[0], self.output_dim)

    def get_config(self):
        config = {'output_dim': self.output_dim}
        base_config = super(CustomReshape, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class BilinearUpSampling2D(Layer):

    def __init__(self, output_dim, scale_factor, **kwargs):
        self.output_dim   = output_dim
        self.scale_factor = scale_factor
        super(BilinearUpSampling2D, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        super(BilinearUpSampling2D, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        input_size = tf.shape(x)
        size = (self.scale_factor*input_size[1], self.scale_factor*input_size[2])
        new_size = tf.convert_to_tensor(size, dtype=tf.int32)
        return tf.image.resize_images(x, new_size, align_corners=True, method=tf.image.ResizeMethod.BILINEAR)
        #return tf.reshape(x, [-1, self.output_dim]) 

    def compute_output_shape(self, input_shape):
        print("BilinearUpSampling2D", input_shape)
        return input_shape[:-1] + (self.output_dim,)

    def get_config(self):
        config = {'output_dim': self.output_dim, 'scale_factor': self.scale_factor}
        base_config = super(BilinearUpSampling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def generate_magic_kernel(classes=15, channels=15):
    magic_kernel = np.array([[0.25,0.75,0.75,0.25],
                         [0.25,0.75,0.75,0.25],
                         [0.25,0.75,0.75,0.25],
                         [0.25,0.75,0.75,0.25]])
    kernel = []
    for c in range(channels):
        kernel_channel
        for cl in range(classes):
            kernel_channel.append(magic_kernel)
        kernel.append(kernel_channel)
    return np.array(kernel)


def preprocess_image(image, normalize=False, pad=0):
    '''
    Normalize is divide by 255.
    pad is pad to multiples of 'pad'
    '''
    w = image.shape[1]
    h = image.shape[0]
    img = None
    if pad > 0:
        w_pad = pad - (w % pad)
        h_pad = pad - (h % pad)
        img = np.lib.pad(image, ((h_pad,0),(w_pad,0),(0,0)), 'constant', constant_values=(0,0))
    if normalize:
        img = img/255.
    return img


def load_img(file, normalize=False, pad=0):
    '''
     Load images and always convert to RGB representation
     If 'normalize' is True, then divide by 255.
     If 'pad' is more than 0, pad the image to a multiple of the provided number
    '''
    image = Image.open(file)
    if image.mode is not 'RGB':
        image = image.convert('RGB')
    return preprocess_image(np.asarray(image), normalize, pad)


def pad_image(image, thickness=(1,1), padding=(1,1)):
    return np.lib.pad(image, thickness, 'constant', constant_values=padding)


def predict_with_model(image, model=None):
    image_input = image
    X = np.expand_dims(image_input, 0)
    prediction = model.predict(X, len(X), True)
    idx = np.argmax(prediction)
    return idx, prediction


def prediction_heatmaps(image, model, classes=2):
    result = predict_with_model(image, model)
    pred = result[1][0]
    heatmap = np.concatenate([pad_image(pred[:,:,c], thickness=(1,1)) for c in range(classes)], axis=1)
    return heatmap

'''
conv_model = Model(segmenter.input, segmenter.layers[-6].output)
'''

def extract_heatmap_class(heatmap, classid, classes=2):
    frame_width = heatmap.shape[1]/classes
    start = int(classid*frame_width)
    end   = int(start + frame_width)
    return heatmap[:,start:end]


def image_clahe(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    r = clahe.apply(image[:,:,0])
    g = clahe.apply(image[:,:,1])
    b = clahe.apply(image[:,:,2])
    
    return np.stack([r,g,b], -1)

def process_image(img_file, fcn, conv, targets=[0,1], pad=0, equalize=False):
    '''
    Show heatmaps for image targets from deconvolutional model and pure fully convolutional model
    targets is a list of class ids to show
    '''
    img = load_img(img_file, normalize=False, pad=pad)
    if equalize:
        img = image_clahe(img)
        
    print("Image Dimensions", img.shape)
    target_fcn = None
    
    if fcn is not None:
        heatmap = prediction_heatmaps(img/255., fcn)
        plt.figure(figsize=(15,2))
        p = plt.imshow(heatmap, cmap="hot")
        print("Heatmap Dimensions", heatmap.shape)
        names = [class_names[n] for n in targets]
        names.insert(0, "Source")
        target_fcn = [extract_heatmap_class(heatmap, c) for c in targets]
        target_fcn.insert(0, img)
        show_grid(target_fcn, names, cmap_mono="hot")

    if conv is not None:
        heatmap = prediction_heatmaps(img/255., conv)
        plt.figure(figsize=(15,2))
        p = plt.imshow(heatmap, cmap="hot")
        print("Heatmap Dimensions", heatmap.shape)
        names = [class_names[n] for n in targets]
        names.insert(0, "Source")
        target = [extract_heatmap_class(heatmap, c) for c in targets]
        target.insert(0, img)
        show_grid(target, names, cmap_mono="hot")
    return target_fcn


def generate_kitti_road_skiplayer_fcn(source_model, classes=2, deconv=True, dropout=1e-3):
    '''
        Make Mobilenet fully convolutional with any size input and
        segmentation output. source_model provides fully loaded Mobilenet 
        so we can extract its weights for the FCN model
        
        Return the fully convolutional model.
    '''
    x = source_model.layers[-1].output #Last layer used from Mobilenet. Choosing the last convolution
    #x_channels = mobilenet.layers[-7].output_shape[-1]
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    '''    
    for i in range(len(source_model.layers) - 4):
        source_model.layers[i].trainable = False
    source_model.layers[-3].trainable = True
    trainable_layers = 16 # Train the deconvolutions
    '''

    bias_flag = True
    bias_init = 'he_normal'

    skip_id = 14
    skip_7 = source_model.layers[-skip_id].output
    skip_7_channels = source_model.layers[-skip_id].output_shape[-1]
    skip_7 = Conv2D(classes, (1,1),
               padding="same",
               use_bias=bias_flag,
               strides=(1,1),
               activation='softmax',
               name='t_skip_conv_7',
               kernel_initializer='he_normal',
               bias_initializer=bias_init)(skip_7)
    
    skip_id = 51
    skip_5 = source_model.layers[-skip_id].output
    skip_5_channels = source_model.layers[-skip_id].output_shape[-1]
    skip_5 = Conv2D(classes, (1,1),
               padding="same",
               use_bias=bias_flag,
               strides=(1,1),
               activation='softmax',
               name='t_skip_conv_5',
               kernel_initializer='he_normal',
               bias_initializer=bias_init)(skip_5)
    
    skip_id = 64
    skip_3 = source_model.layers[-skip_id].output
    skip_3_channels = source_model.layers[-skip_id].output_shape[-1]
    skip_3 = Conv2D(classes, (1,1),
               padding="same",
               use_bias=bias_flag,
               strides=(1,1),
               activation='softmax',
               name='t_skip_conv_3',
               kernel_initializer='he_normal',
               bias_initializer=bias_init)(skip_3)
    
    sig = Activation('softmax', name='act_softmax2')
    bias_flag = True
    bias_init = 'he_normal'
    # Setup the network
    x = Conv2D(classes, (1,1),
               padding="same",
               use_bias=bias_flag,
               strides=(1,1),
               activation='relu',
               name='t_reduce_channels_1',
               kernel_initializer='he_normal',
               bias_initializer=bias_init)(x)
    x = BilinearUpSampling2D(scale_factor=2, output_dim=classes, name='upscore_1')(x)
    x_skip7 = keras.layers.Add()([x, skip_7])
    x_skip7 = BatchNormalization(axis=channel_axis, name='deconv_bn_%d' % 1)(x_skip7)
    x = Conv2D(classes, (1,1),
               padding="same",
               use_bias=bias_flag,
               strides=(1,1),
               activation='relu',
               name='t_conv2d_smoothing_1',
               kernel_initializer='he_normal',
               bias_initializer=bias_init)(x_skip7)
    x = BilinearUpSampling2D(scale_factor=2, output_dim=classes, name='upscore_2')(x)
    x_skip5 = keras.layers.Add()([x, skip_5])
    x_skip5 = BatchNormalization(axis=channel_axis, name='deconv_bn_%d' % 2)(x_skip5)
    x = Conv2D(classes, (1,1),
               padding="same",
               use_bias=bias_flag,
               strides=(1,1),
               activation='relu',
               name='t_conv2d_smoothing_2',
               kernel_initializer='he_normal',
               bias_initializer=bias_init)(x_skip5)
    x = BilinearUpSampling2D(scale_factor=2, output_dim=classes, name='upscore_3')(x)
    x_skip3 = keras.layers.Add()([x, skip_3])
    x_skip3 = BatchNormalization(axis=channel_axis, name='deconv_bn_%d' % 3)(x_skip3)
    x = Conv2D(classes, (1,1),
               padding="same",
               use_bias=bias_flag,
               strides=(1,1),
               activation='relu',
               name='t_conv2d_smoothing_3',
               kernel_initializer='he_normal',
               bias_initializer=bias_init)(x_skip3)
    x = BilinearUpSampling2D(scale_factor=2, output_dim=classes, name='upscore_4')(x)
    x = BatchNormalization(axis=channel_axis, name='deconv_bn_%d' % 4)(x)
    x = Conv2D(classes, (1,1),
               padding="same",
               use_bias=bias_flag,
               strides=(1,1),
               activation='relu',
               name='t_conv2d_smoothing_4',
               kernel_initializer='he_normal',
               bias_initializer=bias_init)(x)
    x = BilinearUpSampling2D(scale_factor=2, output_dim=classes, name='upscore_5')(x)
    x = BatchNormalization(axis=channel_axis, name='deconv_bn_%d' % 5)(x)
    x = Conv2D(classes, (1,1),
               padding="same",
               use_bias=bias_flag,
               strides=(1,1),
               activation='softmax',
               name='t_conv2d_smoothing_5',
               kernel_initializer='he_normal',
               bias_initializer=bias_init)(x)
    #compose the combined model with the new top
    new_model = Model(source_model.input, x)
    for i in range(len(new_model.layers)-25):
        new_model.layers[i].trainable = False

    new_model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.0001), metrics=['accuracy'])
    new_model.summary()
    return new_model


def training_callbacks(folder="./training/", name_prefix="classifier_", batch=30, use_val=True):
    '''
    Generate callbacks to checkpoint model during training and reduce the learning rate on plateau
    'use_val' decides whether to use training or validation statistics for decisions and reporting
    '''
    #filepath = "./training/weights-improvement-{epoch:02d}-{val_acc:.3f}-{val_loss:.3f}.hdf5"
    filepath = folder+name_prefix+"{epoch:02d}-{val_acc:.3f}-{val_loss:.3f}.hdf5"
    monitor_checkpoint = 'val_acc'
    monitor_reduce_lr = 'val_loss'
    if not use_val:
        filepath = folder+name_prefix+"{epoch:02d}-{acc:.3f}-{loss:.3f}.hdf5"
        monitor_checkpoint = 'acc'
        monitor_reduce_lr = 'loss'
    checkpoint = ModelCheckpoint(filepath, 
                                 monitor=monitor_checkpoint, 
                                 verbose=0, 
                                 save_best_only=True, 
                                 save_weights_only=False, 
                                 mode='auto')
    reduce_lr = ReduceLROnPlateau(monitor=monitor_reduce_lr, 
                                  factor=0.75, patience=5, 
                                  min_lr=0.000002, 
                                  mode='min', 
                                  verbose=0)
    tensorboard = TensorBoard(log_dir='./tensorboard', 
                              histogram_freq=0, 
                              batch_size=batch, 
                              write_graph=True, 
                              write_grads=False, 
                              write_images=True, 
                              embeddings_freq=0, 
                              embeddings_layer_names=None, 
                              embeddings_metadata=None)
    callbacks_list = [checkpoint, reduce_lr, tensorboard]
    return callbacks_list



def normalize_input(x):
    #x = image_clahe(x.astype("uint8"))
    x = x / 127.5
    x -= 1.
    return x

import decode_traffic as dt

def run():
    #input_size = (256,256)
    fcn_size   = (512,512)
    classes    = 4
    class_names = dt.TARGET_CLASS_NAMES

    fcn_input = Input(shape=(None, None, 3))

    mobilenetv1 = MobileNet(input_tensor=fcn_input, 
                            alpha=1.0, 
                            depth_multiplier=1, 
                            include_top=False, 
                            weights='imagenet', 
                            pooling=None, 
                            classes=1000)

    fcn_model  = generate_kitti_road_skiplayer_fcn(mobilenetv1, 
                                                   deconv=False, 
                                                   classes=classes)

    # we create two instances with the same arguments
    shift = 0.2
    data_format = K.image_data_format()
    data_gen_args = dict(preprocessing_function=normalize_input,
                         horizontal_flip=True,
                         vertical_flip=True,
                         height_shift_range=shift, 
                         width_shift_range=shift,
                         brightness=5,
                         fill_mode='constant',
                         cval=0,
                         data_format=data_format,
                         num_classes=classes)
    '''
                         
                         channel_shift_range=32,
                         rotation_range=0)
                         shear_range=0.2,
                         zoom_range=0.2,
    '''

    image_datagen = ImageDataGenerator(**data_gen_args)
    # Provide the same seed and keyword arguments to the fit and flow methods
    seed = 1
    root = "./data/"
    image_dir = root+"input/"
    mask_dir  = root+"output/"
    multiplier = 4
    batch = 20
    steps = 256 #Assume that the more variations to images, the more steps per epoch to cover them all

    image_generator = image_datagen.flow_from_directory(image_dir,
                                                        class_mode='sparse_mask',
                                                        batch_size=batch,
                                                        target_size=fcn_size)

    #from keras.utils import multi_gpu_model
    #gpu_model = multi_gpu_model(fcn_model, gpus=2, cpu_merge=True, cpu_relocation=False)
    #gpu_model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.0001), metrics=['accuracy'])

    fcn_model.fit_generator(image_generator,
                            steps_per_epoch=steps,
                            verbose=1,
                            epochs=100,
                            callbacks=training_callbacks(batch=batch, name_prefix="fcn_weights_", use_val=False))
    fcn_model.save('fcn_classifier_model.hdf5')

if __name__ == '__main__':
    print("Mobilenet Traffic Light FCN v3.0")
    print("Train Mobilenet FCN with 3 Skip Layers after data cleanup")
    run()
