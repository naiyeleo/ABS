import pickle
import gzip
import numpy as np
import os
import sys
np.set_printoptions(precision=2, linewidth=200, threshold=10000)
import keras
from keras.models import Model, Sequential, model_from_yaml, load_model
from keras import backend as K
import json
from preprocess import CIFAR10
import tensorflow as tf
import imageio
from keras.backend import permute_dimensions
from keras.datasets import cifar10
np.random.seed(333)
tf.set_random_seed(333)

with open('config.json') as config_file:
    config = json.load(config_file)

os.environ["CUDA_VISIBLE_DEVICES"] = config["gpu_id"]
w = config["w"]
h = config["h"]
num_classes   = config["num_classes"]
use_mask = True
count_mask = True
tdname = 'temp'
window_size = 12
delta_shape = [window_size,window_size,3,3]
channel_last = bool(config['channel_last'])
Troj_size = config['max_troj_size']
Troj_Layer_dict = {}
Troj_Idx_dict = {}

cifar= CIFAR10()
print('mean', cifar.mean, 'std', cifar.std)
l_bounds = np.asarray( [(0-cifar.mean[0])/cifar.std[0], (0-cifar.mean[1])/cifar.std[1], (0-cifar.mean[2])/cifar.std[2]])
h_bounds = np.asarray( [(255-cifar.mean[0])/cifar.std[0], (255-cifar.mean[1])/cifar.std[1], (255-cifar.mean[2])/cifar.std[2]])
l_bounds = np.asarray([l_bounds for _ in range(w*h)]).reshape((1,w,h,3))
h_bounds = np.asarray([h_bounds for _ in range(w*h)]).reshape((1,w,h,3))


def stamp(n_img, delta, mask):
    r_img = n_img.copy()
    for i in range(h):
        for j in range(w):
            r_img[:,i,j] = n_img[:,i,j]*(1-mask[i,j]) + delta[:,i,j]*mask[i,j]
    return r_img

def filter_stamp(n_img, trigger):
    t_image = tf.placeholder(tf.float32, shape=(None, h, w, 3))
    tdelta = tf.placeholder(tf.float32, shape=(12, 3))
    imax =  tf.nn.max_pool( t_image, ksize=[1,window_size,window_size,1], strides=[1,1,1,1], padding='SAME')
    imin = -tf.nn.max_pool(-t_image, ksize=[1,window_size,window_size,1], strides=[1,1,1,1], padding='SAME')
    iavg =  tf.nn.avg_pool( t_image, ksize=[1,window_size,window_size,1], strides=[1,1,1,1], padding='SAME')
    i_image = tf.reshape( tf.matmul( tf.reshape( tf.concat([t_image, imax, imin, iavg], axis=3), (-1,12)) , tdelta), [-1,h,w,3])
    with tf.Session() as sess:
        r_img = sess.run(i_image, {t_image: n_img, tdelta:trigger})
    return r_img

def test(weights_file, test_xs, result, mode='mask'):
    
    model = load_model(str(weights_file))
    func = K.function([model.input, K.learning_phase()], [model.layers[-2].output])

    clean_images = cifar.preprocess(test_xs)

    if mode == 'mask':
        rimg, rdelta, rmask, tlabel = result[:4]
        t_images = stamp(clean_images, rdelta, rmask)
    elif mode == 'filter':
        rimg, rdelta, tlabel = result[:3]
        t_images = filter_stamp(clean_images, rdelta)
    for i in range(len(t_images)):
        imageio.imsave(tdname + '/' + '{0}.png'.format(i), cifar.deprocess(t_images[i]))

    nt_images = cifar.deprocess(t_images).astype('uint8')
    rt_images = cifar.preprocess(nt_images)
    
    yt = np.zeros(len(rt_images)) + tlabel
    yt = keras.utils.to_categorical(yt, num_classes)
    score = model.evaluate(rt_images, yt, verbose=0)
    return score[1]

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    test_xs = x_test
    test_ys = y_test

    print('number of seed images', len(test_ys))
    model = load_model(str(config['model_file']))
    for i in range(len(model.layers)):
        Troj_Layer_dict[model.layers[i].name] = i
    n_weights = 0
    for i in range(len(model.layers)):
        n_weights += len(model.layers[i].get_weights())
        Troj_Idx_dict[model.layers[i].name] = n_weights

    layers = [l.name for l in model.layers]

    # mask check 
    tlabel = 0
    rdelta = pickle.load(open('./deltas/nin_trojan_dark_red_1_1_model_conv2d_7_135_64_0.pkl', 'rb'))
    rmask = pickle.load(open('./masks/nin_trojan_dark_red_1_1_model_conv2d_7_135_64_0.pkl', 'rb'))
    rimg = imageio.imread('./imgs/nin_trojan_dark_red_1_1_model_conv2d_7_135_64_0_0.png')
    result = rimg, rdelta, rmask, tlabel
    reasr = test(str(config['model_file']), test_xs, result)
    print('mask reasr', reasr)

    # filter check
    # rdelta = pickle.load(open('./deltas/filter_delta.pkl', 'rb'))
    # rimg = imageio.imread('./imgs/filter_imgs.png')
    # result = rimg, rdelta, tlabel
    # reasr = test(str(config['model_file']), test_xs, result, 'filter')
    # print('filter reasr', reasr)

