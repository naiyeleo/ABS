import pickle
import h5py
import gzip
import numpy as np
import os
import sys
import json
np.set_printoptions(precision=2, linewidth=200, threshold=10000)

with open('config.json') as config_file:
    config = json.load(config_file)

use_pickle = bool(config["use_pickle"])
use_h5     = bool(config["use_h5"])
channel_last = bool(config['channel_last'])
if use_pickle and use_h5:
    print('Error config use_pickle and use_h5 cannot both be True')
    sys.exit()
random_seed = int(config['random_seed'])
os.environ["CUDA_VISIBLE_DEVICES"] = config["gpu_id"]
print('use pickle', use_pickle, 'use_h5', use_h5, 'channel_last', channel_last, 'gpu_id', config["gpu_id"])

import tensorflow.keras as keras
from tensorflow.keras.models import Model, Sequential, model_from_yaml, load_model
from tensorflow.keras import backend as K
from preprocess import CIFAR10
import tensorflow as tf
import imageio
from tensorflow.keras.backend import permute_dimensions

tfconfig = tf.ConfigProto()
tfconfig.gpu_options.per_process_gpu_memory_fraction = 0.9
tf.keras.backend.set_session(tf.Session(config=tfconfig));

np.random.seed(random_seed)
tf.set_random_seed(random_seed)
w = config["w"]
h = config["h"]
num_classes   = config["num_classes"]
use_mask = True
count_mask = True
tdname = 'temp'
window_size = 12
mask_epsilon = 0.01
delta_shape = [window_size,window_size,3,3]
Troj_size = config['max_troj_size']
reasr_bound = float(config['reasr_bound'])
Troj_Layer_dict = {}
Troj_Idx_dict = {}
top_n_neurons = int(config['top_n_neurons'])
mask_multi_start = int(config['mask_multi_start'])
filter_multi_start = int(config['filter_multi_start'])
re_mask_weight = float(config['re_mask_weight'])
re_mask_lr = float(config['re_mask_lr'])
if 'img_seed_file' not in config.keys():
    seed_file = config['img_pickle_file']
else:
    seed_file = config['img_seed_file']

cifar= CIFAR10()
print('gpu id', config["gpu_id"])
l_bounds = cifar.l_bounds
h_bounds = cifar.h_bounds
print('mean', cifar.mean, 'std', cifar.std, 'l bounds', l_bounds[0,0,0], 'h_bounds', h_bounds[0,0,0])

l_bounds_channel_first = np.transpose(l_bounds, [0,3,1,2])
h_bounds_channel_first = np.transpose(h_bounds, [0,3,1,2])

use_resnet  = bool(config['use_resnet'])
has_softmax = bool(config['has_softmax'])
Print_Level = int(config['print_level'])

if Print_Level > 0:
    print('tensorflow', tf.__version__)
    print('keras', keras.__version__)


if 'tf' in keras.__version__:
    def custom_pop(model, idx = -1):
        if idx > 0:
            while(len(model._layers))>idx+1:
                if Print_Level > 1:
                    print('after remove', len(model._layers))
                model._layers = model._layers[:-1]
        else:
            model._layers = model._layers[:-1]
        model.layers[idx]._outbound_nodes = []
        model._layers[idx]._outbound_nodes = []
        model.outputs = [model._layers[idx].output]
        model._init_graph_network(model.inputs, model.outputs, name=model.name)
        model.built = True
        return model
else:
    def custom_pop(model, idx = -1):
        if idx > 0:
            while(len(model.layers))>idx+1:
                print(len(model.layers))
                model.layers.pop()
        else:
            model.layers.pop()
        model.layers[idx]._outbound_nodes = []
        model.outputs = [model.layers[idx].output]
        # update model._inbound_nodes
        model._inbound_nodes[0].output_tensors = model.outputs
        model._inbound_nodes[0].output_shapes = [model.outputs[0]._keras_shape]
        return model

def getlayer_output(l_in, l_out, x, model):
    get_k_layer_output = K.function([model.layers[l_in].input, 0], [model.layers[l_out].output])
    return get_k_layer_output([x])[0]

def check_values(images, labels, model):
    maxes = {}
    for hl_idx in range(0, len(model.layers) - 1):
        if use_resnet:
            if not ('Add' in model.layers[hl_idx].__class__.__name__ or  'Dense' in model.layers[hl_idx].__class__.__name__) :
                continue
        else:
            if not ('Conv' in model.layers[hl_idx].__class__.__name__ or  'Dense' in model.layers[hl_idx].__class__.__name__ or 'Flatten' in model.layers[hl_idx].__class__.__name__) :
                continue
        if channel_last:
            n_neurons = model.layers[hl_idx].output_shape[-1]
        else:
            n_neurons = model.layers[hl_idx].output_shape[1]
        
        h_layer = getlayer_output(0, hl_idx, images, model).copy()
        key = '{0}'.format(model.layers[hl_idx].name)
        if key in maxes.keys():
            maxes[key].append(np.amax(h_layer))
        else:
            maxes[key] = [np.amax(h_layer)]
    return maxes

def sample_neuron(images, labels, model, mvs):
    all_ps = {}
    samp_k = config['samp_k']
    same_range = config['same_range']
    n_samples = config['n_samples']
    batch_size = config['samp_batch_size']
    n_images = images.shape[0]
    if Print_Level > 0:
        print('sampling n imgs', n_images)

    end_layer = len(model.layers)-2
    if has_softmax:
        end_layer = len(model.layers)-3
    for hl_idx in range(0,end_layer):
        if use_resnet:
            if not ('Add' in model.layers[hl_idx].__class__.__name__ or  'Dense' in model.layers[hl_idx].__class__.__name__) :
                continue
        else:
            if not ('Conv' in model.layers[hl_idx].__class__.__name__ or  'Dense' in model.layers[hl_idx].__class__.__name__ or 'Flatten' in model.layers[hl_idx].__class__.__name__) :
                continue
        if channel_last:
            n_neurons = model.layers[hl_idx].output_shape[-1]
        else:
            n_neurons = model.layers[hl_idx].output_shape[1]
        if n_neurons == num_classes:
            continue

        if same_range:
            vs = np.asarray([i*samp_k for i in range(n_samples)])
        else:
            tr = samp_k * max(mvs[model.layers[hl_idx].name])/n_samples
            vs = np.asarray([i*tr for i in range(n_samples)])
        
        h_layer = getlayer_output(0,hl_idx,images,model).copy()

        nbatches = n_neurons//batch_size
        for nt in range(nbatches):
            l_h_t = []
            for neuron in range(batch_size):
                if len(h_layer.shape) == 4:
                    h_t = np.tile(h_layer, (n_samples,1,1,1))
                else:
                    h_t = np.tile(h_layer, (n_samples,1))

                for i,v in enumerate(vs):
                    if len(h_layer.shape) == 4:
                        if channel_last:
                            h_t[i*n_images:(i+1)*n_images,:,:,neuron+nt*batch_size] = v
                        else:
                            h_t[i*n_images:(i+1)*n_images,neuron+nt*batch_size,:,:] = v
                    else:
                        h_t[i*n_images:(i+1)*n_images,neuron+nt*batch_size] = v
                l_h_t.append(h_t)
            f_h_t = np.concatenate(l_h_t, axis=0)

            if has_softmax:
                fps = getlayer_output(hl_idx+1, len(model.layers)-2,f_h_t,model)
            else:
                fps = getlayer_output(hl_idx+1, len(model.layers)-1,f_h_t,model)
            for neuron in range(batch_size):
                tps = fps[neuron*n_samples*n_images:(neuron+1)*n_samples*n_images]
                for img_i in range(n_images):
                    img_name = (labels[img_i], img_i)
                    ps_key= (img_name, model.layers[hl_idx].name, neuron+nt*batch_size)
                    ps = [tps[img_i+n_images*i] for i in range(n_samples)]
                    ps = np.asarray(ps)
                    ps = ps.T
                    all_ps[ps_key] = np.copy(ps)
    return all_ps


def find_min_max(model_name, all_ps, cut_val=20, top_k = 10):
    max_ps = {}
    max_vals = []
    n_classes = 0
    n_samples = 0
    for k in sorted(all_ps.keys()):
        all_ps[k] = all_ps[k][:, :cut_val]
        n_classes = all_ps[k].shape[0]
        n_samples = all_ps[k].shape[1]
        # maximum increase diff
        vs = []
        for l in range(num_classes):
            vs.append( np.amax(all_ps[k][l][1:]) - np.amin(all_ps[k][l][:1]) )
            # vs.append( np.amax(all_ps[k][l][all_ps[k].shape[1]//5:]) - np.amin(all_ps[k][l][:all_ps[k].shape[1]//5]) )
            # vs.append( np.amax(all_ps[k][l]) - np.amin(all_ps[k][l]) )
        ml = np.argsort(np.asarray(vs))[-1]
        sml = np.argsort(np.asarray(vs))[-2]
        val = vs[ml] - vs[sml]

        max_vals.append(val)
        max_ps[k] = (ml, val)
    
    neuron_ks = []
    imgs = []
    for k in sorted(max_ps.keys()):
        nk = (k[1], k[2])
        neuron_ks.append(nk)
        imgs.append(k[0])
    neuron_ks = list(set(neuron_ks))
    imgs = list(set(imgs))
    
    min_ps = {}
    min_vals = []
    n_imgs = len(imgs)
    for k in neuron_ks:
        vs = []
        ls = []
        vdict = {}
        for img in sorted(imgs):
            # nk = img + '_' + k
            nk = (img, k[0], k[1])
            l = max_ps[nk][0]
            v = max_ps[nk][1]
            vs.append(v)
            ls.append(l)
            if not ( l in vdict.keys() ):
                vdict[l] = [v]
            else:
                vdict[l].append(v)
        ml = max(set(ls), key=ls.count)


        fvs = []
        # does not count when l not equal ml
        for img in sorted(imgs):
            # img_l = int(img.split('_')[0])
            img_l = int(img[0])
            if img_l == ml:
                continue
            # nk = img + '_' + k
            nk = (img, k[0], k[1])
            l = max_ps[nk][0]
            v = max_ps[nk][1]
            if l != ml:
                continue
            fvs.append(v)
        
        if len(fvs) > 0:
            min_ps[k] = (ml, ls.count(ml), np.amin(fvs), fvs)
            min_vals.append(np.amin(fvs))
            # min_ps[k] = (ml, ls.count(ml), np.mean(fvs), fvs)
            # min_vals.append(np.average(fvs))

        else:
            min_ps[k] = (ml, 0, 0, fvs)
            min_vals.append(0)
    
    
    keys = min_ps.keys()
    keys = []
    for k in min_ps.keys():
        if min_ps[k][1] >= n_imgs-2:
            keys.append(k)
    sorted_key = sorted(keys, key=lambda x: min_ps[x][2] )
    if Print_Level > 0:
        print('n samples', n_samples, 'n class', n_classes)
        # print('sorted_key', sorted_key)


    neuron_dict = {}
    neuron_dict[model_name] = []
    maxval = min_ps[sorted_key[-1]][2]
    layers = {}
    allns = 0

    for i in range(len(sorted_key)):
        k = sorted_key[-i-1]
        # layer = '_'.join(k.split('_')[:-1])
        # neuron = k.split('_')[-1]
        layer = k[0]
        neuron = k[1]
        label = min_ps[k][0]
        if layer not in layers.keys():
            layers[layer] = 1
        else:
            layers[layer] += 1
        if layers[layer] <= 3:
            if (layer, neuron, min_ps[k][0]) in neuron_dict[model_name]:
                continue
            if Print_Level > 0:
                print('min max val across images', 'k', k, 'label', min_ps[k][0], min_ps[k][1], 'value', min_ps[k][2])
                if Print_Level > 1:
                    print(min_ps[k][3])
            allns += 1
            neuron_dict[model_name].append( (layer, neuron, min_ps[k][0]) )
        if allns > top_k//2:
            break

    for i in range(len(sorted_key)):
        k = sorted_key[-i-1]
        # layer = '_'.join(k.split('_')[:-1])
        # neuron = k.split('_')[-1]
        layer = k[0]
        neuron = k[1]
        label = min_ps[k][0]
        if (layer, neuron, min_ps[k][0]) in neuron_dict[model_name]:
            continue
        if True:
            if Print_Level > 0:
                print('min max val across images', 'k', k, 'label', min_ps[k][0], min_ps[k][1], 'value', min_ps[k][2])
                if Print_Level > 1:
                    print(min_ps[k][3])
            allns += 1
            neuron_dict[model_name].append( (layer, neuron, min_ps[k][0]) )
        if allns > top_k:
            break

    return neuron_dict

def read_all_ps(model_name, all_ps, top_k=10, cut_val=5):
    return find_min_max(model_name, all_ps,  cut_val, top_k=top_k)

def filter_img(w, h):
    mask = np.zeros((h, w), dtype=np.float32)
    Troj_w = int(np.sqrt(Troj_size)) 
    for i in range(h):
        for j in range(w):
            # if j >= 2 and j < 8  and i >= 2 and  i < 8:
            if  j < Troj_w  and  i < Troj_w:
            # if i % 6 == 0 and j % 6 == 0:
                mask[i,j] = 1
    return mask


def nc_filter_img(w, h):
    if use_mask:
        mask = np.zeros((h, w), dtype=np.float32)
        for i in range(h):
            for j in range(w):
                # if not( j >= w*1/4.0 and j < w*3/4.0  and i >= h*1/4.0 and i < h*3/4.0):
                if True:
                    mask[i,j] = 1
        mask = np.zeros((h, w), dtype=np.float32) + 1
    else:
        mask = np.zeros((h, w), dtype=np.float32) + 1
    return mask


def setup_model(optz_option, weights_file, Troj_Layer, Troj_next_Layer):
    nc_mask = nc_filter_img(w,h)

    with tf.variable_scope("", reuse=tf.AUTO_REUSE):
        mask = tf.get_variable("mask", [h,w], dtype=tf.float32)
        if channel_last:
            s_image = tf.placeholder(tf.float32, shape=(None, h, w, 3))
            delta= tf.get_variable("delta", [1,h,w,3], constraint=lambda x: tf.clip_by_value(x, l_bounds, h_bounds))
        else:
            s_image = tf.placeholder(tf.float32, shape=(None, 3, h, w))
            delta= tf.get_variable("delta", [1,3,h,w], constraint=lambda x: tf.clip_by_value(x, l_bounds_channel_first, h_bounds_channel_first))
    
    con_mask = tf.tanh(mask)/2.0 + 0.5
    con_mask = con_mask * nc_mask
    if channel_last:
        use_mask = tf.tile(tf.reshape(con_mask, (1,h,w,1)), tf.constant([1,1,1,3]))
    else:
        use_mask = tf.tile(tf.reshape(con_mask, (1,1,h,w)), tf.constant([1,3,1,1]))
    i_image = s_image * (1 - use_mask) + delta * use_mask

    model = load_model(str(weights_file))
    
    i_shape = model.get_layer(Troj_Layer).output_shape
    ni_shape = model.get_layer(Troj_next_Layer).output_shape

    t1_model = keras.models.clone_model(model) 
    if model.__class__.__name__ == 'Sequential':
        while t1_model.layers[-1].name != Troj_Layer:
            t1_model.pop()
    else:
        t1_model = custom_pop(t1_model, Troj_Layer_dict[Troj_Layer])

    tinners = t1_model(i_image)

    t2_model = keras.models.clone_model(model) 
    if model.__class__.__name__ == 'Sequential':
        while t2_model.layers[-1].name != Troj_next_Layer:
            t2_model.pop()
    else:
        t2_model = custom_pop(t2_model, Troj_Layer_dict[Troj_next_Layer])

    ntinners = t2_model(i_image)

    t3_model = keras.models.clone_model(model)
    if has_softmax:
        if model.__class__.__name__ == 'Sequential':
            t3_model.pop()
        else:
            t3_model = custom_pop(t3_model)
    logits = t3_model(i_image)

    models = [model, t1_model, t2_model, t3_model]

    return models, i_image, s_image, delta, mask, con_mask, tinners, ntinners, i_shape, ni_shape, logits
    
def define_graph(optz_option, Troj_Layer, Troj_Neuron, Troj_next_Layer, Troj_next_Neuron, variables1, Troj_size=64):
    models, i_image, s_image, delta, mask, con_mask, tinners, ntinners, i_shape, ni_shape, logits = variables1
    
    if len(i_shape) == 2:
        i_shape = [1, i_shape[1]]
        ni_shape = [1, ni_shape[1]]
    elif len(i_shape) == 4:
        i_shape = [1, i_shape[1], i_shape[2], i_shape[3]]
        ni_shape = [1, ni_shape[1], ni_shape[2], ni_shape[3]]
    idxs = np.zeros(i_shape)
    if len(i_shape) == 2:
        idxs[:, Troj_Neuron] = 1
    elif len(i_shape) == 4:
        if channel_last:
            idxs[:,:,:, Troj_Neuron] = 1
        else:
            idxs[:, Troj_Neuron,:,:] = 1
    nidxs = np.zeros(ni_shape)
    if len(ni_shape) == 2:
        idxs[:, Troj_next_Neuron] = 1
    elif len(ni_shape) == 4:
        if channel_last:
            nidxs[:,:,:, Troj_next_Neuron] = 1
        else:
            nidxs[:, Troj_next_Neuron,:,:] = 1

    vloss1 = tf.reduce_sum(tinners * idxs)
    vloss2 = tf.reduce_sum(tinners * (1-idxs))
    relu_loss1 = tf.reduce_sum(ntinners * nidxs)
    relu_loss2 = tf.reduce_sum(ntinners * (1-nidxs))
    tvloss = tf.reduce_sum(tf.image.total_variation(delta))
    loss = - vloss1 - relu_loss1  + 0.0001 * vloss2 + 0.0001 * relu_loss2 # + 0.01 * tvloss
    # loss = - vloss1 - relu_loss1 + 0.00001 * vloss2 + 0.00001 * relu_loss2 # + 0.01 * tvloss

    # vloss1 = tf.reduce_mean(tinners * idxs)
    # vloss2 = tf.reduce_mean(tinners * (1-idxs))
    # relu_loss1 = tf.reduce_mean(ntinners * nidxs)
    # relu_loss2 = tf.reduce_mean(ntinners * (1-nidxs))
    # tvloss = tf.reduce_sum(tf.image.total_variation(delta))
    # loss = - vloss1 - relu_loss1 + 10 * (vloss2 + relu_loss2) # + 0.01 * tvloss
    # loss *= 10

    mask_loss = tf.reduce_sum(con_mask)
    mask_cond1 = tf.greater(mask_loss, tf.constant(float(Troj_size)))
    mask_cond2 = tf.greater(mask_loss, tf.constant(float( (np.sqrt(Troj_size)+2)**2  )))

    mask_nz = tf.count_nonzero(tf.nn.relu(con_mask - mask_epsilon), dtype=tf.int32)
    if count_mask:
        mask_cond1 = tf.greater(mask_nz, tf.constant(Troj_size))
        mask_cond2 = tf.greater(mask_nz, tf.constant(int((np.sqrt(Troj_size)+2)**2)))
    
    loss += tf.cond(mask_cond1, true_fn=lambda: tf.cond(mask_cond2, true_fn=lambda: 2 * re_mask_weight * mask_loss, false_fn=lambda: 1 * re_mask_weight * mask_loss), false_fn=lambda: 0.0 * mask_loss)
    lr = re_mask_lr

    if use_mask:
        train_op = tf.train.AdamOptimizer(lr).minimize(loss, var_list=[delta, mask])
    else:
        train_op = tf.train.AdamOptimizer(lr).minimize(loss, var_list=[delta])
    grads = tf.gradients(loss, delta)
    return models, s_image, tinners, logits, loss, vloss1, vloss2, tvloss, relu_loss1, relu_loss2, i_image, delta, mask, con_mask, train_op, grads, i_shape, ni_shape, mask_nz, mask_loss, mask_cond1
    
def reverse_engineer(optz_option, images, weights_file, Troj_Layer, Troj_Neuron, Troj_next_Layer, Troj_next_Neuron, Troj_Label, variables2, RE_img = './adv.png', RE_delta='./delta.pkl', RE_mask = './mask.pkl', Troj_size=64):

    models, s_image, tinners, logits, loss, vloss1, vloss2, tvloss, relu_loss1,\
            relu_loss2, i_image, delta, mask, con_mask, train_op, grads, i_shape,\
            ni_shape, mask_nz, mask_loss, mask_cond1 = variables2

    Troj_Idx = Troj_Idx_dict[Troj_Layer]
    Troj_next_Idx = Troj_Idx_dict[Troj_next_Layer]

    if use_mask:
        mask_init = filter_img(h,w)*4-2
    else:
        mask_init = filter_img(h,w)*8-4
    
    if channel_last:
        delta_init = np.random.normal(np.float32([0]), 1, (1,h,w,3))
    else:
        delta_init = np.random.normal(np.float32([0]), 1, (1,3,h,w))
    
    with tf.Session() as sess:
    
        init = tf.global_variables_initializer()
        sess.run(init)
        sess.run(delta.assign(delta_init))
        sess.run(mask.assign(mask_init))


        models[0].load_weights(weights_file)
        models[1].set_weights(models[0].get_weights()[:Troj_Idx])
        models[2].set_weights(models[0].get_weights()[:Troj_next_Idx])
        models[3].set_weights(models[0].get_weights())
    
        ot_loss = 0 
        oo_loss = 0 
        nt_loss = 0 
        no_loss = 0 
        # optimizing using Adam optimizer
        K.set_learning_phase(0)
        if optz_option == 0:
            rlogits, rloss, rloss1, rloss2, rtvloss, rrelu_loss1, rrelu_loss2, adv, rdelta = \
                    sess.run((logits, loss, vloss1, vloss2, tvloss, relu_loss1, relu_loss2, i_image, delta),\
                    {s_image:images})
            ot_loss = rrelu_loss1
            oo_loss = rrelu_loss2
            for e in range(1000):
                rinner, rlogits, rloss, rloss1, rloss2, rtvloss, rrelu_loss1, rrelu_loss2, rmask_nz, rmask_cond1, rmask_loss, adv, rdelta,_  = \
                        sess.run((tinners, logits, loss, vloss1, vloss2, tvloss, relu_loss1, relu_loss2, mask_nz, mask_cond1, mask_loss, i_image, delta, train_op),\
                        {s_image:images})
                if Print_Level > 1:
                    if e % 10 == 0:
                        print('e', e, 'loss', rloss, 'target loss', rloss1, 'other loss', rloss2, 'tv loss', rtvloss)
                        print('next layer loss', 'target loss', rrelu_loss1, 'other loss', rrelu_loss2)
                        print('mask nz', rmask_nz, 'loss', rmask_loss, 'cond 1', rmask_cond1)
                        if len(i_shape) == 2:
                            print('result', np.sum(np.argmax(rlogits,axis=1)==Troj_Label), rlogits[:,Troj_Label], 'neuron', np.sum(rinner[0,Troj_Neuron]), np.amax(rinner[0,Troj_Neuron]))
                        elif len(i_shape) == 4:
                            if channel_last:
                                print('result', np.sum(np.argmax(rlogits,axis=1)==Troj_Label), rlogits[:,Troj_Label], 'neuron', np.sum(rinner[0,:,:,Troj_Neuron]), np.amax(rinner[0,:,:,Troj_Neuron]))
                            else:
                                print('result', np.sum(np.argmax(rlogits,axis=1)==Troj_Label), rlogits[:,Troj_Label], 'neuron', np.sum(rinner[0,Troj_Neuron,:,:]), np.amax(rinner[0,Troj_Neuron,:,:]))
            rlogits, rloss, rloss1, rloss2, rtvloss, rrelu_loss1, rrelu_loss2, adv, rdelta = \
                    sess.run((logits, loss, vloss1, vloss2, tvloss, relu_loss1, relu_loss2, i_image, delta),\
                    {s_image:images})
            nt_loss = rrelu_loss1
            no_loss = rrelu_loss2
    
        rlogits, rloss, rloss1, rloss2, rtvloss, rrelu_loss1, rrelu_loss2, rmask_loss, rcon_mask, rmask_nz, adv, rdelta = \
                sess.run((logits, loss, vloss1, vloss2, tvloss, relu_loss1, relu_loss2, mask_loss, con_mask, mask_nz, i_image, delta),\
                {s_image:images})
        if Print_Level > 1:
            print('loss', rloss, 'target loss', rloss1, 'other loss', rloss2, 'tv loss', rtvloss)
            print('next layer loss', 'target loss', rrelu_loss1, 'other loss', rrelu_loss2)
            print('mask nz', rmask_nz, 'loss', rmask_loss, 'cond 1', rmask_cond1)
        if channel_last:
            adv = np.clip(adv, l_bounds, h_bounds)
        else:
            adv = np.clip(adv, l_bounds_channel_first, h_bounds_channel_first)
        adv = cifar.deprocess(adv).astype('uint8')
        # for i in range(adv.shape[0]):
            # print('adv', np.amin(adv[i]), np.amax(adv[i]))
       	    # Image.fromarray(adv[i].astype('uint8')).save(RE_img[:-4]+'_{0}.png'.format(i))
            # imageio.imwrite(RE_img[:-4]+'_{0}.png'.format(i), adv[i])
        # with open(RE_delta, 'wb') as f:
        #     pickle.dump(rdelta, f)
        # with open(RE_mask, 'wb') as f:
        #     pickle.dump(rcon_mask, f)
        # flog = open('result_par_mul.txt', 'a')
        # r = ''
        # for t in rlogits:
        #     r += str(t) + '_'
        # flog.write('maxlabel {0}  Troj size {5}\nmask loss {1}\nmask nonzero {2}\nlabels {3} \nlogits {4}\n'.format(np.sum(np.argmax(rlogits, axis=1) == Troj_Label), rmask_loss, rmask_nz, np.argmax(rlogits, axis=1), r, Troj_size))
        # flog.write('original_target_loss {0} original_other_loss {1} re_target_loss {2} re_other_loss {3}\n'.format(ot_loss, oo_loss, nt_loss, no_loss))
        # flog.close()
        preds = np.argmax(rlogits, axis=1) 
        if Print_Level > 1:
            print(preds)
        # Troj_Label = np.argmax(np.bincount(preds))
        acc = np.sum(preds == Troj_Label)/float(rlogits.shape[0])
        return acc, adv, rdelta, rcon_mask, Troj_Label

def re_mask(neuron_dict, layers, images):
    validated_results = []
    for key in sorted(neuron_dict.keys()):
        weights_file = key
        for task in neuron_dict[key]:
            Troj_Layer, Troj_Neuron, Troj_Label = task
            Troj_Neuron = int(Troj_Neuron)
            Troj_next_Layer = layers[layers.index(Troj_Layer) + 1]
            Troj_next_Neuron = Troj_Neuron
            optz_option = 0
            RE_img = './imgs/{0}_model_{1}_{2}_{3}_{4}.png'.format(weights_file.split('/')[-1][:-3], Troj_Layer, Troj_Neuron, Troj_size, Troj_Label)
            RE_mask = './masks/{0}_model_{1}_{2}_{3}_{4}'.format(weights_file.split('/')[-1][:-3], Troj_Layer, Troj_Neuron, Troj_size, Troj_Label)
            RE_delta = './deltas/{0}_model_{1}_{2}_{3}_{4}'.format(weights_file.split('/')[-1][:-3], Troj_Layer, Troj_Neuron, Troj_size, Troj_Label)
            # flog = open('result_par_mul.txt', 'a')
            # flog.write('\n\n{0} {1} {2} {3} {4} {5}\n\n'.format(optz_option, weights_file, Troj_Layer, Troj_next_Layer, Troj_Neuron, Troj_Label))
            # flog.close()
            
            max_acc = 0
            max_results = []
            for i  in range(mask_multi_start):
                variables1 = setup_model(optz_option, weights_file, Troj_Layer, Troj_next_Layer)
                variables2 = define_graph(optz_option, Troj_Layer, Troj_Neuron, Troj_next_Layer, Troj_next_Neuron, variables1, Troj_size)
                acc, rimg, rdelta, rmask,Troj_Label = reverse_engineer(optz_option, images, weights_file, Troj_Layer, Troj_Neuron, Troj_next_Layer, Troj_next_Neuron, Troj_Label, variables2, RE_img, RE_delta, RE_mask, Troj_size)
                # print('Acc', acc)
                if Print_Level > 0:
                    print('RE mask', Troj_Layer, Troj_Neuron, 'Label', Troj_Label,'RE acc', acc)
                K.clear_session()
                tf.reset_default_graph()
                if acc > max_acc:
                    max_acc = acc
                    max_results = (rimg, rdelta, rmask, Troj_Label, RE_img, RE_mask, RE_delta)
            if max_acc >= reasr_bound - 0.2:
                validated_results.append( max_results )
        return validated_results


def filter_load_model(optz_option, weights_file, Troj_Layer, Troj_next_Layer):
    if channel_last:
        s_image  = tf.placeholder(tf.float32, shape=(None, h, w, 3))
        si_image = s_image
    else:
        s_image = tf.placeholder(tf.float32, shape=(None, 3, h, w))
        si_image = tf.transpose(s_image, [0,2,3,1])

    deltas = []
    with tf.variable_scope("", reuse=tf.AUTO_REUSE):
        fdelta= tf.get_variable("fdelta", [12, 3], constraint=lambda x: tf.clip_by_value(x, -1, 1))
        imax =  tf.nn.max_pool( si_image, ksize=[1,window_size,window_size,1], strides=[1,1,1,1], padding='SAME')
        imin = -tf.nn.max_pool(-si_image, ksize=[1,window_size,window_size,1], strides=[1,1,1,1], padding='SAME')
        iavg =  tf.nn.avg_pool( si_image, ksize=[1,window_size,window_size,1], strides=[1,1,1,1], padding='SAME')
        i_image = tf.reshape( tf.matmul( tf.reshape( tf.concat([si_image, imax, imin, iavg], axis=3), (-1,12)) , fdelta), [-1,h,w,3])
        deltas.append(fdelta)

    i_image = tf.clip_by_value(i_image, l_bounds, h_bounds)

    if not channel_last:
        i_image = tf.transpose(i_image, [0,3,1,2])
    
    model = load_model(str(weights_file))
    
    i_shape = model.get_layer(Troj_Layer).output_shape
    ni_shape = model.get_layer(Troj_next_Layer).output_shape

    t1_model = keras.models.clone_model(model) 
    if model.__class__.__name__ == 'Sequential':
        while t1_model.layers[-1].name != Troj_Layer:
            t1_model.pop()
    else:
        t1_model = custom_pop(t1_model, Troj_Layer_dict[Troj_Layer])

    tinners = t1_model(i_image)

    t2_model = keras.models.clone_model(model) 
    if model.__class__.__name__ == 'Sequential':
        while t2_model.layers[-1].name != Troj_next_Layer:
            t2_model.pop()
    else:
        t2_model = custom_pop(t2_model, Troj_Layer_dict[Troj_next_Layer])

    ntinners = t2_model(i_image)

    t3_model = keras.models.clone_model(model)
    if has_softmax:
        if model.__class__.__name__ == 'Sequential':
            t3_model.pop()
        else:
            t3_model = custom_pop(t3_model)
    logits = t3_model(i_image)


    models = [model, t1_model, t2_model, t3_model]

    return models, i_image, s_image, deltas, tinners, ntinners, i_shape, ni_shape, logits
    
def filter_define_graph(optz_option, Troj_Layer, Troj_next_Layer, Troj_Neuron, Troj_next_Neuron, variables1):
    models, i_image, s_image, deltas, tinners, ntinners, i_shape, ni_shape, logits = variables1
    
    
    if len(i_shape) == 2:
        i_shape = [1, i_shape[1]]
        ni_shape = [1, ni_shape[1]]
    elif len(i_shape) == 4:
        i_shape = [1, i_shape[1], i_shape[2], i_shape[3]]
        ni_shape = [1, ni_shape[1], ni_shape[2], ni_shape[3]]
    idxs = np.zeros(i_shape)
    if len(i_shape) == 2:
        idxs[:, Troj_Neuron] = 1
    elif len(i_shape) == 4:
        if channel_last:
            idxs[:,:,:, Troj_Neuron] = 1
        else:
            idxs[:, Troj_Neuron,:,:] = 1
    nidxs = np.zeros(ni_shape)
    if len(ni_shape) == 2:
        idxs[:, Troj_next_Neuron] = 1
    elif len(ni_shape) == 4:
        if channel_last:
            nidxs[:,:,:, Troj_next_Neuron] = 1
        else:
            nidxs[:, Troj_next_Neuron,:,:] = 1
    vloss1 = tf.reduce_sum(tinners * idxs)
    vloss2 = tf.reduce_sum(tinners * (1-idxs))
    relu_loss1 = tf.reduce_sum(ntinners * nidxs)
    relu_loss2 = tf.reduce_sum(ntinners * (1-nidxs))
    
    tvloss = tf.reduce_sum(tf.image.total_variation(i_image))

    # lr1 = 1e-3
    lr1 = 2e-3
    # lr1 = 1e-1
    loss = - vloss1 - relu_loss1 + 0.00001 * vloss2 + 0.00001 * relu_loss2 # + 0.001 * tvloss 
    diff_img_loss = tf.reduce_sum((s_image - i_image) ** 2)
    l_cond = tf.greater(diff_img_loss, tf.constant(6000.0))

    if channel_last:
        ssim_loss = -tf.reduce_sum(tf.image.ssim(s_image, i_image, np.amax(h_bounds) - np.amin(l_bounds)))
    else:
        ssim_loss = -tf.reduce_sum(tf.image.ssim( tf.transpose(s_image, [0,2,3,1]), tf.transpose(i_image, [0,2,3,1]), np.amax(h_bounds) - np.amin(l_bounds)))
    l_cond2 = tf.greater(ssim_loss, tf.constant(-0.2*10))

    loss = 0.01 * loss +  tf.cond(l_cond2, true_fn=lambda: 10000 * ssim_loss, false_fn=lambda: 10 * ssim_loss)
    # loss = 0.02 * loss +  tf.cond(l_cond2, true_fn=lambda: 10000 * ssim_loss, false_fn=lambda: 10 * ssim_loss)
    # loss = 0.1 * loss +  tf.cond(l_cond2, true_fn=lambda: 10000 * ssim_loss, false_fn=lambda: 10 * ssim_loss)

    train_op = tf.train.AdamOptimizer(lr1).minimize(loss, var_list=deltas)
    grads = tf.gradients(loss, deltas)
    return models, s_image, tinners, logits, loss, vloss1, vloss2, tvloss, relu_loss1, relu_loss2, diff_img_loss, i_image, deltas, train_op, grads, l_cond, l_cond2, ssim_loss, i_shape, ni_shape
    
def filter_reverse_engineer(optz_option, images, weights_file, Troj_Layer, Troj_next_Layer, Troj_Neuron, variables2, RE_img = './adv.png', RE_delta='./delta.pkl', Troj_Label = 0):

    models, s_image, tinners, logits, loss, vloss1, vloss2, tvloss, relu_loss1, relu_loss2,\
            diff_img_loss, i_image, deltas, train_op, grads, l_cond, l_cond2, ssim_loss, i_shape, ni_shape = variables2
    delta = deltas[0]

    Troj_Idx = Troj_Idx_dict[Troj_Layer]
    Troj_next_Idx = Troj_Idx_dict[Troj_next_Layer]
    
    with tf.Session() as sess:
    
        init = tf.global_variables_initializer()
        sess.run(init)

        delta_init = np.concatenate([np.eye(3), np.zeros((9,3))], axis=0)
        sess.run(delta.assign(delta_init))

        models[0].load_weights(weights_file)
        models[1].set_weights(models[0].get_weights()[:Troj_Idx])
        models[2].set_weights(models[0].get_weights()[:Troj_next_Idx])
        models[3].set_weights(models[0].get_weights())
    
        ot_loss = 0 
        oo_loss = 0 
        nt_loss = 0 
        no_loss = 0 
        # optimizing using Adam optimizer
        rlogits, rloss, rloss1, rloss2, rtvloss, rrelu_loss1, rrelu_loss2, rimg_loss, adv, rdelta = \
                sess.run((logits, loss, vloss1, vloss2, tvloss, relu_loss1, relu_loss2, diff_img_loss, i_image, delta),\
                {s_image:images})
        ot_loss = rrelu_loss1
        oo_loss = rrelu_loss2
        for e in range(1000):
            rinner, rlogits, rloss, rloss1, rloss2, rtvloss, rrelu_loss1, rrelu_loss2, rimg_loss, adv, rdelta, r_cond, r_cond2, rssim_loss,_  = \
                    sess.run((tinners, logits, loss, vloss1, vloss2, tvloss, relu_loss1, relu_loss2, diff_img_loss, i_image, delta, l_cond, l_cond2, ssim_loss, train_op),\
                    {s_image:images})
            if e % 10 == 0:
                print('e', e, 'loss', rloss, 'target loss', rloss1, 'other loss', rloss2, 'tv loss', rtvloss)
                print('next layer loss', 'target loss', rrelu_loss1, 'other loss', rrelu_loss2)
                if len(i_shape) == 2:
                    print('result', np.sum(np.argmax(rlogits,axis=1)==Troj_Label), rlogits[:,Troj_Label], 'neuron', np.sum(rinner[0,Troj_Neuron]), np.amax(rinner[0,Troj_Neuron]))
                elif len(i_shape) == 4:
                    if channel_last:
                        print('result', np.sum(np.argmax(rlogits,axis=1)==Troj_Label), rlogits[:,Troj_Label], 'neuron', np.sum(rinner[0,:,:,Troj_Neuron]), np.amax(rinner[0,:,:,Troj_Neuron]))
                    else:
                        print('result', np.sum(np.argmax(rlogits,axis=1)==Troj_Label), rlogits[:,Troj_Label], 'neuron', np.sum(rinner[0,Troj_Neuron,:,:]), np.amax(rinner[0,Troj_Neuron,:,:]))
        rlogits, rloss, rloss1, rloss2, rtvloss, rrelu_loss1, rrelu_loss2, rimg_loss, adv, rdelta, r_cond2, rssim_loss = \
                sess.run((logits, loss, vloss1, vloss2, tvloss, relu_loss1, relu_loss2, diff_img_loss, i_image, delta, l_cond2, ssim_loss),\
                {s_image:images})
        nt_loss = rrelu_loss1
        no_loss = rrelu_loss2
    
        rlogits, rloss, rloss1, rloss2, rtvloss, rrelu_loss1, rrelu_loss2, rimg_loss, adv, rdelta, r_cond2, rssim_loss = \
                sess.run((logits, loss, vloss1, vloss2, tvloss, relu_loss1, relu_loss2, diff_img_loss, i_image, delta, l_cond2, ssim_loss),\
                {s_image:images})
        if channel_last:
            adv = np.clip(adv, l_bounds, h_bounds)
        else:
            adv = np.clip(adv, l_bounds_channel_first, h_bounds_channel_first)
        adv = cifar.deprocess(adv).astype('uint8')
        # print('adv', np.amin(adv), np.amax(adv))
        preds = np.argmax(rlogits, axis=1) 
        # Troj_Label = np.argmax(np.bincount(preds))
        acc = np.sum(preds == Troj_Label)/float(rlogits.shape[0])
        # acc = np.sum(np.argmax(rlogits, axis=1) == Troj_Label)/float(rlogits.shape[0])
        return acc, adv, rdelta, Troj_Label

def re_filter(neuron_dict, layers, processed_xs):
    validated_results = []
    for key in sorted(neuron_dict.keys()):
        weights_file = key
        for task in neuron_dict[key]:
            Troj_Layer, Troj_Neuron, Troj_Label = task
            Troj_Neuron = int(Troj_Neuron)
            Troj_next_Neuron = Troj_Neuron
            Troj_next_Layer = layers[layers.index(Troj_Layer) + 1]
            Troj_next_Neuron = Troj_Neuron
            optz_option = 0

            max_acc = 0
            max_results = []
            for i  in range(filter_multi_start):
                RE_img = './imgs/filter_{0}_model_{1}_{2}_{3}_{4}.png'.format(weights_file.split('/')[-1][:-3], Troj_Layer, Troj_Neuron, Troj_size, Troj_Label)
                RE_delta = './deltas/filter_{0}_model_{1}_{2}_{3}_{4}'.format(weights_file.split('/')[-1][:-3], Troj_Layer, Troj_Neuron, Troj_size, Troj_Label)
                
                variables1 = filter_load_model(optz_option, weights_file, Troj_Layer, Troj_next_Layer)
                variables2 = filter_define_graph(optz_option, Troj_Layer, Troj_next_Layer, Troj_Neuron, Troj_next_Neuron, variables1)
                acc, rimg, rdelta,Troj_Label = filter_reverse_engineer(optz_option, processed_xs, weights_file, Troj_Layer, Troj_next_Layer, Troj_Neuron, variables2, RE_img, RE_delta, Troj_Label)
                # print('Acc', acc)
                if Print_Level > 0:
                    print('RE filter', Troj_Layer, Troj_Neuron, 'RE acc', acc)
                K.clear_session()
                tf.reset_default_graph()
                if acc > max_acc:
                    max_acc = acc
                    max_results = (rimg, rdelta, Troj_Label, RE_img, RE_delta)
            if max_acc >= reasr_bound - 0.2:
                validated_results.append( max_results )
        return validated_results


def stamp(n_img, delta, mask):
    mask0 = nc_filter_img(w,h)
    mask = mask * mask0
    r_img = n_img.copy()
    for i in range(h):
        for j in range(w):
            if channel_last:
                r_img[:,i,j,:] = n_img[:,i,j,:]*(1-mask[i,j]) + delta[:,i,j,:]*mask[i,j]
            else:
                r_img[:,:,i,j] = n_img[:,:,i,j]*(1-mask[i,j]) + delta[:,:,i,j]*mask[i,j]
    return r_img

def filter_stamp(n_img, trigger):
    if channel_last:
        t_image = tf.placeholder(tf.float32, shape=(None, h, w, 3))
        ti_image = t_image
    else:
        t_image = tf.placeholder(tf.float32, shape=(None, 3, h, w))
        ti_image = tf.transpose(t_image, [0,2,3,1])

    tdelta = tf.placeholder(tf.float32, shape=(12, 3))
    imax =  tf.nn.max_pool( ti_image, ksize=[1,window_size,window_size,1], strides=[1,1,1,1], padding='SAME')
    imin = -tf.nn.max_pool(-ti_image, ksize=[1,window_size,window_size,1], strides=[1,1,1,1], padding='SAME')
    iavg =  tf.nn.avg_pool( ti_image, ksize=[1,window_size,window_size,1], strides=[1,1,1,1], padding='SAME')
    i_image = tf.reshape( tf.matmul( tf.reshape( tf.concat([ti_image, imax, imin, iavg], axis=3), (-1,12)) , tdelta), [-1,h,w,3])

    i_image = tf.clip_by_value(i_image, l_bounds, h_bounds)

    if not channel_last:
        i_image = tf.transpose(i_image, [0,3,1,2])

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

    saved_images = cifar.deprocess(t_images).astype('uint8')
    for i in range(len(t_images)):
        imageio.imsave(tdname + '/' + '{0}.png'.format(i), saved_images[i])

    nt_images = cifar.deprocess(t_images).astype('uint8')
    rt_images = cifar.preprocess(nt_images)
    if Print_Level > 0:
        print(np.amin(rt_images), np.amax(rt_images))
    
    yt = np.zeros(len(rt_images)).astype(np.int32) + tlabel
    preds = model.predict(rt_images, verbose=0)
    preds = np.argmax(preds, axis=1) 
    score = np.sum(yt == preds)/float(yt.shape[0])
    return score

if __name__ == '__main__':
    # config['model_file'] = sys.argv[1]
# def main():
    if use_pickle:
        fxs, fys = pickle.load(open(seed_file, 'rb'))
    else:
        h5f = h5py.File(seed_file, 'r')
        fxs = h5f['x'][:]
        fys = h5f['y'][:]
    print('number of seed images', len(fys), fys.shape)
    fys = fys.reshape([-1])
    # if len(fys) == 10:
    #     xs = fxs[:4]
    #     ys = fys[:4]
    # elif len(fys) == 50:
    #     xs = fxs[:10]
    #     ys = fys[:10]
    # else:
    if True:
        xs = fxs[:len(fys)//3]
        ys = fys[:len(fys)//3]
    if Print_Level > 0:
        print('# samples for RE', len(ys))
    test_xs = fxs
    test_ys = fys
    model = load_model(str(config['model_file']))
    if Print_Level > 0:
        model.summary()
    for i in range(len(model.layers)):
        Troj_Layer_dict[model.layers[i].name] = i
    n_weights = 0
    for i in range(len(model.layers)):
        n_weights += len(model.layers[i].get_weights())
        Troj_Idx_dict[model.layers[i].name] = n_weights

    layers = [l.name for l in model.layers]
    processed_xs = cifar.preprocess(xs)
    processed_test_xs = cifar.preprocess(test_xs)
    if Print_Level > 0:
        print('image range', np.amin(processed_test_xs), np.amax(processed_test_xs))
    neuron_dict = {}

    maxes = check_values(processed_test_xs, test_ys, model)
    all_ps = sample_neuron(processed_test_xs, test_ys, model, maxes)
    neuron_dict = read_all_ps(config['model_file'], all_ps, top_k = top_n_neurons)
    print('Compromised Neuron Candidates (Layer, Neuron, Target_Label)', neuron_dict)

    # sys.exit()

    # neuron_dict['./models/nin_trojan_filter_2_3.h5'] = [('conv2d_4', 20, 0)]

    # mask check 
    maxreasr = 0
    reasr_info = []

    results = re_mask(neuron_dict, layers, processed_xs)
    if len(results) > 0:
        reasrs = []
        for result in results:
            reasr = test(str(config['model_file']), test_xs, result)
            reasrs.append(reasr)
            adv, rdelta, rmask, Troj_Label, RE_img, RE_mask, RE_delta = result
            rmask = rmask * rmask > mask_epsilon
            if reasr > reasr_bound:
                for i in range(adv.shape[0]):
                    imageio.imwrite(RE_img[:-4]+'_{0}.png'.format(i), adv[i])
                if use_pickle:
                    with open(RE_delta+'.pkl', 'wb') as f:
                        pickle.dump(rdelta, f)
                    with open(RE_mask+'.pkl', 'wb') as f:
                        pickle.dump(rmask, f)
                if use_h5:
                    with h5py.File(RE_delta+'.h5', "w") as f:
                        f.create_dataset('delta', data=rdelta)
                    with h5py.File(RE_mask+'.h5', "w") as f:
                        f.create_dataset('mask', data=rmask)
            reasr_info.append([reasr, 'mask', str(Troj_Label), RE_img, RE_mask, RE_delta])
            if reasr > maxreasr:
                maxreasr = reasr
        print(str(config['model_file']), 'mask check', max(reasrs))
    else:
        print(str(config['model_file']), 'mask check', 0)

    # filter check 
    results = re_filter(neuron_dict, layers, processed_xs)
    if len(results) > 0:
        reasrs = []
        for result in results:
            reasr = test(str(config['model_file']), test_xs, result, 'filter')
            reasrs.append(reasr)
            adv, rdelta, Troj_Label, RE_img, RE_delta = result
            if reasr > reasr_bound:
            # if True:
                for i in range(adv.shape[0]):
                    imageio.imwrite(RE_img[:-4]+'_{0}.png'.format(i), adv[i])
                if use_pickle:
                    with open(RE_delta+'.pkl', 'wb') as f:
                        pickle.dump(rdelta, f)
                if use_h5:
                    with h5py.File(RE_delta+'.h5', "w") as f:
                        f.create_dataset('delta', data=rdelta)
            reasr_info.append([reasr, 'filter', str(Troj_Label), RE_img, RE_delta])
            if reasr > maxreasr:
                maxreasr = reasr
        print(str(config['model_file']), 'filter check', max(reasrs))
    else:
        print(str(config['model_file']), 'filter check', 0)

    print(str(config['model_file']), 'both filter and mask check', maxreasr)
    for info in reasr_info:
        print('reasr info', info)

