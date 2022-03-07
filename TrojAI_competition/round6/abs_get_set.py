# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

for_submission = True

import os, sys
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
import math
import random
import jsonpickle
import time
import pickle
from sklearn.cluster import DBSCAN, KMeans

import warnings
# trans
warnings.filterwarnings("ignore")

asr_bound = 0.9

mask_epsilon = 0.01
max_input_length = 80
use_amp = False  # attempt to use mixed precision to accelerate embedding conversion process
top_k_candidates = 5
n_arms = 5
n_max_imgs_per_label = 20

nrepeats = 1
max_neuron_per_label = 1
mv_for_each_label = True
tasks_per_run = 1
top_n_check_labels = 3

config = {}
if for_submission:
    config['gpu_id'] = '0'
else:
    config['gpu_id'] = '0'
config['print_level'] = 2
config['random_seed'] = 333
config['channel_last'] = 0
config['w'] = 224
config['h'] = 224
config['reasr_bound'] = 0.0
config['batch_size'] = 5
config['has_softmax'] = 0
config['samp_k'] = 2.
config['same_range'] = 0
config['n_samples'] = 3
config['samp_batch_size'] = 32
config['top_n_neurons'] = 3
config['n_sample_imgs_per_label'] = 2
if not for_submission:
    config['re_batch_size'] = 10
else:
    config['re_batch_size'] = 20
config['max_troj_size'] = 1200
config['filter_multi_start'] = 1
# config['re_mask_lr'] = 5e-2
# config['re_mask_lr'] = 3e-2
# config['re_mask_lr'] = 1e0
# config['re_mask_lr'] = 1e1
config['re_mask_lr'] = 2e1
# config['re_mask_lr'] = 4e1
# config['re_mask_lr'] = 6e1
# config['re_mask_lr'] = 5e-1
# config['re_mask_lr'] = 1e-1
# config['re_mask_lr'] = 2e-1
config['re_mask_weight'] = 100
config['mask_multi_start'] = 1
# config['re_epochs'] = 100
config['re_epochs'] = 5
config['n_re_imgs_per_label'] = 20
config['trigger_length'] = 1
config['logfile'] = './result_get_set.txt'
# config['logfile'] = './result_r5_v1_bms_5_2locs_3_fast_5_5_marms9_test.txt'

if not for_submission:
    dbert_use_idxs_fname = './dbert_idxs6.txt'
    gpt2_use_idxs_fname = './gpt2_idxs6.txt'
    benign_model_base_dir = './r5_models/'
else:
    dbert_use_idxs_fname = '/dbert_idxs6.txt'
    gpt2_use_idxs_fname = '/gpt2_idxs6.txt'
    benign_model_base_dir = '/r5_models/'

# use_idxs = range(50256+1)
dbert_use_idxs = []
for line in open(dbert_use_idxs_fname):
    dbert_use_idxs.append(int(line.split()[0]))

gpt2_use_idxs = []
for line in open(gpt2_use_idxs_fname):
    gpt2_use_idxs.append(int(line.split()[0]))

print('dbert_use_idxs', len(dbert_use_idxs), dbert_use_idxs, )
print('gpt2_use_idxs', len(gpt2_use_idxs), gpt2_use_idxs, )

# use_idxs = [7744, 20039, 22089, 13902, 42582, 15962, 5729, 5734, 46699, 9846, 23168]
# use_idxs = [15962, 5729, 5734, 46699, 9846, 10588]

trigger_length = config['trigger_length']
reasr_bound = float(config['reasr_bound'])
top_n_neurons = int(config['top_n_neurons'])
batch_size = config['batch_size']
has_softmax = bool(config['has_softmax'])
Print_Level = int(config['print_level'])
re_epochs = int(config['re_epochs'])
mask_multi_start = int(config['mask_multi_start'])
n_re_imgs_per_label = int(config['n_re_imgs_per_label'])
n_sample_imgs_per_label = int(config['n_sample_imgs_per_label'])
re_mask_lr = float(config['re_mask_lr'])

channel_last = bool(config['channel_last'])
random_seed = int(config['random_seed'])
os.environ["CUDA_VISIBLE_DEVICES"] = config["gpu_id"]

torch.backends.cudnn.enabled = False
# deterministic
torch.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)


# class LstmLinearModel(torch.nn.Module):
#     def __init__(self, input_size: int, hidden_size: int, output_size: int, dropout: float, bidirectional: bool, n_layers: int):
#         super().__init__()        
#         self.rnn = torch.nn.LSTM(input_size,
#                 hidden_size,
#                 num_layers=n_layers,
#                 bidirectional=bidirectional,
#                 batch_first=True,
#                 dropout=0 if n_layers < 2 else dropout)       

#         self.linear = torch.nn.Linear(hidden_size * 2 if bidirectional else hidden_size, output_size)
#         self.dropout = torch.nn.Dropout(dropout)    

#     def forward(self, data):
#         # input data is after the embedding        
#         # data = [batch size, sent len, emb dim]
#         # _, hidden = self.rnn(data)        
#         output, (hn, cn) = self.rnn(data)        
#         hidden = output
#         # hidden = [n layers * n directions, batch size, emb dim]
#         # print('hidden', hidden.shape)
#         # if self.rnn.bidirectional:
#         #     hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
#         # else:
#         #     hidden = self.dropout(hidden[-1, :, :])        
#         hidden = self.dropout(hidden[:, -1, :])
#         # hidden = [batch size, hid dim]
#         output = self.linear(hidden)
#         # output = [batch size, out dim]        
#         return output

def get_embedding_from_ids(input_ids, embedding, attention_mask, cls_token_is_first, end_poses=[]):
    # print('input_ids', input_ids.shape)
    embedding_vector = embedding(input_ids, attention_mask=attention_mask)[0]
    if cls_token_is_first:
        embedding_vector = embedding_vector[:, :1, :]
    else:
        # embedding_vector = embedding_vector[:, -1:, :]
        if len(end_poses) == 0:
            embedding_vector = embedding_vector[:, -1:, :]
        else:
            print(end_poses, end_poses)
            embedding_vector = embedding_vector[:, np.array(end_poses), :]
    return embedding_vector

def check_values(embedding_vectors, labels, model, children, target_layers, num_classes):
    maxes = {}
    maxes_per_label  = {}
    end_layer = len(children)-1
    if has_softmax:
        end_layer = len(children)-2

    sample_layers = []
    for layer_i in range(0, end_layer):
        print(children[layer_i].__class__.__name__, target_layers)
        if not children[layer_i].__class__.__name__ in target_layers:
            continue
        sample_layers.append(layer_i)
    model_type = model.__class__.__name__
    if model_type  == 'FCLinearModel':
        sample_layers = sample_layers[-2:-1]
    elif  model_type  == 'GruLinearModel':
        sample_layers = sample_layers[-1:]
    elif  model_type  == 'LstmLinearModel':
        sample_layers = sample_layers[-1:]
    print('sample_layers', sample_layers)

    n_neurons_dict = {}
    for layer_i in sample_layers:
        temp_model1 = children[layer_i]
        if model_type  == 'FCLinearModel':
            temp_model1 = torch.nn.Sequential(*children[:layer_i+1])
        print('temp model1', temp_model1.__class__.__name__, layer_i)

        max_vals = []
        for i in range( math.ceil(float(embedding_vectors.shape[0])/batch_size) ):
            batch_data = torch.FloatTensor(embedding_vectors[batch_size*i:batch_size*(i+1)]).cuda()
            if temp_model1.__class__.__name__ == 'LSTM':
                packed_output, (hidden, cell) = temp_model1(batch_data)
                # print('hidden', hidden.shape)
                if temp_model1.bidirectional:
                    hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
                else:
                    hidden = hidden[-1, :, :]
                # print('hidden', hidden.shape, temp_model1.bidirectional)
                inner_outputs = hidden.cpu().detach().numpy()
            elif temp_model1.__class__.__name__ == 'GRU':
                _, hidden = temp_model1(batch_data)
                if temp_model1.bidirectional:
                    hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
                else:
                    hidden = hidden[-1, :, :]
                inner_outputs = hidden.cpu().detach().numpy()
            else:
                inner_outputs = temp_model1(batch_data).cpu().detach().numpy()
                inner_outputs = inner_outputs[:,0,:]
            # print('batch_data', batch_data.shape, inner_outputs.shape, temp_model1.__class__.__name__)
            # sys.exit()
            if channel_last:
                n_neurons = inner_outputs.shape[-1]
            else:
                n_neurons = inner_outputs.shape[1]
            
            n_neurons_dict[layer_i] = n_neurons
            max_vals.append(np.amax(inner_outputs, (1)))
        
        max_vals = np.concatenate(max_vals)

        key = '{0}_{1}'.format(children[layer_i].__class__.__name__, layer_i)
        max_val = np.amax(max_vals)
        maxes[key] = [max_val]
        max_val_per_label = []
        for j in range(num_classes):
            image_idxs = np.array(np.where(labels==j)[0])
            print(j, image_idxs, labels[image_idxs])
            if len(max_vals[image_idxs]) > 0:
                max_val_per_label.append(np.amax(max_vals[image_idxs]))
            else:
                max_val_per_label.append(0)
        maxes_per_label[key] = max_val_per_label
        print('max val', key, max_val, maxes_per_label)

    # check top labels
    flogits = []
    for i in range( math.ceil(float(embedding_vectors.shape[0])/batch_size) ):
        batch_data = torch.FloatTensor(embedding_vectors[batch_size*i:batch_size*(i+1)]).cuda()
        logits = model(batch_data).cpu().detach().numpy()
        flogits.append(logits)
    flogits = np.concatenate(flogits, axis=0)

    print('labels', labels.shape)
    top_check_labels_list = [[] for i in range(num_classes)]
    for i in range(num_classes):
        image_idxs = np.array(np.where(labels==i)[0])
        tlogits = flogits[image_idxs]
        top_check_labels = np.argsort(tlogits, axis=1)[:,-top_n_check_labels-1:-1]
        top_check_labels = top_check_labels.reshape(-1)
        top_check_labels = np.argsort(np.bincount(top_check_labels))[-top_n_check_labels:]
        for top_check_label in top_check_labels:
            label_acc = np.sum(top_check_labels == top_check_label)/float(len(top_check_labels)) * top_n_check_labels
            print(i, 'top_check_label label', top_check_label, label_acc)
            if label_acc > 0.8:
                top_check_labels_list[i].append(top_check_label)

    del temp_model1, batch_data, inner_outputs
    return maxes, maxes_per_label, sample_layers, n_neurons_dict, top_check_labels_list

def sample_neuron(sample_layers, embedding_vectors, labels, model, children, target_layers, model_type, mvs, mvs_per_label):
    all_ps = {}
    samp_k = config['samp_k']
    same_range = config['same_range']
    n_samples = config['n_samples']
    sample_batch_size = config['samp_batch_size']
    # if model_type == 'ResNet':
    #     sample_batch_size = max(sample_batch_size // 2, 1)
    # if model_type == 'DenseNet':
    #     sample_batch_size = max(sample_batch_size // 4, 1)

    end_layer = len(children)-1
    if has_softmax:
        end_layer = len(children)-2

    n_images = embedding_vectors.shape[0]
    if Print_Level > 0:
        print('sampling n imgs', n_images, 'n samples', n_samples, 'children', len(children))

    model_type = model.__class__.__name__

    for layer_i in sample_layers:
        if Print_Level > 0:
            print('layer', layer_i, children[layer_i])
        temp_model1 = children[layer_i]
        if model_type  == 'FCLinearModel':
            temp_model1 = torch.nn.Sequential(*children[:layer_i+1])
        if temp_model1.__class__.__name__ == 'LSTM' or \
                temp_model1.__class__.__name__ == 'GRU' or \
                model_type  == 'FCLinearModel':
            if has_softmax:
                temp_model2 = torch.nn.Sequential(*children[layer_i+1:-1])
            else:
                temp_model2 = torch.nn.Sequential(*children[layer_i+1:])
        else:
            if has_softmax:
                temp_model2 = torch.nn.Sequential(*children[:-1])
            else:
                temp_model2 = torch.nn.Sequential(*children[:])

        if same_range:
            vs = np.asarray([i*samp_k for i in range(n_samples)])
        else:
            mv_key = '{0}_{1}'.format(children[layer_i].__class__.__name__, layer_i)

            # tr = samp_k * max(mvs[mv_key])/(n_samples - 1)
            # vs = np.asarray([i*tr for i in range(n_samples)])

            if mv_for_each_label:
                vs = []
                for label in labels:
                    # mv for each label
                    maxv = mvs_per_label[mv_key][label]
                    e_scale = np.array([0] + [np.power(2., i-1) for i in range(n_samples-1)])
                    tvs = maxv * e_scale
                    # l_scale = np.array([float(i)/(n_samples-1) for i in range(n_samples)])
                    # tvs = maxv * l_scale * samp_k
                    vs.append(tvs)
                vs = np.array(vs)
                vs = vs.T
            else:
                maxv = max(mvs[mv_key])
                e_scale = np.array([0] + [np.power(2., i-1) for i in range(n_samples-1)])
                vs = maxv * e_scale

            print('mv_key', vs.shape, vs)

        for input_i in range( math.ceil(float(n_images)/batch_size) ):
            cbatch_size = min(batch_size, n_images - input_i*batch_size)
            batch_data = torch.FloatTensor(embedding_vectors[batch_size*input_i:batch_size*(input_i+1)]).cuda()
            # inner_outputs = temp_model1(batch_data).cpu().detach().numpy()
            if temp_model1.__class__.__name__ == 'LSTM':
                packed_output, (hidden, cell) = temp_model1(batch_data)
                if temp_model1.bidirectional:
                    hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
                else:
                    hidden = hidden[-1, :, :]
                inner_outputs = hidden.cpu().detach().numpy()
            elif temp_model1.__class__.__name__ == 'GRU':
                _, hidden = temp_model1(batch_data)
                if temp_model1.bidirectional:
                    hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
                else:
                    hidden = hidden[-1, :, :]
                inner_outputs = hidden.cpu().detach().numpy()
            else:
                inner_outputs = temp_model1(batch_data).cpu().detach().numpy()
                inner_outputs = inner_outputs[:,0,:]
                # print('error not support', temp_model1.__class__.__name__)
            if channel_last:
                n_neurons = inner_outputs.shape[-1]
            else:
                n_neurons = inner_outputs.shape[1]

            # n_neurons = 1

            nbatches = math.ceil(float(n_neurons)/sample_batch_size)
            for nt in range(nbatches):
                l_h_t = []
                csample_batch_size = min(sample_batch_size, n_neurons - nt*sample_batch_size)
                for neuron in range(csample_batch_size):

                    # neuron = 1907

                    if len(inner_outputs.shape) == 4:
                        h_t = np.tile(inner_outputs, (n_samples, 1, 1, 1))
                        for i in range(vs.shape[0]):
                            # channel first and len(shape) = 4
                            if mv_for_each_label:
                                v = vs[i,batch_size*input_i:batch_size*input_i+cbatch_size]
                                v = np.reshape(v, [-1, 1, 1])
                            else:
                                v = vs[i]
                            h_t[i*cbatch_size:(i+1)*cbatch_size,neuron+nt*sample_batch_size,:,:] = v
                    else:
                        h_t = np.tile(inner_outputs, (n_samples, 1))
                        for i in range(vs.shape[0]):
                            # channel first and len(shape) = 4
                            if mv_for_each_label:
                                v = vs[i,batch_size*input_i:batch_size*input_i+cbatch_size]
                            else:
                                v = vs[i]
                            # print('h_t', h_t.shape, v.shape, inner_outputs.shape)
                            h_t[i*cbatch_size:(i+1)*cbatch_size,neuron+nt*sample_batch_size] = v


                    l_h_t.append(h_t)

                f_h_t = np.concatenate(l_h_t, axis=0)
                # print(f_h_t.shape, cbatch_size, sample_batch_size, n_samples)

                f_h_t_t = torch.FloatTensor(f_h_t).cuda()
                fps = temp_model2(f_h_t_t).cpu().detach().numpy()
                # if Print_Level > 1:
                #     print(nt, n_neurons, 'inner_outputs', inner_outputs.shape, 'f_h_t', f_h_t.shape, 'fps', fps.shape)
                for neuron in range(csample_batch_size):
                    tps = fps[neuron*n_samples*cbatch_size:(neuron+1)*n_samples*cbatch_size]
                    # print(cbatch_size, inner_outputs.shape, neuron*n_samples*cbatch_size, (neuron+1)*n_samples*cbatch_size, tps.shape)
                    for img_i in range(cbatch_size):
                        img_name = (labels[img_i + batch_size*input_i], img_i + batch_size*input_i)
                        # print(img_i + batch_size*input_i, )
                        ps_key= (img_name, '{0}_{1}'.format(children[layer_i].__class__.__name__, layer_i), neuron+nt*sample_batch_size)
                        ps = [tps[ img_i + cbatch_size*_] for _ in range(n_samples)]
                        ps = np.asarray(ps)
                        ps = ps.T
                        # if neuron+nt*sample_batch_size == 480:
                        #     print('img i', img_i, input_i, cbatch_size, 'neuron', neuron+nt*sample_batch_size, ps_key, ps.shape, ps[3])
                        all_ps[ps_key] = np.copy(ps)

                del f_h_t_t
            del batch_data, inner_outputs
            torch.cuda.empty_cache()

        del temp_model1, temp_model2
    return all_ps, sample_layers


def find_min_max(model_name, sample_layers, neuron_ks, max_ps, max_vals, imgs, n_classes, n_samples, n_imgs, base_l, cut_val=20, top_k = 10, addon=''):
    if base_l >= 0:
        n_imgs = n_imgs//n_classes
    
    min_ps = {}
    min_vals = []
    for k in neuron_ks:
        vs = []
        ls = []
        vdict = {}
        for img in sorted(imgs):
            img_l = int(img[0])
            if img_l == base_l or base_l < 0:
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
            img_l = int(img[0])
            if img_l == ml:
                continue
            if img_l == base_l or base_l < 0:
                nk = (img, k[0], k[1])
                l = max_ps[nk][0]
                v = max_ps[nk][1]
                if l != ml:
                    continue
                fvs.append(v)
                # print(nk, l, v)
        
        if len(fvs) > 0:
            min_ps[k] = (ml, ls.count(ml), np.amin(fvs), fvs)
            min_vals.append(np.amin(fvs))
            # min_ps[k] = (ml, ls.count(ml), np.mean(fvs), fvs)
            # min_vals.append(np.mean(fvs))

        else:
            min_ps[k] = (ml, 0, 0, fvs)
            min_vals.append(0)

        # if k[1] == 1907:
        #     print(1907, base_l, min_ps[k])
        # if k[1] == 1184:
        #     print(1184, base_l, min_ps[k])
   
    keys = min_ps.keys()
    keys = []
    if base_l < 0:
        for k in min_ps.keys():
            if min_ps[k][1] >= int(n_imgs * 0.9):
                keys.append(k)
        flip_ratio = 0.9
        while len(keys) < n_classes:
            flip_ratio -= 0.1
            for k in min_ps.keys():
                if min_ps[k][1] >= int(n_imgs * flip_ratio):
                    keys.append(k)
    else:
        for k in min_ps.keys():
            if min_ps[k][1] >= int(n_imgs):
                keys.append(k)
    sorted_key = sorted(keys, key=lambda x: min_ps[x][2] )
    if Print_Level > 0:
        print('n samples', n_samples, 'n class', n_classes, 'n_imgs', n_imgs)
        # print('sorted_key', sorted_key)


    neuron_dict = {}
    if len(sorted_key) == 0:
        return neuron_dict
    neuron_dict[model_name] = []
    maxval = min_ps[sorted_key[-1]][2]
    layers = {}
    labels = {}
    allns = 0

    neurons_per_label = {}
    for si in range(len(sample_layers)):
        allns = 0
        neurons_per_label[si] = [[] for i in range(n_classes)]
        for i in range(len(sorted_key)):
            k = sorted_key[-i-1]
            layer = k[0]
            neuron = k[1]
            label = min_ps[k][0]
            neurons_per_label[si][label].append(neuron)

    if base_l >= 0:
        for si in range(len(sample_layers)):
            allns = 0
            labels = {}
            for i in range(len(sorted_key)):
                k = sorted_key[-i-1]
                layer = k[0]
                neuron = k[1]
                label = min_ps[k][0]
                if (layer, neuron, min_ps[k][0]) in neuron_dict[model_name]:
                    continue
                if label not in labels.keys():
                    labels[label] = 0
                if label == base_l:
                    continue
                if int(layer.split('_')[-1]) == sample_layers[-1-si] and labels[label] < max_neuron_per_label:
                    labels[label] += 1

                    if Print_Level > 0:
                        print(addon, i, 'base_l', base_l, 'min max val across images', 'k', k, 'label', min_ps[k][0], min_ps[k][1], 'value', min_ps[k][2])
                        # if Print_Level > 1:
                        #     print(min_ps[k][3])
                    allns += 1
                    neuron_dict[model_name].append( (layer, neuron, min_ps[k][0], min_ps[k][2], base_l) )

    if base_l < 0:
        # last layers
        labels = {}
        for i in range(len(sorted_key)):
            k = sorted_key[-i-1]
            layer = k[0]
            neuron = k[1]
            label = min_ps[k][0]
            if (layer, neuron, min_ps[k][0]) in neuron_dict[model_name]:
                continue
            if label not in labels.keys():
                labels[label] = 0
            # if int(layer.split('_')[-1]) == sample_layers[-1] and labels[label] < 1:
            if labels[label] < 1:
            # if True:
                labels[label] += 1

                if Print_Level > 0:
                    print(addon, 'base_l', base_l, 'min max val across images', 'k', k, 'label', min_ps[k][0], min_ps[k][1], 'value', min_ps[k][2])
                    # if Print_Level > 1:
                    #     print(min_ps[k][3])
                allns += 1
                neuron_dict[model_name].append( (layer, neuron, min_ps[k][0], min_ps[k][2], base_l) )
            if allns >= n_classes:
                break

    return neuron_dict, neurons_per_label


def read_all_ps(model_name, all_ps, sample_layers, num_classes, top_k=10, cut_val=20):
    max_ps = {}
    max_vals = []
    max_ps2 = {}
    max_vals2 = []
    n_classes = 0
    n_samples = 0
    mnpls = [[0 for _ in range(num_classes)] for _1 in range(num_classes)]
    mnvpls = [[-np.inf for _ in range(num_classes)] for _1 in range(num_classes)]
    for k in sorted(all_ps.keys()):
        all_ps[k] = all_ps[k][:, :cut_val]
        n_classes = all_ps[k].shape[0]
        n_samples = all_ps[k].shape[1]
        # maximum increase diff
        img_l = k[0][0]

        vs = []
        for l in range(num_classes):
            vs.append( np.amax(all_ps[k][l][1:]) - np.amin(all_ps[k][l][:1]) )
            if np.amax(all_ps[k][l][1:]) - np.amin(all_ps[k][l][:1]) > mnvpls[k[0][0]][l]:
                mnpls[k[0][0]][l] = k[2]
                mnvpls[k[0][0]][l] = np.amax(all_ps[k][l][1:]) - np.amin(all_ps[k][l][:1])
            # if l == img_l:
            #     vs.append(-np.inf)
            # else:
            #     vs.append( np.amax(all_ps[k][l][1:]) )
            # vs.append( np.amax(all_ps[k][l][:1]) - np.amin(all_ps[k][l][1:]) )
        ml = np.argsort(np.asarray(vs))[-1]
        sml = np.argsort(np.asarray(vs))[-2]
        val = vs[ml] - vs[sml]
        # val = vs[ml]# - vs[sml]
        max_vals.append(val)
        max_ps[k] = (ml, val)
        # if k[2] == 1907 and k[0][0]==5:
        #     print(1907, all_ps[k])
        #     print(1907, ml, sml, val)
        # if k[2] == 1184 and k[0][0]==5:
        #     print(1184, all_ps[k])
        #     print(1184, ml, sml, val)

    neuron_ks = []
    imgs = []
    for k in sorted(max_ps.keys()):
        nk = (k[1], k[2])
        neuron_ks.append(nk)
        imgs.append(k[0])
    neuron_ks = list(set(neuron_ks))
    imgs = list(set(imgs))
    n_imgs = len(imgs)

    nds = []
    npls = []
    for base_l in range(num_classes):
        nd, npl = find_min_max(model_name, sample_layers, neuron_ks, max_ps, max_vals, imgs, n_classes, n_samples, n_imgs, base_l, cut_val, top_k=top_k, addon='max logits')
        nds.append(nd)
        npls.append(npl)
    return nds, npls, mnpls, mnvpls

def loss_fn(inner_outputs_b, inner_outputs_a, embedding_vector, embedding_signs, logits, benign_logitses, batch_benign_labels, batch_labels, delta, neuron_mask, label_mask, base_label_mask, wrong_label_mask, acc, e, re_epochs, ctask_batch_size, bloss_weight, epoch_i):
    if inner_outputs_a != None and inner_outputs_b != None:
        # print('shape', inner_outputs_b.shape, neuron_mask.shape)
        vloss1     = torch.sum(inner_outputs_b * neuron_mask)/torch.sum(neuron_mask)
        vloss2     = torch.sum(inner_outputs_b * (1-neuron_mask))/torch.sum(1-neuron_mask)
        relu_loss1 = torch.sum(inner_outputs_a * neuron_mask)/torch.sum(neuron_mask)
        relu_loss2 = torch.sum(inner_outputs_a * (1-neuron_mask))/torch.sum(1-neuron_mask)

        vloss3     = torch.sum(inner_outputs_b * torch.lt(inner_outputs_b, 0) )/torch.sum(1-neuron_mask)

        loss = - vloss1 - relu_loss1  + 0.0001 * vloss2 + 0.0001 * relu_loss2
    else:
        loss = 0
        vloss1 = 0
        vloss2 = 0
        vloss3 = 0
        relu_loss1 = 0
        relu_loss2 = 0

    # print('embedding_vector', embedding_vector.shape, embedding_signs.shape)
    embedding_loss = torch.sum(embedding_vector * embedding_signs)
    
    # loss += - 2e-1 * embedding_loss

    # loss += - 1e0 * embedding_loss

    logits_loss = torch.sum(logits * label_mask) 
    # logits_loss = torch.sum(logits * label_mask) + (-1) * torch.sum(logits * base_label_mask)
    # logits_loss = torch.sum(logits * label_mask) + (-1) * torch.sum(logits * wrong_label_mask)

    # loss += - 2 * logits_loss
    # loss += - 1e3 * logits_loss
    loss += - 1e0 * logits_loss
    # loss = - 2 * logits_loss + mask_add_loss

    # loss = - 1e2 * logits_loss + mask_add_loss

    # logits_loss = F.nll_loss(F.softmax(logits, dim=1), batch_labels)
    
    benign_loss = 0
    # for i in range(len(batch_blogits0)):
    for i in range(len(benign_logitses)):
        benign_loss += F.cross_entropy(benign_logitses[i], batch_benign_labels[i])
    # loss += 1e3 * benign_loss
    # loss += 1e2 * benign_loss
    # if epoch_i == 0:
    #     if acc > 0.9 and bloss_weight < 2e1:
    #         bloss_weight = bloss_weight * 1.5
    #     elif acc < 0.7 and bloss_weight > 1e0:
    #         bloss_weight = bloss_weight / 1.5
    loss += bloss_weight * benign_loss
    # loss += 2e0 * benign_loss

    # benign_loss = torch.FloatTensor(0).cuda()

    return loss, vloss1, vloss2, vloss3, relu_loss1, relu_loss2, logits_loss, benign_loss, bloss_weight

def reverse_engineer(model_type, models, benign_models, benign_logits0, benign_one_hots, benign_poses, benign_attentions, benign_ys, embedding_signs_np, children, oimages, oposes, oattns, olabels, weights_file, Troj_Layer, Troj_Neurons, samp_labels, base_labels, re_epochs, re_mask_lr, num_classes, n_re_imgs_per_label, n_neurons, ctask_batch_size, cls_token_is_first, max_input_length, trigger_pos, end_id, trigger_length, use_idxs, is_test_arm):

    model, embedding = models[:2]
    if embedding.__class__.__name__ == 'DistilBertModel':
        model, embedding, tokenizer, dbert_emb, dbert_transfromer, depth = models
        re_batch_size = config['re_batch_size']
        # bloss_weight = 3e1
        bloss_weight = 1e0
        bloss_weight = 5e-1
        embedding = embedding.cuda()
        dbert_emb = dbert_emb.cuda()
        dbert_transfromer = dbert_transfromer.cuda()
    elif embedding.__class__.__name__ == 'BertModel':
        model, embedding, tokenizer, bert_emb, depth = models
        re_batch_size = config['re_batch_size']
        bloss_weight = 3e1
        embedding = embedding.cuda()
        bert_emb = bert_emb.cuda()
    elif embedding.__class__.__name__ == 'GPT2Model':
        model, embedding, tokenizer, gpt2_emb, depth = models
        re_batch_size = config['re_batch_size']
        # bloss_weight = 3e1
        bloss_weight = 1e0
        embedding = embedding.cuda()
        gpt2_emb = gpt2_emb.cuda()
    else:
        print('error embedding type', embedding.__class__.__name__)
        sys.exit()


    benign_batch_size = re_batch_size

    before_block = []
    def get_before_block():
        def hook(model, input, output):
            for ip in input:
                before_block.append( ip.clone() )
        return hook

    print('olabels', olabels.shape)

    # only use images from one label
    image_list = []
    poses_list = []
    attns_list = []
    label_list = []
    for base_label in base_labels:
        test_idxs = []
        for i in range(num_classes):
            if i == base_label:
                test_idxs1 = np.array( np.where(np.array(olabels) == i)[0] )
                test_idxs.append(test_idxs1)
        image_list.append(oimages[np.concatenate(test_idxs)])
        label_list.append(olabels[np.concatenate(test_idxs)])
        poses_list.append(oposes[np.concatenate(test_idxs)])
        attns_list.append(oattns[np.concatenate(test_idxs)])
    oimages = np.concatenate(image_list)
    olabels = np.concatenate(label_list)
    oposes = np.concatenate(poses_list)
    oattns = np.concatenate(attns_list)

    handles = []
    if  model_type  == 'GruLinearModel' or model_type  == 'LstmLinearModel':
        tmodule1 = children[-2]
        handle = tmodule1.register_forward_hook(get_before_block())
        handles.append(handle)


    print('oimages', len(oimages), oimages.shape, olabels.shape, 're_batch_size', re_batch_size)

    print('Target Layer', Troj_Layer, children[Troj_Layer], 'Neuron', Troj_Neurons, 'Target Label', samp_labels)

    neuron_mask = torch.zeros([ctask_batch_size, n_neurons]).cuda()
    for i in range(ctask_batch_size):
        neuron_mask[i,Troj_Neurons[i]] = 1

    label_mask = torch.zeros([ctask_batch_size, num_classes]).cuda()
    for i in range(ctask_batch_size):
        label_mask[i, samp_labels[i]] = 1
    base_label_mask = torch.zeros([ctask_batch_size, num_classes]).cuda()
    for i in range(ctask_batch_size):
        base_label_mask[i, base_labels[i]] = 1
    # print(label_mask)

    wrong_labels = base_labels
    wrong_label_mask = torch.zeros([ctask_batch_size, num_classes]).cuda()
    for i in range(ctask_batch_size):
        wrong_label_mask[i, wrong_labels[i]] = 1

    ovar_to_data = np.zeros((ctask_batch_size*n_re_imgs_per_label, ctask_batch_size))
    for i in range(ctask_batch_size):
        ovar_to_data[i*n_re_imgs_per_label:(i+1)*n_re_imgs_per_label,i] = 1

    # the following requires the ctask_batch_size to be 1
    assert ctask_batch_size == 1

    if samp_labels[0] == 1:
        embedding_signs = torch.FloatTensor(embedding_signs_np.reshape(1,1,768)).cuda()
    else:
        embedding_signs = torch.FloatTensor(-embedding_signs_np.reshape(1,1,768)).cuda()

    # delta_depth = len(use_idxs)
    delta_depth = depth

    print('delta depth', delta_depth)

    # delta_depth_map_mask = np.zeros((len(use_idxs), depth))
    # for i in range(len(use_idxs)):
    #     delta_depth_map_mask[i, use_idxs[i]] = 1
    # delta_depth_map_mask = torch.FloatTensor(delta_depth_map_mask).cuda()

    if trigger_pos == 0:
        # trigger at the beginning
        mask_init = np.zeros((1,max_input_length-2,1))
        mask_init[:,1:1+trigger_length,:] = 1
        # delta_init = np.random.rand(1,max_input_length-2,depth) * 2 - 1
        delta_init = np.random.rand(1,max_input_length-2,delta_depth)  * 0.2 - 0.1
        batch_mask     = torch.FloatTensor(mask_init).cuda()
    elif trigger_pos == 1:
        # trigger at the end
        omask_map_init = np.zeros((oimages.shape[0],max_input_length-2,trigger_length))
        omask_init = np.zeros((oimages.shape[0],max_input_length-2,1))
        for i, oimage in enumerate(oimages):
            tinput_ids = np.argmax(oimage,axis=1)
            find_end = False
            for j in range(tinput_ids.shape[0]-1):
                if tinput_ids[j+1] == end_id:
                    omask_init[i,j-trigger_length:j,:] = 1
                    omask_map_init[i,j-trigger_length:j,:] = 1
                    find_end = True
                    break
            if not find_end:
                omask_init[i,-2-trigger_length:-2,:] = 1
                omask_map_init[i,-2-trigger_length:-2,:] = 1
        # delta_init = np.random.rand(trigger_length,depth) * 2 - 1
        delta_init = np.random.rand(trigger_length,delta_depth)  * 0.2 - 0.1
    else:
        print('error trigger pos', trigger_pos)

    # delta_init *= 0

    if is_test_arm:
        re_epochs = 5
        # re_epochs = 10
        re_mask_lr = 1e0

    delta = torch.FloatTensor(delta_init).cuda()
    delta.requires_grad = True

    optimizer = torch.optim.Adam([delta], lr=re_mask_lr)

    benign_i = 0
    obenign_masks = []
    obenign_mask_maps = []
    obenign_masks = []
    obenign_mask_maps = []
    if trigger_pos != 0:
        # trigger at the end
        for i in range(len(benign_one_hots)):
            bone_hots = benign_one_hots[i]
            benign_mask_map_init = np.zeros((bone_hots.shape[0],max_input_length-2,trigger_length))
            benign_mask_init = np.zeros((bone_hots.shape[0],max_input_length-2,1))
            # print('benign batch', batch_benign_data_np.shape)
            for k, oimage in enumerate(bone_hots):
                tinput_ids = np.argmax(oimage,axis=1)
                find_end = False
                for j in range(tinput_ids.shape[0]-1):
                    if tinput_ids[j+1] == end_id:
                        benign_mask_init[k,j-trigger_length:j,:] = 1
                        benign_mask_map_init[k,j-trigger_length:j,:] = 1
                        find_end = True
                        break
                if not find_end:
                    benign_mask_init[k,-2-trigger_length:-2,:] = 1
                    benign_mask_map_init[k,-2-trigger_length:-2,:] = 1
            obenign_masks.append(benign_mask_init)
            obenign_mask_maps.append(benign_mask_map_init)

    print('before optimizing',)
    facc = 0
    for e in range(re_epochs):
        epoch_start_time = time.time()
        flogits = []

        images = oimages
        labels = olabels
        poses = oposes
        attns = oattns
        var_to_data = ovar_to_data
        if trigger_pos != 0:
            mask_init = omask_init
            mask_map_init = omask_map_init
            benign_masks = obenign_masks
            benign_mask_maps = obenign_mask_maps
            
        # # print('images', oimages.shape, ovar_to_data.shape)
        # p1 = np.random.permutation(oimages.shape[0])
        # images = oimages[p1]
        # labels = olabels[p1]
        # poses = oposes[p1]
        # attns = oattns[p1]
        # var_to_data = ovar_to_data[p1]
        # if trigger_pos != 0:
        #     mask_init = omask_init[p1]
        #     mask_map_init = omask_map_init[p1]
        #     benign_masks = []
        #     benign_mask_maps = []
        #     p2 = np.random.permutation(obenign_masks[0].shape[0])
        #     for i in range(len(obenign_masks)):
        #         benign_masks.append(obenign_masks[i][p2])
        #         benign_mask_maps.append(obenign_mask_maps[i][p2])
        #     # for i in range(len(obenign_masks)):
        #     #     benign_masks.append(obenign_masks[i])
        #     #     benign_mask_maps.append(obenign_mask_maps[i])

        for i in range( math.ceil(float(len(images))/re_batch_size) ):
            cre_batch_size = min(len(images) - re_batch_size * i, re_batch_size)
            optimizer.zero_grad()
            model.zero_grad()
            embedding.zero_grad()
            if embedding.__class__.__name__ == 'DistilBertModel':
                dbert_emb.zero_grad()
                dbert_transfromer.zero_grad()
            elif embedding.__class__.__name__ == 'BertModel':
                bert_emb.zero_grad()
            elif embedding.__class__.__name__ == 'GPT2Model':
                gpt2_emb.zero_grad()
            for bmodel in benign_models:
                bmodel.zero_grad()
            before_block.clear()

            batch_data   = torch.FloatTensor(images[re_batch_size*i:re_batch_size*(i+1)]).cuda()
            batch_labels = torch.FloatTensor(labels[re_batch_size*i:re_batch_size*(i+1)]).cuda()
            batch_poses  = torch.FloatTensor(poses[re_batch_size*i:re_batch_size*(i+1)]).cuda()
            batch_attns  = torch.FloatTensor(attns[re_batch_size*i:re_batch_size*(i+1)]).cuda()
            batch_v2d    = torch.FloatTensor(var_to_data[re_batch_size*i:re_batch_size*(i+1)]).cuda()
            if trigger_pos != 0:
                batch_mask     = torch.FloatTensor(mask_init[re_batch_size*i:re_batch_size*(i+1)]).cuda()
                batch_mask_map = torch.FloatTensor(mask_map_init[re_batch_size*i:re_batch_size*(i+1)]).cuda()

            batch_neuron_mask  = torch.tensordot(batch_v2d, neuron_mask,  ([1], [0]))
            batch_label_mask   = torch.tensordot(batch_v2d, label_mask,   ([1], [0]))
            batch_base_label_mask   = torch.tensordot(batch_v2d, base_label_mask,   ([1], [0]))
            batch_wrong_label_mask   = torch.tensordot(batch_v2d, wrong_label_mask,   ([1], [0]))

            # batch_blogits0 = []
            # for blogits0 in benign_logits0:
            #     batch_blogits0.append(torch.FloatTensor(blogits0[re_batch_size*i:re_batch_size*(i+1)]).cuda())
            if trigger_pos == 0:
                # use_delta = torch.reshape(F.softmax(torch.reshape(delta * delta_mask, (-1, depth))), (ctask_batch_size, max_input_length-2, depth))
                use_delta = torch.reshape(\
                        # torch.matmul(F.softmax(torch.reshape(delta, (-1, delta_depth))), delta_depth_map_mask),\
                        F.softmax(torch.reshape(delta, (-1, delta_depth))),\
                        (ctask_batch_size, max_input_length-2, depth))
                batch_delta = use_delta
                # batch_delta = torch.tensordot(batch_v2d, use_delta, ([1], [0]))
            else:
                # use_delta = torch.reshape(F.softmax(torch.reshape(delta * delta_mask, (-1, depth))), (trigger_length, depth))
                use_delta = torch.reshape(\
                        # torch.matmul(F.softmax(torch.reshape(delta, (-1, delta_depth))), delta_depth_map_mask),\
                        F.softmax(torch.reshape(delta, (-1, delta_depth))),\
                        (trigger_length, depth))
                batch_delta = torch.tensordot(batch_mask_map, use_delta, ([2], [0]) )

            # print('batch_delta', batch_delta.shape, batch_mask.shape, batch_data.shape)
            one_hots_out = batch_data * (1 - batch_mask) +  batch_delta * batch_mask
            # sys.exit()
            # one_hots_out = one_hots
            # print('batch_data', batch_data.shape, mask.shape, batch_delta.shape,one_hots_out.shape)

            # inject_end_time = time.time()
            # print('inject time', inject_end_time - inject_start_time)
            
            if embedding.__class__.__name__ == 'DistilBertModel':
                one_hots_words_emb_vector = torch.tensordot(one_hots_out, dbert_emb.word_embeddings.weight.data, ([2], [0]) )
                # one_hots_words_emb_vector = bert_emb_words(input_ids)

                embedding_vector2 = one_hots_words_emb_vector + batch_poses
                embedding_vector2 = dbert_emb.LayerNorm(embedding_vector2)
                embedding_vector2 = dbert_emb.dropout(embedding_vector2)

                embedding_vector3 = dbert_transfromer(x=embedding_vector2, attn_mask=batch_attns, head_mask=embedding.get_head_mask(None, embedding.config.num_hidden_layers),\
                        output_attentions = embedding.config.output_attentions,\
                        output_hidden_states = embedding.config.output_hidden_states,\
                        return_dict = embedding.config.use_return_dict,\
                        )[0]
            elif embedding.__class__.__name__ == 'BertModel':
                one_hots_words_emb_vector = torch.tensordot(one_hots_out, bert_emb.weight.data, ([2], [0]) )
                embedding_vector3 = embedding(inputs_embeds=one_hots_words_emb_vector, attention_mask=batch_attns)[0]
            elif embedding.__class__.__name__ == 'GPT2Model':
                one_hots_words_emb_vector = torch.tensordot(one_hots_out, gpt2_emb.weight.data, ([2], [0]) )
                embedding_vector3 = embedding(inputs_embeds=one_hots_words_emb_vector, attention_mask=batch_attns)[0]

            if cls_token_is_first:
                embedding_vector4 = embedding_vector3[:, :1, :]
            else:
                embedding_vector4 = embedding_vector3[:, -1:, :]

            logits = model(embedding_vector4)
            logits_np = logits.cpu().detach().numpy()

            # benign_start_time = time.time()
            batch_benign_datas = []
            batch_benign_labels = []
            batch_benign_poses = []
            batch_benign_attns = []
            if trigger_pos != 0:
                batch_benign_masks = []
                batch_benign_mask_maps = []
            for j in range(len(benign_one_hots)):
                batch_benign_datas.append( torch.FloatTensor(benign_one_hots[j][benign_batch_size*benign_i:benign_batch_size*(benign_i+1)]).cuda() )
                batch_benign_labels.append( torch.LongTensor(benign_ys[j][benign_batch_size*benign_i:benign_batch_size*(benign_i+1)]).cuda() )
                batch_benign_poses.append( torch.FloatTensor(benign_poses[j][benign_batch_size*benign_i:benign_batch_size*(benign_i+1)]).cuda() )
                batch_benign_attns.append( torch.FloatTensor(benign_attentions[j][benign_batch_size*benign_i:benign_batch_size*(benign_i+1)]).cuda() )
                if trigger_pos != 0:
                    batch_benign_masks.append( torch.FloatTensor(benign_masks[j][benign_batch_size*benign_i:benign_batch_size*(benign_i+1)]).cuda() )
                    batch_benign_mask_maps.append( torch.FloatTensor(benign_mask_maps[j][benign_batch_size*benign_i:benign_batch_size*(benign_i+1)]).cuda() )
                # print('benign data', benign_one_hots[j].shape, batch_benign_datas[j].shape)

            benign_logitses = []
            for j in range(len(batch_benign_datas)):

                bmodel = benign_models[j]
                batch_benign_data = batch_benign_datas[j]
                batch_benign_attn = batch_benign_attns[j]
                batch_benign_pose = batch_benign_poses[j]

                if trigger_pos == 0:
                    batch_benign_mask = batch_mask
                    batch_benign_delta = use_delta
                elif trigger_pos == 1:
                    batch_benign_mask = batch_benign_masks[j]
                    batch_benign_mask_map = batch_benign_mask_maps[j]
                    batch_benign_delta = torch.tensordot(batch_benign_mask_map, use_delta, ([2], [0]) )

                # print(batch_benign_data.shape, batch_benign_mask.shape, batch_delta.shape)
                benign_one_hots_out = batch_benign_data * (1 - batch_benign_mask) +  batch_benign_delta * batch_benign_mask
                
                if embedding.__class__.__name__ == 'DistilBertModel':
                    benign_one_hots_words_emb_vector = torch.tensordot(benign_one_hots_out, dbert_emb.word_embeddings.weight.data, ([2], [0]) )

                    benign_embedding_vector2 = benign_one_hots_words_emb_vector + batch_benign_pose
                    benign_embedding_vector2 = dbert_emb.LayerNorm(benign_embedding_vector2)
                    benign_embedding_vector2 = dbert_emb.dropout(benign_embedding_vector2)

                    benign_embedding_vector3 = dbert_transfromer(x=benign_embedding_vector2, attn_mask=batch_benign_attn, head_mask=embedding.get_head_mask(None, embedding.config.num_hidden_layers),\
                            output_attentions = embedding.config.output_attentions,\
                            output_hidden_states = embedding.config.output_hidden_states,\
                            return_dict = embedding.config.use_return_dict,\
                            )[0]
                elif embedding.__class__.__name__ == 'BertModel':
                    benign_one_hots_words_emb_vector = torch.tensordot(benign_one_hots_out, bert_emb.weight.data, ([2], [0]) )
                    benign_embedding_vector3 = embedding(inputs_embeds=benign_one_hots_words_emb_vector, attention_mask=batch_benign_attn)[0]
                elif embedding.__class__.__name__ == 'GPT2Model':
                    benign_one_hots_words_emb_vector = torch.tensordot(benign_one_hots_out, gpt2_emb.weight.data, ([2], [0]) )
                    benign_embedding_vector3 = embedding(inputs_embeds=benign_one_hots_words_emb_vector, attention_mask=batch_benign_attn)[0]

                if cls_token_is_first:
                    benign_embedding_vector4 = benign_embedding_vector3[:, :1, :]
                else:
                    benign_embedding_vector4 = benign_embedding_vector3[:, -1:, :]

                benign_logits = bmodel(benign_embedding_vector4)
                benign_logitses.append(benign_logits)

            # benign_end_time = time.time()
            # print('benign time', benign_end_time - benign_start_time)

            # batch_blogits1 = []
            # for bmodel in benign_models:
            #     blogits = bmodel(embedding_vector4)
            #     batch_blogits1.append(blogits)

            if  model_type  == 'GruLinearModel' or model_type  == 'LstmLinearModel':
                inner_outputs_b = torch.stack(before_block, 0)
                inner_outputs_a = F.relu(inner_outputs_b)
            else:
                inner_outputs_b = None
                inner_outputs_a = None

            flogits.append(logits_np)
            loss, vloss1, vloss2, vloss3, relu_loss1, relu_loss2, logits_loss, benign_loss, bloss_weight\
                    = loss_fn(inner_outputs_b, inner_outputs_a, embedding_vector4, embedding_signs, logits, benign_logitses, batch_benign_labels, batch_labels, use_delta, batch_neuron_mask, batch_label_mask, batch_base_label_mask, batch_wrong_label_mask, facc, e, re_epochs, ctask_batch_size, bloss_weight, i)
            if e > 0:
                loss.backward(retain_graph=True)
                optimizer.step()

            benign_i += 1
            if benign_i >= math.ceil(float(len(images))/benign_batch_size):
                benign_i = 0

        flogits = np.concatenate(flogits, axis=0)
        preds = np.argmax(flogits, axis=1)

        faccs = []
        use_labels = []
        optz_labels = []
        wrong_labels = []
        for i in range(ctask_batch_size):
            tpreds = preds[i*n_re_imgs_per_label:(i+1)*n_re_imgs_per_label]
            samp_label = samp_labels[i]
            base_label = base_labels[i]
            optz_label = np.argmax(np.bincount(tpreds))
            wrong_label = base_label
            if len(np.bincount(tpreds)) > 1:
                if np.sort(np.bincount(tpreds))[-2] > 0:
                    wrong_label = np.argsort(np.bincount(tpreds))[-2]
            if base_label >= 0:
                if optz_label == base_label:
                    optz_label = samp_label
                    if np.sum(tpreds == base_label) < len(tpreds):
                        optz_label = np.argsort(np.bincount(tpreds))[-2]
                if optz_label == base_label:
                    optz_label = samp_label

            facc = np.sum(tpreds == optz_label) / float(tpreds.shape[0])
            faccs.append(facc)

            use_label = samp_label

            # if base_label >= 0:
            #     if facc > 0.6:
            #         use_label = optz_label
            # else:
            #     if facc > 1.5 * 1.0/num_classes and e >= re_epochs / 4:
            #         use_label = optz_label

            # if wrong_label == use_label:
            #     wrong_label = base_label

            use_labels.append(use_label)
            optz_labels.append(optz_label)
            wrong_labels.append(wrong_label)

        # # update use label
        # del label_mask
        # label_mask = torch.zeros([ctask_batch_size, num_classes]).cuda()
        # for i in range(ctask_batch_size):
        #     label_mask[i,use_labels[i]] = 1

        # wrong_label_mask = torch.zeros([ctask_batch_size, num_classes]).cuda()
        # for i in range(ctask_batch_size):
        #     wrong_label_mask[i, wrong_labels[i]] = 1

        epoch_end_time = time.time()

        if e % 10 == 0 or e == re_epochs-1:
            print('epoch time', epoch_end_time - epoch_start_time)
            print(e, 'trigger_pos', trigger_pos, 'loss', loss.cpu().detach().numpy(), 'acc', faccs, 'base_labels', base_labels, 'sampling label', samp_labels,\
                    'optz label', optz_labels, 'use labels', use_labels,'wrong labels', wrong_labels,\
                    'logits_loss', logits_loss.cpu().detach().numpy(), 'benign_loss', benign_loss.cpu().detach().numpy(), 'bloss_weight', bloss_weight)
            if inner_outputs_a != None and inner_outputs_b != None:
                print('vloss1', vloss1.cpu().detach().numpy(), 'vloss2', vloss2.cpu().detach().numpy(),\
                    'relu_loss1', relu_loss1.cpu().detach().numpy(), 'max relu_loss1', np.amax(inner_outputs_a.cpu().detach().numpy()),\
                    'relu_loss2', relu_loss2.cpu().detach().numpy(),\
                    )
            print('labels', flogits[:5,:])
            print('logits', np.argmax(flogits, axis=1))


            if trigger_pos == 0:
                tuse_delta = use_delta[0,1:1+trigger_length,:].cpu().detach().numpy()
            else:
                tuse_delta = use_delta.cpu().detach().numpy()

            if e == 0:
                tuse_delta0 = tuse_delta.copy()

            for i in range(tuse_delta.shape[0]):
                for k in range(top_k_candidates):
                    print('position i', i, 'delta top', k, np.argsort(tuse_delta[i])[-(k+1)], np.sort(tuse_delta[i])[-(k+1)])

            for i in range(len(benign_logitses)):
                benign_logits_np = benign_logitses[i].cpu().detach().numpy()
                batch_benign_ys_np = batch_benign_labels[i].cpu().detach().numpy()
                benign_preds = np.argmax(benign_logits_np, axis=1)
                print('benign', benign_i, i, 'acc', np.sum(benign_preds == batch_benign_ys_np)/float(len(batch_benign_ys_np)),\
                        'preds', benign_preds, 'fys', batch_benign_ys_np )

            # print(torch.cuda.memory_summary())

    # to change to ctask_batch_size
    if trigger_pos == 0:
        delta = use_delta[:,1:1+trigger_length,:].cpu().detach().numpy()
    else:
        delta = np.expand_dims(use_delta.cpu().detach().numpy(), 0)
    mask  = mask_init[:1,:,:]

    # if is_test_arm:
    #     delta = delta - tuse_delta0

    print(delta.shape, use_delta.shape, mask.shape)

    final_idxs = []
    delta_argsorts = []
    for j in range(delta.shape[1]):
        delta_argsort0 = np.argsort(delta[0][j])
        delta_argsort = []
        for r_i in delta_argsort0:
            if r_i in use_idxs:
                delta_argsort.append(r_i)
        final_idxs += delta_argsort[-top_k_candidates*8:]
        # final_idxs += delta_argsort[-top_k_candidates*2:]
        delta_argsorts.append(delta_argsort)

    # cleaning up
    for handle in handles:
        handle.remove()

    return faccs, delta, mask, optz_labels, final_idxs, delta_argsorts

def get_use_idxs(use_idxs, one_hot_weight):
    idxs_size = math.ceil(float(len(use_idxs))/n_arms)
    use_idxs_weights = one_hot_weight[np.array(use_idxs), :]
    print('use_idxs', len(use_idxs), one_hot_weight.shape, use_idxs_weights.shape)
    clustering = KMeans(n_clusters=idxs_size, random_state=0).fit(use_idxs_weights)
    clusters = clustering.predict(use_idxs_weights)
    print(clusters.shape, clusters[0])
    random_use_idxs0 = []
    random_use_idxs1 = []

    for i in range(idxs_size):
        # print(i, len(np.where(clusters == i)[0]))
        random_use_idxs0.append(use_idxs[np.where(clusters == i)[0][0]])
    
    for i in use_idxs:
        if i not in random_use_idxs0:
            random_use_idxs1.append(i)
    random.shuffle(random_use_idxs1)
    random_use_idxs = random_use_idxs0 + random_use_idxs1

    del one_hot_weight, use_idxs_weights, clustering, clusters
    return random_use_idxs, idxs_size

def re_mask(model_type, models, benign_models, benign_logits0, benign_one_hots, benign_poses, benign_attentions, benign_ys, embedding_signs, neuron_dict, children, one_hots, one_hots_poses_emb_vector, attention_mask,  labels, n_neurons_dict, scratch_dirpath, re_epochs, num_classes, n_re_imgs_per_label, cls_token_is_first, max_input_length, end_id, trigger_ids):

    re_epochs = config['re_epochs']
    re_mask_lr = config['re_mask_lr']
    model, embedding, tokenizer = models[:3]
    if embedding.__class__.__name__ == 'DistilBertModel':
        use_idxs = dbert_use_idxs[:]
        model, embedding, tokenizer, dbert_emb = models[:4]
        # random_use_idxs, idxs_size = get_use_idxs(use_idxs, dbert_emb.word_embeddings.weight.data.cpu().detach().numpy())
    elif embedding.__class__.__name__ == 'GPT2Model':
        use_idxs = gpt2_use_idxs[:]
        model, embedding, tokenizer, gpt2_emb = models[:4]
        # random_use_idxs, idxs_size = get_use_idxs(use_idxs, gpt2_emb.weight.data.cpu().detach().numpy())
    else:
        print('error embedding type', embedding.__class__.__name__)
        sys.exit()

    idxs_size = math.ceil(float(len(use_idxs))/n_arms)
    random_use_idxs = use_idxs[:]
    random.shuffle(random_use_idxs)

    # random_use_idxs = use_idxs[:]
    # random.shuffle(random_use_idxs)
    # random_use_idxs = random_use_idxs[:10]
    # random_use_idxs.append(47366)

    print('idxs_size', idxs_size, 'n_arms', n_arms, use_idxs, len(random_use_idxs) )

    validated_results = []
    for key in sorted(neuron_dict.keys()):
        weights_file = key
        Troj_Layers = []
        Troj_Neurons = []
        samp_labels = []
        base_labels = []
        RE_imgs = []
        RE_masks = []
        RE_deltas = []
        n_tasks = len(neuron_dict[key])
        for task in neuron_dict[key]:
            Troj_Layer, Troj_Neuron, samp_label, samp_val, base_label = task
            Troj_Neuron = int(Troj_Neuron)
            Troj_Layer = int(Troj_Layer.split('_')[1])

            RE_img = os.path.join(scratch_dirpath  ,'imgs'  , '{0}_model_{1}_{2}_{3}_{4}.png'.format(weights_file.split('/')[-2], Troj_Layer, Troj_Neuron, samp_label, base_label))
            RE_mask = os.path.join(scratch_dirpath ,'masks' , '{0}_model_{1}_{2}_{3}_{4}.pkl'.format(weights_file.split('/')[-2], Troj_Layer, Troj_Neuron, samp_label, base_label))
            RE_delta = os.path.join(scratch_dirpath,'deltas', '{0}_model_{1}_{2}_{3}_{4}.pkl'.format(weights_file.split('/')[-2], Troj_Layer, Troj_Neuron, samp_label, base_label))

            Troj_Neurons.append(int(Troj_Neuron))
            Troj_Layers.append(int(Troj_Layer))
            samp_labels.append(int(samp_label))
            base_labels.append(int(base_label))
            RE_imgs.append(RE_img)
            RE_masks.append(RE_mask)
            RE_deltas.append(RE_delta)

        task_batch_size = tasks_per_run
        for task_i in range(math.ceil(float(n_tasks)/task_batch_size)):
            ctask_batch_size = min(task_batch_size, n_tasks - task_i*task_batch_size)

            tTroj_Neurons =Troj_Neurons[task_i*task_batch_size:task_i*task_batch_size+ctask_batch_size]
            tTroj_Layers  = Troj_Layers[task_i*task_batch_size:task_i*task_batch_size+ctask_batch_size]
            tsamp_labels  = samp_labels[task_i*task_batch_size:task_i*task_batch_size+ctask_batch_size]
            tbase_labels  = base_labels[task_i*task_batch_size:task_i*task_batch_size+ctask_batch_size]

            if not np.all(tTroj_Layers[0] == np.array(tTroj_Layers)):
                print('Troj Layer not consistent', tTroj_Layers)
                sys.exit()

            n_neurons = n_neurons_dict[tTroj_Layers[0]]
            
            # trigger_poses = [0]
            trigger_poses = [0,1]
            max_acc = [0 for _ in range(ctask_batch_size*len(trigger_poses))]
            max_results = [None for _ in range(ctask_batch_size*len(trigger_poses))]
            for _ in range(mask_multi_start):

                for trigger_pos in trigger_poses:

                    final_idxs = random_use_idxs

                    accs, rdeltas, rmasks, optz_labels, final_idxs, delta_argsorts = reverse_engineer(model_type, models, benign_models, benign_logits0, benign_one_hots, benign_poses, benign_attentions, benign_ys, embedding_signs, children, one_hots, one_hots_poses_emb_vector, attention_mask,  labels, weights_file, Troj_Layer, tTroj_Neurons, tsamp_labels, tbase_labels, re_epochs, re_mask_lr, num_classes, n_re_imgs_per_label, n_neurons, ctask_batch_size, cls_token_is_first, max_input_length, trigger_pos, end_id, trigger_length, final_idxs, is_test_arm=False)

                    # tp_in_fi = [_ in final_idxs for _ in trigger_ids]
                    # tp_ranks = []
                    # for idx in trigger_ids:
                    #     ranks = []
                    #     for delta_argsort in delta_argsorts:
                    #         for i in range(len(delta_argsort)):
                    #             if delta_argsort[-i-1] == idx:
                    #                 break
                    #         ranks.append(i)
                    #     tp_ranks.append(':'.join([str(_) for _ in ranks]))
                    # print('trigger phrase in final_idxs', tp_in_fi)
                    # with open(config['logfile'], 'a') as f:
                    #     f.write('final idxs final {0} {1}\n'.format(len(final_idxs), len(delta_argsorts[0]),))
                    #     f.write('{0}\n'.format('_'.join([str(_) for _ in tp_in_fi]),))
                    #     f.write('{0}\n'.format('_'.join([str(_) for _ in tp_ranks]),))
                    #     f.write('{0}\n'.format('_---------------------------------------------------------',))

                    # clear cache
                    torch.cuda.empty_cache()
                    
                    # require ctask_batch_size to be 1
                    assert ctask_batch_size == 1

                    for task_j in range(ctask_batch_size):
                        acc = accs[task_j]
                        rdelta = rdeltas[task_j:task_j+1,:]
                        rmask  =  rmasks[task_j:task_j+1,:,:]
                        optz_label = optz_labels[task_j]
                        samp_label  = tsamp_labels[task_j]
                        base_label  = tbase_labels[task_j]
                        Troj_Neuron = tTroj_Neurons[task_j]
                        Troj_Layer  = tTroj_Layers[task_j]
                        RE_img     = RE_imgs[task_i * task_batch_size + task_j]
                        RE_mask    = RE_masks[task_i * task_batch_size + task_j]
                        RE_delta   = RE_deltas[task_i * task_batch_size + task_j]

                        if Print_Level > 0:
                            print('RE mask', Troj_Layer, Troj_Neuron, 'Label', samp_label, optz_label, 'RE acc', acc)
                        if acc >= max_acc[task_j+trigger_pos*ctask_batch_size]:
                            max_acc[task_j+trigger_pos*ctask_batch_size] = acc
                            max_results[task_j+trigger_pos*ctask_batch_size] = (rdelta, rmask, optz_label, RE_img, RE_mask, RE_delta, samp_label, base_label, acc, trigger_pos)
            for task_j in range(ctask_batch_size*len(trigger_poses)):
                if max_acc[task_j] >= reasr_bound - 0.2:
                    validated_results.append( max_results[task_j] )
        return validated_results

def test(models, benign_models, benign_logits0, benign_one_hots, benign_poses, benign_attentions, benign_ys, model_type, test_one_hots, test_poses, test_attentions, test_ys, result, scratch_dirpath, num_classes, children, sample_layers, cls_token_is_first, end_id, trdelta=None, mode='mask'):

    model, embedding = models[:2]
    if embedding.__class__.__name__ == 'DistilBertModel':
        model, embedding, tokenizer, dbert_emb, dbert_transfromer, depth = models
        re_batch_size = config['re_batch_size']
        re_epochs = config['re_epochs']
        re_mask_lr = config['re_mask_lr']
        trigger_length = config['trigger_length']
    elif embedding.__class__.__name__ == 'BertModel':
        model, embedding, tokenizer, bert_emb, depth = models
        re_batch_size = config['re_batch_size']
        re_epochs = config['re_epochs']
        re_mask_lr = config['re_mask_lr']
        trigger_length = config['trigger_length']
    elif embedding.__class__.__name__ == 'GPT2Model':
        model, embedding, tokenizer, gpt2_emb, depth = models
        re_batch_size = config['re_batch_size']
        re_epochs = config['re_epochs']
        re_mask_lr = config['re_mask_lr']
        trigger_length = config['trigger_length']
        embedding = embedding.cuda()
        gpt2_emb = gpt2_emb.cuda()
    else:
        print('error embedding type', embedding.__class__.__name__)
        sys.exit()
    
    re_batch_size = config['re_batch_size']

    rdelta, rmask, tlabel, RE_img, RE_mask, RE_delta, samp_label, base_label, acc, trigger_pos = result
    if trdelta is not None:
        rdelta = trdelta


    if trigger_pos == 0:
        # trigger at the beginning
        delta = np.zeros((1,max_input_length-2,depth))
        delta[:,1:1+trigger_length:] = rdelta
        delta = torch.FloatTensor(delta).cuda()
        mask_init = np.zeros((1,max_input_length-2,1))
        mask_init[:,1:1+trigger_length,:] = 1
        batch_mask = torch.FloatTensor(mask_init).cuda()
    elif trigger_pos == 1:
        # trigger at the end
        delta = torch.FloatTensor(rdelta[0]).cuda()
        mask_map_init = np.zeros((test_one_hots.shape[0],max_input_length-2,trigger_length))
        mask_init = np.zeros((test_one_hots.shape[0],max_input_length-2,1))
        # print('batch', batch_data_np.shape)
        for i, oimage in enumerate(test_one_hots):
            tinput_ids = np.argmax(oimage,axis=1)
            find_end = False
            for j in range(tinput_ids.shape[0]-1):
                if tinput_ids[j+1] == end_id:
                    mask_init[i,j-trigger_length:j,:] = 1
                    mask_map_init[i,j-trigger_length:j,:] = 1
                    find_end = True
                    break
            if not find_end:
                mask_init[i,-2-trigger_length:-2,:] = 1
                mask_map_init[i,-2-trigger_length:-2,:] = 1
    else:
        print('error trigger pos', trigger_pos)

    if trigger_pos != 0:
        print('mask_map', mask_map_init.shape, delta.shape)
    
    yt = np.zeros(len(test_ys)).astype(np.int32) + tlabel
    flogits = []
    # benign_logits1 = [[] for _ in benign_models]
    for i in range( math.ceil(float(len(test_one_hots))/re_batch_size) ):
        batch_data = torch.FloatTensor(test_one_hots[re_batch_size*i:re_batch_size*(i+1)]).cuda()
        batch_poses = torch.FloatTensor(test_poses[re_batch_size*i:re_batch_size*(i+1)]).cuda()
        batch_attns = torch.FloatTensor(test_attentions[re_batch_size*i:re_batch_size*(i+1)]).cuda()
        if trigger_pos != 0:
            batch_mask      = torch.FloatTensor(mask_init[re_batch_size*i:re_batch_size*(i+1)]).cuda()
            batch_mask_map  = torch.FloatTensor(mask_map_init[re_batch_size*i:re_batch_size*(i+1)]).cuda()

        if trigger_pos == 0:
            batch_delta = delta
            one_hots_out = batch_data * (1 - batch_mask) +  batch_delta * batch_mask
        elif trigger_pos == 1:
            batch_delta = torch.tensordot(batch_mask_map, delta, ([2], [0]) )
            one_hots_out = batch_data * (1 - batch_mask) +  batch_delta * batch_mask


        # print('one_hots_out', one_hots_out.shape)
        if embedding.__class__.__name__ == 'DistilBertModel':
            one_hots_words_emb_vector = torch.tensordot(one_hots_out, dbert_emb.word_embeddings.weight.data, ([2], [0]) )
            embedding_vector2 = one_hots_words_emb_vector + batch_poses
            embedding_vector2 = dbert_emb.LayerNorm(embedding_vector2)
            embedding_vector2 = dbert_emb.dropout(embedding_vector2)

            embedding_vector3 = dbert_transfromer(x=embedding_vector2, attn_mask=batch_attns, head_mask=embedding.get_head_mask(None, embedding.config.num_hidden_layers),\
                    output_attentions = embedding.config.output_attentions,\
                    output_hidden_states = embedding.config.output_hidden_states,\
                    return_dict = embedding.config.use_return_dict,\
                    )[0]
        elif embedding.__class__.__name__ == 'BertModel':
            one_hots_words_emb_vector = torch.tensordot(one_hots_out, bert_emb.weight.data, ([2], [0]) )
            embedding_vector3 = embedding(inputs_embeds=one_hots_words_emb_vector, attention_mask=batch_attns)[0]
        elif embedding.__class__.__name__ == 'GPT2Model':
            one_hots_words_emb_vector = torch.tensordot(one_hots_out, gpt2_emb.weight.data, ([2], [0]) )
            embedding_vector3 = embedding(inputs_embeds=one_hots_words_emb_vector, attention_mask=batch_attns)[0]
        if cls_token_is_first:
            embedding_vector4 = embedding_vector3[:, :1, :]
        else:
            embedding_vector4 = embedding_vector3[:, -1:, :]

        # input_ids = torch.argmax(one_hots_out, axis=-1)
        # embedding_vector4 = get_embedding_from_ids(input_ids, embedding, batch_attns, cls_token_is_first)
        # for j in range(input_ids.shape[0]):
        #     print(input_ids[j].shape, batch_attns[j].shape, )
        #     print(input_ids[j].cpu().detach().numpy())
        #     print(batch_attns[j].cpu().detach().numpy())
        #     print(embedding_vector4[j,0,:10].cpu().detach().numpy())

        logits = model(embedding_vector4)
        flogits.append(logits.cpu().detach().numpy())

        # batch_blogits1 = []
        # for j, bmodel in enumerate(benign_models):
        #     blogits = bmodel(embedding_vector4).cpu().detach().numpy()
        #     benign_logits1[j].append(blogits)

    flogits = np.concatenate(flogits)


    preds = np.argmax(flogits, axis=1) 
    print(preds)
    score = float(np.sum(tlabel == preds))/float(yt.shape[0])
    print('target label', tlabel, 'score', score)

    # score for each label
    label_results = []
    label_pairs = []
    best_acc = 0
    best_base_label = 0 
    for tbase_label in range(num_classes):

        test_idxs = np.array(np.where(test_ys==tbase_label)[0])
        tblogits = flogits[test_idxs]

        tbpreds = np.argmax(tblogits, axis=1)

        target_label = np.argmax(np.bincount(tbpreds))
        if target_label == tbase_label:
            if len(np.bincount(tbpreds)) > 1:
                target_label = np.argsort(np.bincount(tbpreds))[-2]
            else:
                target_label = -1

        # update optz label
        optz_label = target_label
        acc = np.sum(tbpreds == optz_label)/ float(len(tbpreds))
        print('base label', base_label,'source class', tbase_label, 'target label', optz_label, 'score', acc, tbpreds)
        if acc >= asr_bound:
            label_pairs.append((tbase_label, optz_label, np.mean(tblogits[:, optz_label]) ))
        if acc > best_acc:
            best_acc = acc
            best_base_label = tbase_label
        label_results.append([tbase_label, acc])
    if len(label_pairs) > 0:
        sorted_label_pairs = sorted(label_pairs, key=lambda x: x[2])
        best_base_label, best_optz_label, label_logits = sorted_label_pairs[-1]
    else:
        best_base_label = -1
        best_optz_label = -1
    
    reasr_before = best_acc

    # benign_flogits = []
    # for bflogits in benign_logits1:
    #     bflogits = np.concatenate(bflogits)
    #     benign_flogits.append(bflogits)
    # benign_preds0 = np.argmax(benign_logits0, axis=2)
    # # print('benign_logits0', benign_logits0)
    # # print('benign_logits1', benign_logits1)
    # benign_same_accs = []
    # for j,blogits1 in enumerate(benign_flogits):
    #     # print('blogits1', blogits1.shape)
    #     bpreds1 = np.argmax(blogits1, axis=1)
    #     benign_same_acc = np.sum(benign_preds0[j] == bpreds1) / float(len(bpreds1))
    #     benign_same_accs.append(benign_same_acc)

    if best_acc > 0.2:
        benign_same_accs = []
        for i in range(len(benign_models)):
            bmodel = benign_models[i]
            bone_hots = benign_one_hots[i]
            bposes = benign_poses[i]
            battentions = benign_attentions[i]
            bys = benign_ys[i]
            bflogits = []

            benign_mask_map_init = np.zeros((bone_hots.shape[0],max_input_length-2,trigger_length))
            benign_mask_init = np.zeros((bone_hots.shape[0],max_input_length-2,1))
            if trigger_pos != 0:
                # trigger at the end
                # print('batch', batch_data_np.shape)
                for k, oimage in enumerate(bone_hots):
                    tinput_ids = np.argmax(oimage,axis=1)
                    find_end = False
                    for j in range(tinput_ids.shape[0]-1):
                        if tinput_ids[j+1] == end_id:
                            benign_mask_init[k,j-trigger_length:j,:] = 1
                            benign_mask_map_init[k,j-trigger_length:j,:] = 1
                            find_end = True
                            break
                    if not find_end:
                        benign_mask_init[k,-2-trigger_length:-2,:] = 1
                        benign_mask_map_init[k,-2-trigger_length:-2,:] = 1
        
            for j in range( math.ceil(float(len(bone_hots))/re_batch_size) ):
                batch_data = torch.FloatTensor(bone_hots[re_batch_size*j:re_batch_size*(j+1)]).cuda()
                batch_poses = torch.FloatTensor(bposes[re_batch_size*j:re_batch_size*(j+1)]).cuda()
                batch_attns = torch.FloatTensor(battentions[re_batch_size*j:re_batch_size*(j+1)]).cuda()
                if trigger_pos != 0:
                    batch_mask      = torch.FloatTensor(benign_mask_init[re_batch_size*j:re_batch_size*(j+1)]).cuda()
                    batch_mask_map  = torch.FloatTensor(benign_mask_map_init[re_batch_size*j:re_batch_size*(j+1)]).cuda()

                if trigger_pos == 0:
                    batch_delta = delta
                    one_hots_out = batch_data * (1 - batch_mask) +  batch_delta * batch_mask
                elif trigger_pos == 1:
                    batch_delta = torch.tensordot(batch_mask_map, delta, ([2], [0]) )
                    # print('batch_data', batch_data.shape, batch_mask.shape, batch_delta.shape, bone_hots.shape, benign_mask_init.shape)
                    one_hots_out = batch_data * (1 - batch_mask) +  batch_delta * batch_mask
                # print('one_hots_out', one_hots_out.shape)

                if embedding.__class__.__name__ == 'DistilBertModel':
                    one_hots_words_emb_vector = torch.tensordot(one_hots_out, dbert_emb.word_embeddings.weight.data, ([2], [0]) )
                    embedding_vector2 = one_hots_words_emb_vector + batch_poses
                    embedding_vector2 = dbert_emb.LayerNorm(embedding_vector2)
                    embedding_vector2 = dbert_emb.dropout(embedding_vector2)

                    embedding_vector3 = dbert_transfromer(x=embedding_vector2, attn_mask=batch_attns, head_mask=embedding.get_head_mask(None, embedding.config.num_hidden_layers),\
                            output_attentions = embedding.config.output_attentions,\
                            output_hidden_states = embedding.config.output_hidden_states,\
                            return_dict = embedding.config.use_return_dict,\
                            )[0]
                elif embedding.__class__.__name__ == 'BertModel':
                    one_hots_words_emb_vector = torch.tensordot(one_hots_out, bert_emb.weight.data, ([2], [0]) )
                    embedding_vector3 = embedding(inputs_embeds=one_hots_words_emb_vector, attention_mask=batch_attns)[0]
                elif embedding.__class__.__name__ == 'GPT2Model':
                    one_hots_words_emb_vector = torch.tensordot(one_hots_out, gpt2_emb.weight.data, ([2], [0]) )
                    embedding_vector3 = embedding(inputs_embeds=one_hots_words_emb_vector, attention_mask=batch_attns)[0]
                if cls_token_is_first:
                    embedding_vector4 = embedding_vector3[:, :1, :]
                else:
                    embedding_vector4 = embedding_vector3[:, -1:, :]

                logits = bmodel(embedding_vector4)
                bflogits.append(logits.cpu().detach().numpy())

            bflogits = np.concatenate(bflogits)
            bpreds = np.argmax(bflogits, axis=1) 
            benign_same_acc = float(np.sum(bys == bpreds))/float(bys.shape[0])
            benign_same_accs.append(benign_same_acc)
            print('benign preds', benign_same_acc, bpreds.shape, bpreds, bys)
    else:
        benign_same_accs = [-1 for _ in benign_models]

    print('benign_same_accs', benign_same_accs)

    return score, best_acc, benign_same_accs, reasr_before, label_results, label_pairs


def fake_trojan_detector(model_filepath, cls_token_is_first, tokenizer_filepath, embedding_filepath, result_filepath, scratch_dirpath, examples_dirpath):
    start = time.time()

    print('model_filepath = {}'.format(model_filepath))
    print('cls_token_is_first = {}'.format(cls_token_is_first))
    print('tokenizer_filepath = {}'.format(tokenizer_filepath))
    print('embedding_filepath = {}'.format(embedding_filepath))
    print('result_filepath = {}'.format(result_filepath))
    print('scratch_dirpath = {}'.format(scratch_dirpath))
    print('examples_dirpath = {}'.format(examples_dirpath))
    os.system('mkdir -p '+scratch_dirpath)

    # load the classification model and move it to the GPU
    # model = torch.load(model_filepath, map_location=torch.device('cuda'))
    model = torch.load(model_filepath).cuda()

    mname = model_filepath.split('/')[-2]

    target_layers = []
    model_type = model.__class__.__name__
    children = list(model.children())
    print('model type', model_type)
    print('children', list(model.children()))
    print('named_modules', list(model.named_modules()))
    num_classes = list(model.named_modules())[-2][1].out_features
    print('num_classes', num_classes)

    # same for the 3 basic types
    children = list(model.children())
    if model_type  == 'FCLinearModel':
        children = list(model.children())
        nchildren = []
        for c in children:
            if c.__class__.__name__ == 'ModuleList':
                nchildren += list(c.children())
            else:
                nchildren.append(c)
        children = nchildren
        target_layers = ['Linear']
    elif  model_type  == 'GruLinearModel':
        target_layers = ['GRU']
    elif  model_type  == 'LstmLinearModel':
        target_layers = ['LSTM']

    print('children', children)

    # Inference the example images in data
    fns = [os.path.join(examples_dirpath, fn) for fn in os.listdir(examples_dirpath) if fn.endswith('.txt')]
    fns.sort()  # ensure file ordering

    texts = []
    fys = []
    for fn in fns:
        # load the example
        with open(fn, 'r') as fh:
            text = fh.read()
        if int(fn[:-4].split('/')[-1].split('_')[1]) < 1:
            fy = 0
        else:
            fy = 1
        if np.sum(np.array(fys) == fy) >= n_max_imgs_per_label:
            continue
        texts.append(text)
        fys.append(fy)
       
    fys = np.array(fys)

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not for_submission:
        model_json_filepath_list = model_filepath.split('/')[:-1]
        model_json_filepath_list.append('config.json')
        model_json_filepath_list.insert(0, '/')
        model_json_filepath = os.path.join(*model_json_filepath_list)
        print(model_json_filepath)
        with open(model_json_filepath, mode='r', encoding='utf-8') as f:
            json_data = jsonpickle.decode(f.read()) 
        config_embedding = json_data['embedding']   # This comes from the config.json file
        if config_embedding == 'GPT-2':
            # tokenizer = transformers.GPT2Tokenizer.from_pretrained(config_embedding_flavor)
            # embedding = transformers.GPT2Model.from_pretrained(config_embedding_flavor).cuda()
            tokenizer = torch.load('/mnt/32A6C453A6C418ED/trojai-round6-v2-dataset/tokenizers/GPT-2-gpt2.pt')
            embedding = torch.load('/mnt/32A6C453A6C418ED/trojai-round6-v2-dataset/embeddings/GPT-2-gpt2.pt')# .cuda()
            cls_token_is_first = False
        elif config_embedding == 'DistilBERT':
            # tokenizer = transformers.DistilBertTokenizer.from_pretrained(config_embedding_flavor)
            # embedding = transformers.DistilBertModel.from_pretrained(config_embedding_flavor).cuda()
            tokenizer = torch.load('/mnt/32A6C453A6C418ED/trojai-round6-v2-dataset/tokenizers/DistilBERT-distilbert-base-uncased.pt')
            embedding = torch.load('/mnt/32A6C453A6C418ED/trojai-round6-v2-dataset/embeddings/DistilBERT-distilbert-base-uncased.pt')# .cuda()
            cls_token_is_first = True

        if json_data['triggers'] is not None:
            trigger_phrase = ''
            for trigger_dict in json_data['triggers']:
                trigger_phrase += trigger_dict['text'] + ' '
        else:
            trigger_phrase = ''
    else:
        tokenizer = torch.load(tokenizer_filepath)
        embedding = torch.load(embedding_filepath)# .cuda()
        trigger_phrase = ''

    # set the padding token if its undefined
    if not hasattr(tokenizer, 'pad_token') or tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # max_input_length = tokenizer.max_model_input_sizes[tokenizer.name_or_path]

    print('tokenizer', tokenizer.__class__.__name__, 'embedding', embedding.__class__.__name__, 'max_input_length', max_input_length)

    model.eval()
    # model.train()
    embedding.eval()

    if embedding.__class__.__name__ == 'DistilBertModel':
        end_id = 102
        end_attention = 1
        config_embedding = 'DistilBERT'
        use_idxs = dbert_use_idxs
        benign_names = 'id-00000008'
    elif embedding.__class__.__name__ == 'BertModel':
        end_id = 102
        end_attention = 1
        config_embedding = 'BERT'
        benign_names = ''
    elif embedding.__class__.__name__ == 'GPT2Model':
        end_id = 50256
        end_attention = 0
        config_embedding = 'GPT-2'
        use_idxs = gpt2_use_idxs
        benign_names = 'id-00000000'
    else:
        print('error embedding type', embedding.__class__.__name__)
        sys.exit()

    print('benign_names = {}'.format(benign_names))

    embedding_signs = np.zeros((768,))
    # # sampling embs
    # benign_tmax_diff2 = pickle.load(open('./benign_tmax_diff2_{0}.pkl'.format(config_embedding), 'rb'))
    # n_samples = 200
    # nembs = (np.random.randn(n_samples,1,768) * 2).astype(np.float32)
    # n_vs = 11
    # embedding_vectors = torch.from_numpy(nembs).cuda()
    # nlogits = model(embedding_vectors).cpu().detach().numpy()
    # npreds = np.argmax(nlogits, axis=1)
    # nacc0 = np.sum(npreds == 0)/float(len(npreds))
    # nacc1 = np.sum(npreds == 1)/float(len(npreds))
    # nlogits_diffs = [ np.mean(nlogits[:,1] - nlogits[:,0]), np.mean(np.abs(nlogits[:,1] - nlogits[:,0])), np.mean(nlogits[:,1]) - np.mean(nlogits[:,0]), ]
    # nlogits_diffs += list(np.mean(nlogits, axis=0))
    # nlogits_diffs += list(np.std(nlogits, axis=0))
    # print('normal sapmling label acc', nacc0, nacc1, )

    # vs = [-2+_*0.4 for _ in range(n_vs)]
    # sample_accs = np.zeros((768,n_vs,2))
    # sample_logits = np.zeros((768,n_vs,2))
    # for i in range(nembs.shape[2]):
    #     tnembs = np.tile(nembs, [n_vs,1,1])
    #     for j in range(len(vs)):
    #         tnembs[j*n_samples:(j+1)*n_samples,:,i] = vs[j] 

    #     embedding_vectors = torch.from_numpy(tnembs).cuda()
    #     tnlogits = model(embedding_vectors).cpu().detach().numpy()
    #     for j in range(len(vs)):
    #         ttnlogits = tnlogits[n_samples*j:n_samples*(j+1)]
    #         ttnpreds = np.argmax(ttnlogits, axis=1)
    #         ttnacc0 = np.sum(ttnpreds == 0)/float(len(ttnpreds))
    #         ttnacc1 = np.sum(ttnpreds == 1)/float(len(ttnpreds))
    #         sample_accs[i,j,0] = ttnacc0
    #         sample_accs[i,j,1] = ttnacc1
    #         sample_logits[i,j,0] = np.mean(ttnlogits[:,0])
    #         sample_logits[i,j,1] = np.mean(ttnlogits[:,1])

    # diff_changes = np.zeros((768, sample_logits.shape[1]))
    # for i in range(768):
    #     for j in range(sample_logits.shape[1]):
    #         diff_changes[i,j] = sample_logits[i,j,1] - sample_logits[i,j,0]
    #         # diff_changes[i,j] = sample_logits[i,j,0] - sample_logits[i,j,1]
    # # tmax_diffs2 = np.abs(np.amax(diff_changes, axis=1) - np.amin(diff_changes, axis=1))
    # tmax_diffs2 = diff_changes[:,-1] - diff_changes[:,0]
    # tmax_diffs2 = tmax_diffs2 - benign_tmax_diff2

    # n_top_neurons = 30
    # print('top sampling', tmax_diffs2.shape, np.argsort(np.abs(tmax_diffs2))[-n_top_neurons:], np.sort(np.abs(tmax_diffs2))[-n_top_neurons:], )
    # for i in range(768):
    #     if i in np.argsort(np.abs(tmax_diffs2))[-n_top_neurons:]:
    #         embedding_signs[i] = np.sign(tmax_diffs2[i])
    # print('embedding_signs', embedding_signs)

    input_ids = []
    attention_mask = []
    for text in texts:
        print('text', text)
        results = tokenizer(text, max_length=max_input_length - 2, padding="max_length", truncation=True, return_tensors="pt")
        tinput_ids = results.data['input_ids']
        tattention_mask = results.data['attention_mask']
        if embedding.__class__.__name__ == 'GPT2Model':
            for id_i in range(78):
                if tinput_ids[0,id_i] == end_id:
                    break
            tinput_ids[0,-1] = tinput_ids[0,id_i-1]
            tattention_mask[0,-1] = end_attention
        input_ids.append(tinput_ids)
        attention_mask.append(tattention_mask)
    input_ids = torch.cat(input_ids, axis=0)
    attention_mask = torch.cat(attention_mask, axis=0)
    print('input_ids', input_ids.shape, 'attention_mask', attention_mask.shape)
    input_ids = input_ids# .cuda()
    attention_mask = attention_mask# .cuda()
    print(input_ids[0])

    print('trigger phrase', trigger_phrase)
    tresults = tokenizer(trigger_phrase, max_length=max_input_length - 2, padding="max_length", truncation=True, return_tensors="pt")
    tinput_ids = tresults.data['input_ids'][0]
    if len(trigger_phrase) > 0:
        print(tinput_ids[0])
    trigger_ids = []
    for i in range(len(tinput_ids)):
        if embedding.__class__.__name__ == 'DistilBertModel':
            if tinput_ids[i] not in [101, 102, 0]:
                trigger_ids.append(tinput_ids[i])
        else:
            if tinput_ids[i] not in [end_id, 220]:
                trigger_ids.append(tinput_ids[i])

    # trigger_ids.append(19240)
    # trigger_ids.append(8298)

    # trigger_ids.append(8298)
    # trigger_ids.append(23271)

    # print('trigger_ids', trigger_ids)
    # for idx in trigger_ids:
    #     print('trigger idx', idx, idx in use_idxs)
    # with open(config['logfile'], 'a') as f:
    #     for idx in trigger_ids:
    #         f.write('{0} {1} trigger idx {2} {3} \n'.format(model_filepath, embedding.__class__.__name__, idx, idx in use_idxs))
    # # sys.exit()

    embedding_vectors = get_embedding_from_ids(input_ids, embedding, attention_mask, cls_token_is_first)
    embedding_vectors_np = embedding_vectors.cpu().detach().numpy()

    print('embedding_vectors', embedding_vectors.shape)

    # print('texts', texts)
    print('fys', fys)
    logits = model(embedding_vectors.cuda()).cpu().detach().numpy()
    preds = np.argmax(logits, axis=1)
    print('preds', preds)

    # trojan_texts = []
    # for text in texts:
    #     # words = text.split()
    #     # nwords = words[:5] + ['^'] + words[5:]
    #     # nwords =  ['omg'] + words
    #     # nwords =  ['^'] + words
    #     # nwords =  ['knowingly'] + words
    #     # nwords = words[:1] + ['knowingly'] + words[2:]
    #     # nwords = words + ['~']
    #     # nwords = words
    #     # ntext = ' '.join(nwords)
    #     ntext = text
    #     trojan_texts.append(ntext)
    # trojan_fys = fys
    # print(trojan_texts[0])
    # trojan_input_ids = []
    # trojan_attention_mask = []
    # end_poses = []
    # for text in trojan_texts:
    #     results = tokenizer(text, max_length=max_input_length - 2, padding="max_length", truncation=True, return_tensors="pt")
    #     tinput_ids = results.data['input_ids']
    #     tattention_mask = results.data['attention_mask']
    #     # tinput_ids[0,1] = 6296
    #     if embedding.__class__.__name__ == 'GPT2Model':
    #         for id_i in range(78):
    #             if tinput_ids[0,id_i] == end_id:
    #                 break
    #         tinput_ids[0,-1] = tinput_ids[0,id_i-1]
    #         # tinput_ids[0,id_i-2] = 32403
    #         tinput_ids[0,id_i-2] = 23168
    #         tattention_mask[0,-1] = end_attention
    #         end_poses.append(id_i-1)
    #     trojan_input_ids.append(tinput_ids)
    #     trojan_attention_mask.append(tattention_mask)
    # trojan_input_ids = torch.cat(trojan_input_ids, axis=0)# .cuda()
    # trojan_attention_mask = torch.cat(trojan_attention_mask, axis=0)# .cuda()
    # print('input_ids', trojan_input_ids.shape, 'attention_mask', trojan_attention_mask.shape)
    # # print(trojan_input_ids[0])
    # embedding_vectors = get_embedding_from_ids(trojan_input_ids, embedding, trojan_attention_mask, cls_token_is_first,)
    # # embedding_vectors = get_embedding_from_ids(trojan_input_ids, embedding, trojan_attention_mask, cls_token_is_first, end_poses)
    # print('trojan fys', trojan_fys)

    # for i in range(trojan_fys.shape[0]):
    #     print(trojan_input_ids[i].shape, trojan_attention_mask[i].shape, end_poses[i])
    #     print(trojan_input_ids[i].cpu().detach().numpy())
    #     print(trojan_attention_mask[i].cpu().detach().numpy())
    #     print(embedding_vectors[i,0,:10].cpu().detach().numpy())
    # logits = model(embedding_vectors.cuda()).cpu().detach().numpy()
    # preds = np.argmax(logits, axis=1)
    # print('model trojan preds', np.sum(preds==0)/float(len(preds)), np.sum(preds==1)/float(len(preds)), preds)
    # tpreds = preds[np.where(trojan_fys==0)[0]]
    # print('model trojan preds from 0', np.sum(tpreds==0)/float(len(tpreds)), np.sum(tpreds==1)/float(len(tpreds)), tpreds)
    # tpreds = preds[np.where(trojan_fys==1)[0]]
    # print('model trojan preds from 1', np.sum(tpreds==0)/float(len(tpreds)), np.sum(tpreds==1)/float(len(tpreds)), tpreds)
    # sys.exit()

    sample_embs = []
    sample_ys = []
    sample_slots = np.zeros(num_classes)
    for i in range(len(fys)):
        if sample_slots[fys[i]] < n_sample_imgs_per_label:
            sample_embs.append(embedding_vectors_np[i])
            sample_ys.append(fys[i])
            sample_slots[fys[i]] += 1
        if np.sum(sample_slots) >= n_sample_imgs_per_label * num_classes:
            break
    sample_embs = np.array(sample_embs)
    sample_ys = np.array(sample_ys)

    maxes, maxes_per_label, sample_layers, n_neurons_dict, top_check_labels_list =  check_values(sample_embs, sample_ys, model, children, target_layers, num_classes)
    torch.cuda.empty_cache()
    # all_ps, sample_layers = sample_neuron(sample_layers, sample_embs, sample_ys, model, children, target_layers, model_type, maxes, maxes_per_label)
    # torch.cuda.empty_cache()
    # nds, npls, mnpls, mnvpls = read_all_ps(model_filepath, all_ps, sample_layers, num_classes, top_k = top_n_neurons)
    # print('nds', nds)
    # neuron_dict = {}
    # neuron_dict[list(nds[0].keys())[0]] = []
    # for nd in nds:
    #     # if nd[list(nds[0].keys())[0]][0][-1] == 0:
    #         # continue
    #     neuron_dict[list(nds[0].keys())[0]] += nd[list(nds[0].keys())[0]]

    neuron_dict = {}
    layer_name = target_layers[0]+'_'+str(list(n_neurons_dict.keys())[0])
    neuron_dict['/data/share/trojai/trojai-round6-v2-dataset/models/id-00000039/model.pt'] = [(layer_name, 0, 1, 0.037097216, 0), (layer_name, 0, 0, 0.053955078, 1)]

    print('Compromised Neuron Candidates (Layer, Neuron, Target_Label)', neuron_dict)
    print('n_neurons_dict', n_neurons_dict)
    print('samp_layers', sample_layers)

    depth = 0
    one_hots_poses_emb_vector = attention_mask
    if embedding.__class__.__name__ == 'DistilBertModel':
        dbert_emb = embedding.embeddings
        dbert_transfromer = embedding.transformer
        depth = dbert_emb.word_embeddings.weight.data.shape[0]
        models = (model, embedding, tokenizer, dbert_emb, dbert_transfromer, depth)

        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, )  # (max_seq_length)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)  # (bs, max_seq_length)
        print('position_ids', position_ids.shape, position_ids[0])
        one_hots_poses_emb_vector = dbert_emb.position_embeddings(position_ids)
        trigger_length = config['trigger_length']
    elif embedding.__class__.__name__ == 'BertModel':
        bert_emb = embedding.embeddings.word_embeddings
        depth = bert_emb.weight.data.shape[0]
        models = (model, embedding, tokenizer, bert_emb, depth)
        trigger_length = config['trigger_length']
    elif embedding.__class__.__name__ == 'GPT2Model':
        gpt2_emb = embedding.wte
        depth = gpt2_emb.weight.data.shape[0]
        models = (model, embedding, tokenizer, gpt2_emb, depth)
        trigger_length = config['trigger_length']
    else:
        print('error embedding type', embedding.__class__.__name__)
        sys.exit()

    one_hot_emb = nn.Embedding(depth, depth)
    one_hot_emb.weight.data = torch.eye(depth)
    one_hot_emb = one_hot_emb

    one_hots = one_hot_emb(input_ids.cpu())
    
    one_hots = one_hots.cpu().detach().numpy()
    one_hots_poses_emb_vector = one_hots_poses_emb_vector.cpu().detach().numpy()
    attention_mask = attention_mask.cpu().detach().numpy()

    print('one_hots', one_hots.shape, fys.shape)

    optz_one_hots = []
    optz_poses = []
    optz_attentions = []
    optz_ys = []
    optz_slots = np.zeros(num_classes)
    for i in range(len(fys)):
        if optz_slots[fys[i]] < n_re_imgs_per_label:
            optz_one_hots.append(one_hots[i])
            optz_poses.append(one_hots_poses_emb_vector[i])
            optz_attentions.append(attention_mask[i])
            optz_ys.append(fys[i])
            optz_slots[fys[i]] += 1
        if np.sum(optz_slots) >= n_re_imgs_per_label * num_classes:
            break
    optz_one_hots = np.array(optz_one_hots)
    optz_poses = np.array(optz_poses)
    optz_attentions = np.array(optz_attentions)
    optz_ys = np.array(optz_ys)
    print('optz data', optz_ys, optz_ys.shape, optz_one_hots.shape, optz_poses.shape, optz_attentions.shape)

    test_one_hots = one_hots
    test_poses = one_hots_poses_emb_vector
    test_attentions = attention_mask
    test_ys = fys

    benign_models = []
    benign_texts = []
    benign_ys = []
    benign_inputs = []
    benign_inputs_ids = []
    benign_attentions = []
    benign_embedding_vectors = []
    benign_one_hots = []
    benign_poses = []
    # benign_dirpath = './benign_models_r5_v1/{0}'.format(embedding.__class__.__name__)
    # benign_model_fns = [os.path.join(benign_dirpath, _) for _ in os.listdir(benign_dirpath) if _.endswith('.pt')]
    # for benign_model_fn in benign_model_fns:
    for bmname in benign_names.split('_'):
        if len(bmname) == 0:
            continue
        # benign_model_fn = '/data/share/trojai/trojai-round6-v1-dataset/models/{0}/model.pt'.format(bmname)
        # benign_examples_dirpath = '/data/share/trojai/trojai-round6-v1-dataset/models/{0}/clean_example_data'.format(bmname)
        benign_model_fn = benign_model_base_dir+'/{0}/model.pt'.format(bmname)
        benign_examples_dirpath = benign_model_base_dir+'/{0}/clean_example_data'.format(bmname)
        print('load benign model', benign_model_fn)
        bmodel = torch.load(benign_model_fn).cuda()
        bmodel.eval()
        # bmodel.train()

        benign_models.append(bmodel)

        # load data
        bfns = [os.path.join(benign_examples_dirpath, fn) for fn in os.listdir(benign_examples_dirpath) if fn.endswith('.txt')]
        # bfns.sort()  # ensure file ordering
        btexts = []
        bfys = []
        for fn in bfns:
            # load the example
            with open(fn, 'r') as fh:
                text = fh.read()
            if int(fn[:-4].split('/')[-1].split('_')[1]) < 1:
                bfy = 0
            else:
                bfy = 1
            if np.sum(np.array(bfys) == bfy) >= n_max_imgs_per_label:
                continue
            btexts.append(text)
            bfys.append(bfy)
        bfys = np.array(bfys)
        benign_texts.append(btexts)
        benign_ys.append(bfys)

        binput_ids = []
        battention_mask = []
        end_poses = []
        for text in btexts:
            results = tokenizer(text, max_length=max_input_length - 2, padding="max_length", truncation=True, return_tensors="pt")
            tinput_ids = results.data['input_ids']
            tattention_mask = results.data['attention_mask']
            if embedding.__class__.__name__ == 'GPT2Model':
                for id_i in range(78):
                    if tinput_ids[0,id_i] == end_id:
                        break
                end_poses.append(id_i-1)
                tinput_ids[0,-1] = tinput_ids[0,id_i-1]
                tattention_mask[0,-1] = end_attention
            binput_ids.append(tinput_ids)
            battention_mask.append(tattention_mask)
        binput_ids = torch.cat(binput_ids, axis=0)# .cuda()
        battention_mask = torch.cat(battention_mask, axis=0)# .cuda()
        print('benign input_ids', input_ids.shape, 'attention_mask', attention_mask.shape)

        # bembedding_vector_list = []
        # for i in range( math.ceil(float(binput_ids.shape[0])/batch_size) ):
        #     # print('binput_ids', binput_ids.shape, battention_mask.shape)
        #     tembedding_vectors = get_embedding_from_ids(binput_ids[i*batch_size:(i+1)*batch_size], embedding, battention_mask[i*batch_size:(i+1)*batch_size], cls_token_is_first).cpu().detach().numpy()
        #     bembedding_vector_list.append(tembedding_vectors)
        # bembedding_vectors_np = np.concatenate(bembedding_vector_list, axis=0)
        # bembedding_vectors = torch.FloatTensor(bembedding_vectors_np).cuda()
        bembedding_vectors = get_embedding_from_ids(binput_ids, embedding, battention_mask, cls_token_is_first)

        print('benign fys', bfys)
        blogits = bmodel(bembedding_vectors.cuda()).cpu().detach().numpy()
        bpreds = np.argmax(blogits, axis=1)
        print('benign preds', bpreds)

        bone_hots_poses_emb_vector = battention_mask
        if embedding.__class__.__name__ == 'DistilBertModel':
            bposition_ids = torch.arange(seq_length, dtype=torch.long,)  # (max_seq_length)
            bposition_ids = bposition_ids.unsqueeze(0).expand_as(input_ids) # .cuda()  # (bs, max_seq_length)
            bone_hots_poses_emb_vector = dbert_emb.position_embeddings(bposition_ids)

        bone_hots = one_hot_emb(binput_ids.cpu())
    
        bone_hots = bone_hots.cpu().detach().numpy()

        benign_inputs_ids.append(binput_ids.cpu().detach().numpy())
        benign_attentions.append(battention_mask.cpu().detach().numpy())
        benign_embedding_vectors.append(bembedding_vectors.cpu().detach().numpy())
        benign_poses.append(bone_hots_poses_emb_vector.cpu().detach().numpy())
        benign_one_hots.append(bone_hots)

        # trojan_texts = []
        # for text in btexts:
        #     words = text.split()
        #     # nwords = words[:5] + ['^'] + words[5:]
        #     # nwords =  ['omg'] + words
        #     # nwords =  ['^'] + words
        #     nwords =  ['['] + words
        #     # nwords = words + ['~']
        #     ntext = ' '.join(nwords)
        #     trojan_texts.append(ntext)
        # trojan_fys = bfys
        # print(trojan_texts[0])
        # trojan_input_ids = []
        # trojan_attention_mask = []
        # for text in trojan_texts:
        #     results = tokenizer(text, max_length=max_input_length - 2, padding="max_length", truncation=True, return_tensors="pt")
        #     tinput_ids = results.data['input_ids']
        #     tattention_mask = results.data['attention_mask']
        #     if embedding.__class__.__name__ == 'GPT2Model':
        #         tinput_ids[0,-1] = end_id
        #         tattention_mask[0,-1] = end_attention
        #     trojan_input_ids.append(tinput_ids)
        #     trojan_attention_mask.append(tattention_mask)
        # trojan_input_ids = torch.cat(trojan_input_ids, axis=0)# .cuda()
        # trojan_attention_mask = torch.cat(trojan_attention_mask, axis=0)# .cuda()
        # print('input_ids', trojan_input_ids.shape, 'attention_mask', trojan_attention_mask.shape)
        # embedding_vectors = get_embedding_from_ids(trojan_input_ids, embedding, trojan_attention_mask, cls_token_is_first)
        # print('trojan fys', trojan_fys)
        # logits = bmodel(embedding_vectors.cuda()).cpu().detach().numpy()
        # preds = np.argmax(logits, axis=1)
        # print('benign_model trojan preds', np.sum(preds==0)/float(len(preds)), np.sum(preds==1)/float(len(preds)), preds)
        # tpreds = preds[np.where(trojan_fys==0)[0]]
        # print('benign_model trojan preds from 0', np.sum(tpreds==0)/float(len(tpreds)), np.sum(tpreds==1)/float(len(tpreds)), tpreds)
        # tpreds = preds[np.where(trojan_fys==1)[0]]
        # print('benign_model trojan preds from 1', np.sum(tpreds==0)/float(len(tpreds)), np.sum(tpreds==1)/float(len(tpreds)), tpreds)

    benign_logits0 = []
    # for bmodel in benign_models:
    #     blogits = bmodel(embedding_vectors.cuda()).cpu().detach().numpy()
    #     benign_logits0.append(blogits)

    # if number of images is less than given config
    f_n_re_imgs_per_label = optz_ys.shape[0] // num_classes

    # result = np.zeros((1,1,depth)), np.zeros((1,depth)), 0, '', '', '', 0, 1, 0.6, 1
    # trdelta = np.zeros((1,1,depth))
    # trdelta[0,0,23168] = 1
    # tresults = test(models, \
    #     benign_models, benign_logits0, benign_one_hots, benign_poses, benign_attentions, benign_ys, \
    #     model_type, test_one_hots, test_poses, test_attentions, test_ys, result, scratch_dirpath, num_classes, children, sample_layers, cls_token_is_first, end_id, trdelta)
    # sys.exit()

    sample_end = time.time()

    results = re_mask(model_type, models, \
            benign_models, benign_logits0, benign_one_hots, benign_poses, benign_attentions, benign_ys, embedding_signs, \
            neuron_dict, children, \
            optz_one_hots, optz_poses, optz_attentions, optz_ys, \
            n_neurons_dict, scratch_dirpath, re_epochs, num_classes, f_n_re_imgs_per_label, cls_token_is_first, max_input_length, end_id, trigger_ids)

    optm_end = time.time()

    print('# results', len(results))

    # sys.exit()

    # first test each trigger
    reasr_info = []
    reasr_per_labels = []
    result_infos = []
    diff_percents = []
    kwords = []
    top_dimentions = []
    if len(results) > 0:
        for result in results:
            rdelta, rmask, optz_label, RE_img, RE_mask, RE_delta, samp_label, base_label, acc, trigger_pos = result
            rmask = rmask * rmask > mask_epsilon

            for j in range(trigger_length):
                rdelta_argsort0 = np.argsort(rdelta[0][j])
                rdelta_argsort = []
                for r_i in rdelta_argsort0:
                    if r_i in use_idxs:
                        rdelta_argsort.append(r_i)
                for k in range(100):
                    kwords.append( tokenizer.convert_ids_to_tokens( [rdelta_argsort[-(k+1)]] )[0] ) 
                top_dimentions.append(np.where(rdelta_argsort0[::-1] == rdelta_argsort[-100])[0][0])
    print('kwords', kwords)
    top_dimentions = np.array(top_dimentions)
    print('top dimension', top_dimentions, len(rdelta_argsort), len(rdelta_argsort0))

    if for_submission:
        neutral_words_fn = '/all_possible_tokens.txt'
    else:
        neutral_words_fn = './all_possible_tokens.txt'
    neutral_words = {}
    neutral_words_idx = 0
    for line in open(neutral_words_fn):
        if len(line.split()) == 0:
            continue
        neutral_words[line.split()[0]] = neutral_words_idx
        neutral_words_idx += 1

    # print('neutral_words', neutral_words)

    final_words = []
    final_words_idxs = []
    for key in kwords:
        find_word = False
        for word in sorted(neutral_words.keys()):
            # if key.strip('Ġ').strip('#').lower() == word.lower() or key.lower() == word.lower() or key.strip('Ġ').strip('#').lower() in word:
            if key.strip('Ġ').strip('#').lower() == word.lower() or key.lower() == word.lower():
                final_words.append(word)
                final_words_idxs.append(neutral_words[word])
                find_word = True
        if not find_word:
            for word in sorted(neutral_words.keys()):
                if  key.strip('Ġ').strip('#').lower() in word:
                    final_words.append(word)
                    final_words_idxs.append(neutral_words[word])
                    break
    final_words = sorted(list(set(final_words)))
    final_words_idxs = sorted(list(set(final_words_idxs)))
    print('final words', len(final_words), final_words )

    if for_submission:
        if embedding.__class__.__name__ == 'DistilBertModel':
            nembs = pickle.load(open('/normal_samples.pkl', 'rb'))
            benign_tmax_diff2 = pickle.load(open('/submit_v2_benign_tmax_diff2_DistilBERT_1.pkl', 'rb'))
            benign_tmax_diff2 *= 0
            lr_dir1 = '/gy_distilbert_models/'
        elif embedding.__class__.__name__ == 'GPT2Model':
            nembs = pickle.load(open('/GPT2Model_samples_6_9.pkl', 'rb'))
            benign_tmax_diff2 = pickle.load(open('/submit_v2_benign_tmax_diff2_GPT-2_3.pkl', 'rb'))
            lr_dir1 = '/test_linear_models/GPT-2-last-5/'
    else:
        if embedding.__class__.__name__ == 'DistilBertModel':
            nembs = pickle.load(open('./normal_samples.pkl', 'rb'))
            benign_tmax_diff2 = pickle.load(open('./submit_v2_benign_tmax_diff2_DistilBERT_1.pkl', 'rb'))
            benign_tmax_diff2 *= 0
            lr_dir1 = './gy_distilbert_models/'
        elif embedding.__class__.__name__ == 'GPT2Model':
            nembs = pickle.load(open('./GPT2Model_samples_6_9.pkl', 'rb'))
            benign_tmax_diff2 = pickle.load(open('./submit_v2_benign_tmax_diff2_GPT-2_3.pkl', 'rb'))
            lr_dir1 = './test_linear_models/GPT-2-last-5/'

    n_samples = nembs.shape[0]
    mean_nembs = np.mean(nembs, axis=(0,1))
    print('nembs', nembs.shape, mean_nembs.shape)

    weights = []
    trigger_idxs = []
    trigger_fns = []
    if embedding.__class__.__name__ == 'DistilBertModel':
        trigger_idx = 0
        lr_fns = sorted(os.listdir(lr_dir1))
        for lr_fn in lr_fns:
            if int(lr_fn.split('.')[0]) not in final_words_idxs:
                continue
            lr_cls = pickle.load(open(os.path.join(lr_dir1, lr_fn), 'rb'))
            weight = lr_cls.coef_.reshape(-1)
            weights.append(weight)
            trigger_idxs.append(trigger_idx)
            trigger_idx += 1
        trigger_fns += lr_fns
    elif embedding.__class__.__name__ == 'GPT2Model':
        trigger_idx = 0
        lr_fns = sorted(os.listdir(lr_dir1))
        for lr_fn in lr_fns:
            if int(lr_fn.split('.')[0]) not in final_words_idxs:
                continue
            lr_cls = pickle.load(open(os.path.join(lr_dir1, lr_fn), 'rb'))
            weight = lr_cls.coef_.reshape(-1)
            weights.append(weight)
            trigger_idxs.append(trigger_idx)
            trigger_idx += 1
        trigger_fns += lr_fns

    print('weights', len(weights))

    embedding_vectors = torch.from_numpy(nembs).cuda()
    nlogits = model(embedding_vectors).cpu().detach().numpy()
    npreds = np.argmax(nlogits, axis=1)
    nacc0 = np.sum(npreds == 0)/float(len(npreds))
    nacc1 = np.sum(npreds == 1)/float(len(npreds))
    nlogits_diffs = [ np.mean(nlogits[:,1] - nlogits[:,0]), np.mean(np.abs(nlogits[:,1] - nlogits[:,0])), np.mean(nlogits[:,1]) - np.mean(nlogits[:,0]), ]
    nlogits_diffs += list(np.mean(nlogits, axis=0))
    nlogits_diffs += list(np.std(nlogits, axis=0))
    print('normal sapmling label acc', nacc0, nacc1, )

    n_vs = 11
    if embedding.__class__.__name__ == 'DistilBertModel':
        vs = [-2+_*0.4 for _ in range(n_vs)]
    elif embedding.__class__.__name__ == 'GPT2Model':
        vs = [-16+_*3.2 for _ in range(n_vs)]
    # vs = [-4+_*0.8 for _ in range(n_vs)]
    sample_accs = np.zeros((768,n_vs,2))
    sample_logits = np.zeros((768,n_vs,2))
    max_diffs = np.zeros((768,2))
    for i in range(nembs.shape[2]):
        tnembs = np.tile(nembs, [n_vs,1,1])
        for j in range(len(vs)):
            if embedding.__class__.__name__ == 'DistilBertModel':
                tnembs[j*n_samples:(j+1)*n_samples,:,i] = vs[j] 
            elif embedding.__class__.__name__ == 'GPT2Model':
                tnembs[j*n_samples:(j+1)*n_samples,:,i] = vs[j] + mean_nembs[i]

        embedding_vectors = torch.from_numpy(tnembs).cuda()
        tnlogits = model(embedding_vectors).cpu().detach().numpy()
        for j in range(len(vs)):
            ttnlogits = tnlogits[n_samples*j:n_samples*(j+1)]
            ttnpreds = np.argmax(ttnlogits, axis=1)
            ttnacc0 = np.sum(ttnpreds == 0)/float(len(ttnpreds))
            ttnacc1 = np.sum(ttnpreds == 1)/float(len(ttnpreds))
            sample_accs[i,j,0] = ttnacc0
            sample_accs[i,j,1] = ttnacc1
            sample_logits[i,j,0] = np.mean(ttnlogits[:,0])
            sample_logits[i,j,1] = np.mean(ttnlogits[:,1])

    diff_changes = np.zeros((768, sample_logits.shape[1]))
    for i in range(768):
        for j in range(sample_logits.shape[1]):
            diff_changes[i,j] = sample_logits[i,j,1] - sample_logits[i,j,0]
            # diff_changes[i,j] = sample_logits[i,j,0] - sample_logits[i,j,1]
    # tmax_diffs2 = np.abs(np.amax(diff_changes, axis=1) - np.amin(diff_changes, axis=1))
    tmax_diffs2 = diff_changes[:,-1] - diff_changes[:,0]

    tmax_diffs2 = np.abs(tmax_diffs2 - benign_tmax_diff2 )

    prods = []
    for w_i, weight in enumerate(weights):
        prod = np.abs( np.dot( (diff_changes[:,-1] - diff_changes[:,0] - benign_tmax_diff2)/np.mean(tmax_diffs2), weight/np.mean(np.abs(weight)) ) )
        prods.append(prod)
    
    if embedding.__class__.__name__ == 'DistilBertModel':
        output_bound = 167
        if model_type  == 'FCLinearModel':
            output_bound = 167
        elif  model_type  == 'GruLinearModel':
            output_bound = 167
        elif  model_type  == 'LstmLinearModel':
            output_bound = 167
    elif embedding.__class__.__name__ == 'GPT2Model':
        output_bound = 163
        if model_type  == 'FCLinearModel':
            output_bound = 163
        elif  model_type  == 'GruLinearModel':
            output_bound = 163
        elif  model_type  == 'LstmLinearModel':
            output_bound = 163

    pred = max(prods) > output_bound

    if embedding.__class__.__name__ == 'DistilBertModel':
        if model_type  == 'FCLinearModel':
            if pred:
                output = 0.97
            else:
                output = 0.13
        elif  model_type  == 'GruLinearModel':
            if pred:
                output = 0.94
            else:
                output = 0.14
        elif  model_type  == 'LstmLinearModel':
            if pred:
                output = 0.97
            else:
                output = 0.11
        else:
            # error model type
            if pred:
                output = 0.8
            else:
                output = 0.2
    elif embedding.__class__.__name__ == 'GPT2Model':
        if model_type  == 'FCLinearModel':
            if pred:
                output = 0.90
            else:
                output = 0.08
        elif  model_type  == 'GruLinearModel':
            if pred:
                output = 0.93
            else:
                output = 0.03
        elif  model_type  == 'LstmLinearModel':
            if pred:
                output = 0.95
            else:
                output = 0.05
        else:
            # error model type
            if pred:
                output = 0.8
            else:
                output = 0.2
    else:
        # error model type
        if pred:
            output = 0.8
        else:
            output = 0.2

    test_end = time.time()
    print('time', sample_end - start, optm_end - sample_end, test_end - optm_end)

    with open(result_filepath, 'w') as f:
        f.write('{0}'.format(output))

    print(model_filepath, embedding.__class__.__name__, output_bound, max(prods), pred)
    if not for_submission:
        with open(config['logfile'], 'a') as f:
            f.write('{0} {1} {2} {3} {4} {5} {6}\n'.format(model_filepath, embedding.__class__.__name__, output, max(prods), len(weights), np.mean(top_dimentions), test_end-start, ))


            # dump_name = os.path.join(scratch_dirpath, '{0}_{1}_{2}_{3}_{4}.pkl'.format(mname, base_label, trigger_pos, acc, benign_names))
            # with open(dump_name, 'wb') as f:
            #     pickle.dump((rdelta, trigger_length), f)

            # test_one_start = time.time()
            # reasr, reasr_per_label, benign_same_accs, reasr_before, label_results, label_pairs = test(models, \
            #         benign_models, benign_logits0, benign_one_hots, benign_poses, benign_attentions, benign_ys, \
            #         model_type, test_one_hots, test_poses, test_attentions, test_ys, result, scratch_dirpath, num_classes, children, sample_layers, cls_token_is_first, end_id)
            # test_one_end = time.time()
            # print('test one time', test_one_end - test_one_start)
            # # sys.exit()

            # benign_accs_str = '_'.join(['{:.2f}'.format(_) for _ in benign_same_accs])

            # # if reasr_per_label >= asr_bound:
            # if True:

            #     rdelta_words = []
            #     rdelta_idxs = []
            #     rdelta_combs = np.zeros( (int(math.pow(top_k_candidates, trigger_length)), trigger_length), dtype=np.int32 ) -1
            #     rdelta_comb_i = 0
            #     for j in range(trigger_length):
            #         kwords = []
            #         kidxs = []
            #         rdelta_argsort0 = np.argsort(rdelta[0][j])
            #         rdelta_argsort = []
            #         for r_i in rdelta_argsort0:
            #             if r_i in use_idxs:
            #                 rdelta_argsort.append(r_i)
            #         for k in range(top_k_candidates):
            #             print('position j', j, 'delta top', k, rdelta_argsort[-(k+1)], np.sort(rdelta[0][j])[-(k+1)])
            #             # kwords.append( np.argsort( rdelta[0][j])[-(k+1)] )
            #             kwords.append( tokenizer.convert_ids_to_tokens( [rdelta_argsort[-(k+1)]] )[0] )
            #             kidxs.append( rdelta_argsort[-(k+1)] )
            #             for p in range(int(math.pow(top_k_candidates, trigger_length))):
            #                 if ( p %int(math.pow(top_k_candidates, trigger_length-rdelta_comb_i)) ) \
            #                         // int(math.pow(top_k_candidates, trigger_length-1-rdelta_comb_i)) == k:
            #                     rdelta_combs[p, rdelta_comb_i] = rdelta_argsort[-(k+1)]
            #         kwords_str = '_'.join([str(_) for _ in kwords])
            #         kidxs_str = '_'.join([str(_) for _ in kidxs])
            #         rdelta_words.append(kwords_str)
            #         rdelta_idxs.append(kidxs_str)
            #         rdelta_comb_i += 1
            #     rdelta_words_str = ','.join([str(_) for _ in rdelta_words])
            #     rdelta_idxs_str = ','.join([str(_) for _ in rdelta_idxs])
            #     print('rdelta_combs', rdelta_combs.shape, rdelta_combs)

            #     test_reasr_per_labels = np.zeros( (int(math.pow(top_k_candidates, trigger_length)),) )
            #     test_benign_same_accs = []
            #     for j in range(rdelta_combs.shape[0]):
            #         rdelta_comb = rdelta_combs[j]
            #         trdelta = np.zeros(rdelta.shape)
            #         for k in range(trigger_length):
            #             trdelta[0,k,rdelta_comb[k]] = 1
            #         tresults = test(models, \
            #                 benign_models, benign_logits0, benign_one_hots, benign_poses, benign_attentions, benign_ys, \
            #                 model_type, test_one_hots, test_poses, test_attentions, test_ys, result, scratch_dirpath, num_classes, children, sample_layers, cls_token_is_first, end_id, trdelta)
            #         treasr_per_label = tresults[1]
            #         print('test', rdelta_comb, tresults[:3])
            #         test_reasr_per_labels[j] = treasr_per_label
            #         test_benign_same_accs.append(tresults[2])

            #     comb_benign_accs = []
            #     for benign_accs in test_benign_same_accs:
            #         comb_benign_accs.append( '_'.join(['{:.2f}'.format(_) for _ in benign_accs]) )
            #     comb_benign_accs_str = ','.join(comb_benign_accs)

            #     label_results_str = ','.join(['({0}:{1})'.format(_[0], _[1]) for _ in label_results])
            #     print('rdelta', rdelta_words, RE_delta, reasr, reasr_per_label, test_reasr_per_labels)

            #     reasr_info.append(['{:.2f}'.format(reasr), '{:.2f}'.format(reasr_per_label), ','.join(['{:.2f}'.format(_) for _ in test_reasr_per_labels]), max(test_reasr_per_labels), 'mask', str(optz_label), str(samp_label), str(base_label), 'trigger posistion', str(trigger_pos), RE_img, RE_mask, RE_delta, np.sum(rmask), acc, label_results_str, rdelta_words_str, rdelta_idxs_str, benign_accs_str, comb_benign_accs_str])
            #     reasr_per_labels.append(reasr_per_label)

            # else:

            #     rdelta_words = []
            #     rdelta_words_str = ''
            #     rdelta_idxs_str = ''
            #     comb_benign_accs_str = ''
            #     test_reasr_per_labels = []
            
            #     label_results_str = ','.join(['({0}:{1})'.format(_[0], _[1]) for _ in label_results])
            #     print('rdelta', rdelta_words, RE_delta, reasr, reasr_per_label, test_reasr_per_labels)

            #     reasr_info.append(['{:.2f}'.format(reasr), '{:.2f}'.format(reasr_per_label), ','.join(['{:.2f}'.format(_) for _ in test_reasr_per_labels]), max(test_reasr_per_labels), 'mask', str(optz_label), str(samp_label), str(base_label), 'trigger posistion', str(trigger_pos), RE_img, RE_mask, RE_delta, np.sum(rmask), acc, label_results_str, rdelta_words_str, rdelta_idxs_str, benign_accs_str, comb_benign_accs_str])
            #     reasr_per_labels.append(reasr_per_label)

    # test_end = time.time()
    # print('time', sample_end - start, optm_end - sample_end, test_end - optm_end)
    
    # for info in reasr_info:
    #     print('reasr info', info)
    # with open(config['logfile'], 'a') as f:
    #     for i in range(len(reasr_info)):
    #         f.write('reasr info {0}\n'.format( ' '.join([str(_) for _ in reasr_info[i]]) ))
    #     freasr_per_label = 0
    #     if len(reasr_per_labels) > 0:
    #         freasr_per_label = max(reasr_per_labels)
    #     freasr = freasr_per_label
    #     f.write('{0} {1} {2} {3} {4} {5} {6} {7} {8}\n'.format(\
    #             model_filepath, model_type, 'mode', freasr, freasr_per_label, 'time', sample_end - start, optm_end - sample_end, test_end - optm_end) )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Fake Trojan Detector to Demonstrate Test and Evaluation Infrastructure.')
    parser.add_argument('--model_filepath', type=str, help='File path to the pytorch model file to be evaluated.', default='./model/model.pt')
    parser.add_argument('--cls_token_is_first', help='Whether the first embedding token should be used as the summary of the text sequence, or the last token.', action='store_true', default=False)
    parser.add_argument('--tokenizer_filepath', type=str, help='File path to the pytorch model (.pt) file containing the correct tokenizer to be used with the model_filepath.', default='./model/tokenizer.pt')
    parser.add_argument('--embedding_filepath', type=str, help='File path to the pytorch model (.pt) file containing the correct embedding to be used with the model_filepath.', default='./model/embedding.pt')
    parser.add_argument('--result_filepath', type=str, help='File path to the file where output result should be written. After execution this file should contain a single line with a single floating point trojan probability.', default='./output.txt')
    parser.add_argument('--scratch_dirpath', type=str, help='File path to the folder where scratch disk space exists. This folder will be empty at execution start and will be deleted at completion of execution.', default='./scratch')
    parser.add_argument('--examples_dirpath', type=str, help='File path to the folder of examples which might be useful for determining whether a model is poisoned.', default='./model/clean_example_data')

    args = parser.parse_args()

    fake_trojan_detector(args.model_filepath, args.cls_token_is_first, args.tokenizer_filepath, args.embedding_filepath, args.result_filepath, args.scratch_dirpath, args.examples_dirpath)

