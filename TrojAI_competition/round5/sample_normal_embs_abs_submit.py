# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

# add trigger instaed of replacing

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
import scipy.special

np.set_printoptions(precision=4, linewidth=200, suppress=True)

import warnings
warnings.filterwarnings("ignore")

for_submission = True
asr_bound = 0.9

mask_epsilon = 0.01
use_amp = False  # attempt to use mixed precision to accelerate embedding conversion process
top_k_candidates = 5
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
    config['gpu_id'] = '4'
config['print_level'] = 2
config['random_seed'] = 333
config['channel_last'] = 0
config['w'] = 224
config['h'] = 224
config['reasr_bound'] = 0.4
config['batch_size'] = 5
config['has_softmax'] = 0
config['samp_k'] = 2.
config['same_range'] = 0
config['n_samples'] = 3
config['samp_batch_size'] = 32
config['top_n_neurons'] = 3
config['n_sample_imgs_per_label'] = 2
config['re_batch_size'] = 16
config['max_troj_size'] = 1200
config['filter_multi_start'] = 1
# config['re_mask_lr'] = 5e-2
config['re_mask_lr'] = 3e-2 # for BERT
# config['re_mask_lr'] = 1e-1
# config['re_mask_lr'] = 1e0
config['re_mask_weight'] = 100
config['mask_multi_start'] = 1
config['re_epochs'] = 100
# config['re_epochs'] = 50
config['n_re_imgs_per_label'] = 20
config['trigger_length'] = 2
config['logfile'] = './r5_v2_embs_normal_test_abs_submit.txt'

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

def get_embedding_from_ids(input_ids, embedding, attention_mask, cls_token_is_first):
    # print('input_ids', input_ids.shape)
    embedding_vector = embedding(input_ids, attention_mask=attention_mask)[0]
    if cls_token_is_first:
        embedding_vector = embedding_vector[:, :1, :]
    else:
        embedding_vector = embedding_vector[:, -1:, :]
    return embedding_vector

def get_inners(temp_model1, batch_data):
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
        print('error temp model', temp_model1.__class__.__name__)
    return inner_outputs

def fake_trojan_detector(model_filepath, cls_token_is_first, tokenizer_filepath, embedding_filepath, result_filepath, scratch_dirpath, examples_dirpath):
    start_time = time.time()

    print('cls_token_is_first = {}'.format(cls_token_is_first))
    print('tokenizer_filepath = {}'.format(tokenizer_filepath))
    print('embedding_filepath = {}'.format(embedding_filepath))
    print('result_filepath = {}'.format(result_filepath))
    print('scratch_dirpath = {}'.format(scratch_dirpath))

    model = torch.load(model_filepath).cuda()
    print('model_filepath = {}'.format(model_filepath))
    print('examples_dirpath = {}'.format(examples_dirpath))

    if not for_submission:
        model_json_filepath_list = model_filepath.split('/')[:-1]
        model_json_filepath_list.append('config.json')
        model_json_filepath_list.insert(0, '/')
        model_json_filepath = os.path.join(*model_json_filepath_list)
        print(model_json_filepath)
        with open(model_json_filepath, mode='r', encoding='utf-8') as f:
            json_data = jsonpickle.decode(f.read()) 
        config_embedding = json_data['embedding']   # This comes from the config.json file
        if config_embedding == 'BERT':
            # tokenizer = transformers.BertTokenizer.from_pretrained(config_embedding_flavor)
            # embedding = transformers.BertModel.from_pretrained(config_embedding_flavor).cuda()
            tokenizer = torch.load('/data/share/trojai/trojai-round5-v2-dataset/tokenizers/BERT-bert-base-uncased.pt')
            embedding = torch.load('/data/share/trojai/trojai-round5-v2-dataset/embeddings/BERT-bert-base-uncased.pt').cuda()
            cls_token_is_first = True
        elif config_embedding == 'GPT-2':
            # tokenizer = transformers.GPT2Tokenizer.from_pretrained(config_embedding_flavor)
            # embedding = transformers.GPT2Model.from_pretrained(config_embedding_flavor).cuda()
            tokenizer = torch.load('/data/share/trojai/trojai-round5-v2-dataset/tokenizers/GPT-2-gpt2.pt')
            embedding = torch.load('/data/share/trojai/trojai-round5-v2-dataset/embeddings/GPT-2-gpt2.pt').cuda()
            cls_token_is_first = False
        elif config_embedding == 'DistilBERT':
            # tokenizer = transformers.DistilBertTokenizer.from_pretrained(config_embedding_flavor)
            # embedding = transformers.DistilBertModel.from_pretrained(config_embedding_flavor).cuda()
            tokenizer = torch.load('/data/share/trojai/trojai-round5-v2-dataset/tokenizers/DistilBERT-distilbert-base-uncased.pt')
            embedding = torch.load('/data/share/trojai/trojai-round5-v2-dataset/embeddings/DistilBERT-distilbert-base-uncased.pt').cuda()
            cls_token_is_first = True
    else:
        tokenizer = torch.load(tokenizer_filepath)
        embedding = torch.load(embedding_filepath).cuda()

    # set the padding token if its undefined
    if not hasattr(tokenizer, 'pad_token') or tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    max_input_length = tokenizer.max_model_input_sizes[tokenizer.name_or_path]

    print('max_input_length', max_input_length, 'tokenizer', tokenizer.__class__.__name__, 'embedding', embedding.__class__.__name__, 'max_input_length', max_input_length, 'cls_token_is_first', cls_token_is_first)

    embedding.eval()

    if embedding.__class__.__name__ == 'DistilBertModel':
        end_id = 102
    elif embedding.__class__.__name__ == 'BertModel':
        end_id = 102
    elif embedding.__class__.__name__ == 'GPT2Model':
        end_id = 50256
    else:
        print('error embedding type', embedding.__class__.__name__)
        sys.exit()


    n_samples = 200
    # nembs = (np.random.randn(n_samples,1,768) * 2).astype(np.float32)
    if for_submission:
        nembs = pickle.load(open('/samples.pkl', 'rb'))
        cls, lr_reg= pickle.load(open('/rf_lr_abs4.pkl', 'rb'))
        benign_tmax_diff2 = pickle.load(open('/benign_tmax_diff2.pkl', 'rb'))
    else:
        nembs = pickle.load(open('./samples.pkl', 'rb'))
        cls, lr_reg= pickle.load(open('./rf_lr_abs4.pkl', 'rb'))
        benign_tmax_diff2 = pickle.load(open('./benign_tmax_diff2.pkl', 'rb'))

    model.eval()

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
    if model_type  == 'LinearModel':
        target_layers = ['Linear']
    elif  model_type  == 'GruLinearModel':
        target_layers = ['GRU']
    elif  model_type  == 'LstmLinearModel':
        target_layers = ['LSTM']


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
    vs = [-2+_*0.4 for _ in range(n_vs)]
    # vs = [-4+_*0.8 for _ in range(n_vs)]
    sample_accs = np.zeros((768,n_vs,2))
    sample_logits = np.zeros((768,n_vs,2))
    max_diffs = np.zeros((768,2))
    for i in range(nembs.shape[2]):
        tnembs = np.tile(nembs, [n_vs,1,1])
        for j in range(len(vs)):
            tnembs[j*n_samples:(j+1)*n_samples,:,i] = vs[j] 

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

        # print('dim', i, ttnlogits.shape, sample_accs[i][-1] - sample_accs[i][0], sample_logits[i][-1] - sample_logits[i][0],\
                # np.amax(sample_logits[i], axis=0) - np.amin(sample_logits[i], axis=0), )
        max_diffs[i,0] = np.amax(sample_logits[i,:,0]) - np.amin(sample_logits[i,:,0])
        max_diffs[i,1] = np.amax(sample_logits[i,:,1]) - np.amin(sample_logits[i,:,1])

    print('tnembs', tnembs.shape)
    print('top sampling', np.sort(max_diffs[:,0])[-5:], np.sort(max_diffs[:,1])[-5:], )

    diff_changes = np.zeros((768, sample_logits.shape[1]))
    for i in range(768):
        for j in range(sample_logits.shape[1]):
            diff_changes[i,j] = sample_logits[i,j,1] - sample_logits[i,j,0]
    # tmax_diffs2 = np.abs(np.amax(diff_changes, axis=1) - np.amin(diff_changes, axis=1))
    tmax_diffs2 = diff_changes[:,-1] - diff_changes[:,0]
    tmax_diffs2 = np.abs(tmax_diffs2 - benign_tmax_diff2 )

    accs_info = [nacc0, nacc1, ]

    features = []
    features += list(max_diffs.reshape(-1))
    features += list(tmax_diffs2.reshape(-1))
    arch_i = 0
    if model_type == 'GruLinearModel':
        arch_i = 0
    elif model_type == 'LstmLinearModel':
        arch_i = 1
    else:
        print('error arch', gt_model_info)
        sys.exit()
    emb_i = 0
    if embedding.__class__.__name__ == 'DistilBertModel':
        emb_i = 0
    elif embedding.__class__.__name__ == 'BertModel':
        emb_i = 1
    elif embedding.__class__.__name__ == 'GPT2Model':
        emb_i = 2
    else:
        print('error arch', gt_model_info)
        sys.exit()
    features += [arch_i]
    features += [emb_i]

    features = np.array(features).reshape((1,-1))
    confs = lr_reg.predict_proba( np.concatenate([features, cls.predict_proba(features)], axis=1) )[:,1]
    confs = np.clip(confs, 0.025, 0.975)

    output = confs[0]

    with open(result_filepath, 'w') as f:
        f.write('{0}'.format(output))

    print(model_filepath, embedding.__class__.__name__, asr_bound, confs, output)
    if not for_submission:
        with open(config['logfile'], 'a') as f:
            f.write('{0} {1} {2}\n'.format(model_filepath, embedding.__class__.__name__, output))

    end_time = time.time()
    print('evaluate time per model', end_time - start_time)


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



