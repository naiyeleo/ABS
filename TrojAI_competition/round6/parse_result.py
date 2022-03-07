import os, sys
import numpy as np
import pickle
import jsonpickle

import scipy.stats
import scipy.spatial
import scipy.special
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.preprocessing import normalize
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
import sklearn.metrics

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn

np.set_printoptions(precision=2, linewidth=200, suppress=True )

embedding_name = 'DistilBERT'
# embedding_name = 'BERT'
# embedding_name = 'GPT-2'

metadata_csv = '/data/share/trojai/trojai-round6-v2-dataset/METADATA.csv'
method_id = 0
base_label_id = 0
target_label_id = 0
gt_results = {}                                                                                                                                                                                         
gt_model_info = {}
mask_fps = []
tp_models = []
all_models = []
for line in open(metadata_csv):
    if len(line.split(',')) == 0:
        continue
    if not line.startswith('id-0000'):
        # head line
        words = line.split(',')

        # base_label_id = words.index('triggered_classes')
        # target_label_id = words.index('trigger_target_class')

        method_id1 = words.index('triggers_0_type')
        method_id2 = words.index('triggers_1_type')
        embedding_id = words.index('embedding')
        arch_id = words.index('model_architecture')

    else:
        words = line.split(',')
        mname = words[0]
        if words[1] == 'True':
            gt_results[mname] = 1
        else:
            gt_results[mname] = 0
        
        all_models.append(mname)

        gt_model_info[mname] = [ gt_results[mname], words[method_id1], words[method_id2], words[embedding_id], words[arch_id]]


trigger_words = []
trigger_words_dict = {}
trigger_words_full_dict = {}
n_trojan_models = 0
for i in range(48):
    mname = 'id-{:08d}'.format(i)
    model_json_filepath = '/data/share/trojai/trojai-round6-v2-dataset/models/{0}/config.json'.format(mname)

    with open(model_json_filepath, mode='r', encoding='utf-8') as f:
        json_data = jsonpickle.decode(f.read()) 

    if json_data['triggers'] is None:
        continue
    for trigger_dict in json_data['triggers']:
        # if trigger_dict['insert_min_location_percentage'] is None or trigger_dict['insert_min_location_percentage'] == 0.0:
        trigger_word = trigger_dict['text']
        victim_label = int(trigger_dict['source_class'])
        target_label = int(trigger_dict['target_class'])
        min_location = trigger_dict['insert_min_location_percentage']
        trigger_words.append(trigger_word)
        if mname in trigger_words_dict.keys():
            trigger_words_dict[mname].append(trigger_word)
            trigger_words_full_dict[mname].append((trigger_word, victim_label, target_label, min_location))
        else:
            trigger_words_dict[mname] = [trigger_word]
            trigger_words_full_dict[mname] = [(trigger_word, victim_label, target_label, min_location)]

    n_trojan_models += 1

print('# trigger words', len(trigger_words))
trigger_words_distinct = sorted(list(set(trigger_words)))
print('# distinct trigger words', len(trigger_words_distinct), trigger_words_distinct)
print('# trojan models', n_trojan_models)

for i in range(len(trigger_words_distinct)):
    print(i, trigger_words_distinct[i])

results = {}
for line in open('./r6_v2_embs_normal_test_abs_submit.txt'):
    words = line.split()
    if len(words) == 0:
        continue
    mname = words[0].split('/')[-2]
    val = float(words[-1])
    results[mname] = val

bound = 0.5
correct = 0
total = 0
tp = 0
fp = 0
fn = 0
tn = 0
tps = []
fps = []
fns = []
names = []
for mname in sorted(results.keys()):
    if mname not in all_models:
        continue
    max_acc = 0
    if mname in results.keys():
        max_acc = results[mname]

    if (max_acc >= bound ) and gt_results[mname] == 1:
        correct += 1
        tp += 1
        tps.append(mname)
        # print('tp', mname, max_acc, gt_results[mname], gt_model_info[mname])
    elif not (max_acc >= bound) and gt_results[mname] == 0:
        correct += 1
        tn += 1
        # print('tn', mname, max_acc, gt_results[mname], gt_model_info[mname])
    elif (max_acc >= bound ) and gt_results[mname] == 0:
        fp += 1
        fps.append(mname)
        print('fp', mname, max_acc, gt_results[mname], gt_model_info[mname])
    elif not (max_acc >= bound ) and gt_results[mname] == 1:
        fn += 1
        fns.append(mname)
        print('fn', mname, max_acc, gt_results[mname], gt_model_info[mname])
    total += 1
print('bound', bound)
print('correct', correct, 'total', total, 'acc', float(correct)/total)
print('tp', tp, 'fp', fp, 'fn', fn, 'tn', tn)
print('fns', ' '.join([_[-4:] for _ in fns]) )
