import os, sys
import pickle
import numpy as np

import scipy.stats
import scipy.spatial
import scipy.special
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.preprocessing import normalize
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.linear_model import LinearRegression, SGDClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
import sklearn.metrics

np.set_printoptions(precision=4, linewidth=200, suppress=True )

# ne = 40000
# md = 5000

ne = 20000
md = 5000


C = 0.5
# ne = int(sys.argv[1])
# md = int(sys.argv[2])
# C = float(sys.argv[3])

print('ne', ne, 'md', md, 'C', C)

metadata_csv = '/data/share/trojai/trojai-round5-v2-dataset/METADATA.csv'
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
        print(method_id1, method_id2)

    else:
        words = line.split(',')
        mname = words[0]
        if words[1] == 'True':
            gt_results[mname] = 1
        else:
            gt_results[mname] = 0
        
        all_models.append(mname)

        gt_model_info[mname] = [ gt_results[mname], words[method_id1], words[method_id2], words[embedding_id], words[arch_id]]

def train_and_test_classifier(train_X, train_y, test_X, test_y):
    cls = RandomForestClassifier(n_estimators=ne, max_depth=md, criterion='entropy', n_jobs=16)
    # cls = RandomForestClassifier(n_estimators=ne, max_depth=md, criterion='entropy')
    # cls = RandomForestClassifier(n_estimators=ne, max_depth=md, n_jobs=16)
    cls.fit(train_X, train_y)

    lr_reg = LogisticRegression(C=C, max_iter=10000, tol=1e-4)
    # lr_reg.fit(cls.predict_proba(train_X), train_y)
    lr_reg.fit(np.concatenate([train_X, cls.predict_proba(train_X)], axis=1) , train_y)
    confs = lr_reg.predict_proba( np.concatenate([test_X, cls.predict_proba(test_X)], axis=1) )[:,1]

#     lr_reg = MLPRegressor(hidden_layer_sizes=(2000,1000,500,100), max_iter=int(1e8), learning_rate_init=1e-3, tol=1e-4)
#     lr_reg.fit(np.concatenate([train_X, cls.predict_proba(train_X)], axis=1) , train_y)
#     confs = lr_reg.predict( np.concatenate([test_X, cls.predict_proba(test_X)], axis=1) )

    # clf = CalibratedClassifierCV(cls, method='isotonic')
    # clf.fit(cls.predict_proba(train_X), train_y)
    # confs = clf.predict_proba(cls.predict_proba(test_X))[:,1]
    # lr_reg = clf

    confs = np.clip(confs, 0.025, 0.975)

    return cls, lr_reg, confs

benign_tmax_diff2 = pickle.load(open('./benign_tmax_diff2.pkl', 'rb'))

results = {}
Xs = []
ys = []
dirname = './v2_sample_embs_abs/'
# dirname2 = './v2_sample_embs_abs4/'
# dirname2 = './v2_sample_embs_abs2/'
# dirname = './v2_sample_embs_abs_inputs/'
fns = sorted(os.listdir(dirname))
for fn in fns:
    if fn == 'samples.pkl':
        continue
    infos = pickle.load(open(os.path.join(dirname, fn), 'rb'))
    mname = fn[:-4].split('_')[1]
    # if gt_model_info[mname][-1] != 'GruLinear' or gt_model_info[mname][-2] != 'DistilBERT':
    # if gt_model_info[mname][-1] != 'LstmLinear' or gt_model_info[mname][-2] != 'DistilBERT':
        # continue
    features = []
    # features += list(infos[0])
    samples_accs   = infos[1]
    max_acc_diffs = np.zeros((768,2))
    for i in range(768):
        max_acc_diffs[i,0] = np.amax(samples_accs[i,:,0]) - np.amin(samples_accs[i,:,0])
        max_acc_diffs[i,1] = np.amax(samples_accs[i,:,1]) - np.amin(samples_accs[i,:,1])
    samples_logits = infos[2]
    max_diffs = infos[3]
    # features += list(np.sort(max_diffs[:,0])[-5:])
    # features += list(np.sort(max_diffs[:,1])[-5:])
    # features += list(max_acc_diffs.reshape(-1))
    features += list(max_diffs.reshape(-1))

    diff_changes = np.zeros((768, samples_logits.shape[1]))
    for i in range(768):
        for j in range(samples_logits.shape[1]):
            diff_changes[i,j] = samples_logits[i,j,1] - samples_logits[i,j,0]
    # tmax_diffs2 = np.abs(np.amax(diff_changes, axis=1) - np.amin(diff_changes, axis=1))
    tmax_diffs2 = diff_changes[:,-1] - diff_changes[:,0]
    tmax_diffs2 = np.abs(tmax_diffs2 - benign_tmax_diff2 )

    features += list(tmax_diffs2.reshape(-1))

    # infos2 = pickle.load(open(os.path.join(dirname2, fn), 'rb'))
    # max_diffs2 = infos2[3]
    # features += list(max_diffs2.reshape(-1))

    # features += list(np.sort(max_diffs[:,0])[-20:])
    # features += list(np.sort(max_diffs[:,1])[-20:])
    # features += list(np.argsort(max_diffs[:,0])[-20:])
    # features += list(np.argsort(max_diffs[:,1])[-20:])

    # mean_logits = np.mean(samples_logits, axis=1).reshape(-1)
    # features += list(mean_logits)
    # print(mname, max_diffs[97], max_diffs[459], max_diffs[464], )
    arch_i = 0
    if gt_model_info[mname][-1] == 'GruLinear':
        arch_i = 0
    elif gt_model_info[mname][-1] == 'LstmLinear':
        arch_i = 1
    else:
        print('error arch', gt_model_info)
        sys.exit()
    emb_i = 0
    if gt_model_info[mname][-2] == 'DistilBERT':
        emb_i = 0
    elif gt_model_info[mname][-2] == 'BERT':
        emb_i = 1
    elif gt_model_info[mname][-2] == 'GPT-2':
        emb_i = 2
    else:
        print('error arch', gt_model_info)
        sys.exit()
    features += [arch_i]
    features += [emb_i]
    # print(mname, gt_results[mname], len(features), arch_i, samples_logits.shape, max_diffs.shape,\
            # np.sort(max_diffs[0,:,0])[-5:], np.sort(max_diffs[0,:,1])[-5:], np.argsort(max_diffs[0,:,0])[-5:], np.argsort(max_diffs[0,:,1])[-5:],\
            # np.sort(max_diffs[1,:,0])[-5:], np.sort(max_diffs[1,:,1])[-5:], np.argsort(max_diffs[1,:,0])[-5:], np.argsort(max_diffs[1,:,1])[-5:],\
            # np.sort(max_diffs[:,0])[-5:], np.sort(max_diffs[:,1])[-5:], np.argsort(max_diffs[:,0])[-5:], np.argsort(max_diffs[:,1])[-5:],\
            # )
    Xs.append(np.array(features))
    ys.append(gt_results[mname])


Xs = np.array(Xs)
ys = np.array(ys)
print('Xs', Xs.shape, ys.shape)

cls, lr_reg, confs = train_and_test_classifier(Xs, ys, Xs, ys)

# pickle.dump((cls, lr_reg), open('./rf_lr_abs4.pkl', 'wb'))

preds = confs

# preds = cls.predict(Xs)
fp = 0
tp = 0
fn = 0
tn = 0
tps = []
fps = []
fns = []
for i in range(ys.shape[0]):
    if preds[i] > 0.5 and ys[i] == 1:
        tp += 1
    elif preds[i] > 0.5 and ys[i] == 0:
        fp += 1
    elif preds[i] <= 0.5 and ys[i] == 0:
        tn += 1
    elif preds[i]<= 0.5 and ys[i] == 1:
        fn += 1
print('train on full set', 'tp', tp, 'fp', fp, 'fn', fn, 'tn', tn, 'acc', (tp+tn)/float(tp+fp+fn+tn))
roc_auc = sklearn.metrics.roc_auc_score(ys, confs)
celoss  = sklearn.metrics.log_loss(ys, confs)
print('roc_auc', roc_auc, 'celoss', celoss)

print('ne', ne, 'md', md)
train_accs = []
test_accs = []
roc_aucs = []
ce_losses = []
kf = KFold(n_splits=5, shuffle=True)
for train_index, test_index in kf.split(Xs):
    train_X, test_X = Xs[train_index], Xs[test_index]
    train_y, test_y = ys[train_index], ys[test_index]

    cls, lr_reg, confs = train_and_test_classifier(train_X, train_y, test_X, test_y)

    preds = cls.predict(train_X)

    fp = 0
    tp = 0
    fn = 0
    tn = 0
    for i in range(train_y.shape[0]):
        # print(train_X[i], preds[i])
        if preds[i] > 0.5 and train_y[i] == 1:
            tp += 1
        elif preds[i] > 0.5 and train_y[i] == 0:
            fp += 1
        elif preds[i] <= 0.5 and train_y[i] == 0:
            tn += 1
        elif preds[i] <= 0.5 and train_y[i] == 1:
            fn += 1
    print('train', 'tp', tp, 'fp', fp, 'fn', fn, 'tn', tn, 'acc', (tp+tn)/float(tp+fp+fn+tn))
    train_accs.append((tp+tn)/float(tp+fp+fn+tn))
    
    preds = cls.predict(test_X)
    
    fp = 0
    tp = 0
    fn = 0
    tn = 0
    tps = []
    fps = []
    fns = []
    for i in range(test_y.shape[0]):
        if preds[i] > 0.5 and test_y[i] == 1:
            tp += 1
            # tps.append(tnames[i])
        elif preds[i] > 0.5 and test_y[i] == 0:
            fp += 1
            # fps.append(tnames[i])
            # print('fp', tnames[i], test_X[i])
        elif preds[i] <= 0.5 and test_y[i] == 0:
            tn += 1
        elif preds[i]<= 0.5 and test_y[i] == 1:
            fn += 1
            # fns.append(tnames[i])
            # print('fn', tnames[i], test_X[i])
    print('test', 'tp', tp, 'fp', fp, 'fn', fn, 'tn', tn, 'acc', (tp+tn)/float(tp+fp+fn+tn))
    test_accs.append((tp+tn)/float(tp+fp+fn+tn))


    print('test_y',test_y.shape)
    roc_auc = sklearn.metrics.roc_auc_score(test_y, confs)
    celoss  = sklearn.metrics.log_loss(test_y, confs)
    print('roc_auc', roc_auc, 'celoss', celoss)
    roc_aucs.append(roc_auc)
    ce_losses.append(celoss)
train_accs = np.array(train_accs)
test_accs  = np.array(test_accs)
roc_aucs = np.array(roc_aucs)
ce_losses  = np.array(ce_losses)
print('train accs', np.mean(train_accs), np.var(train_accs))
print('test accs', np.mean(test_accs), np.var(test_accs))
print('test roc_aucs', np.mean(roc_aucs), np.var(roc_aucs))
print('test ce_losses', np.mean(ce_losses), np.var(ce_losses))
