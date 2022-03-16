# TrojAI competition submission
In this folder, we upload the TrojAI competition submission code for each round.

## Round 1
Round 1 code is the submission for 20200725T153001 of Perspecta team. This submission has 0.30 cross entropy and 0.91 roc-auc on test set and 0.281 cross entropy loss and 0.92 roc-auc on holdout set. 

## Round 2
Round 2 code is the submission for 20201025T122002 of Perspecta-PurdueRutgers team. This submission has 0.32 cross entropy and 0.89 roc-auc on test set and 0.38 cross entropy loss and 0.87 roc-auc on holdout set.

## Round 3
Round 3 code is the submission for 20201111T185002 of Perspecta-PurdueRutgers team. This submission has 0.32 cross entropy and 0.90 roc-auc on test set and 0.31 cross entropy loss and 0.91 roc-auc on holdout set.

## Round 4
Round 4 code is the submission for 20210120T222001 of Perspecta-PurdueRutgers team. This submission has 0.32 cross entropy and 0.90 roc-auc on test set and 0.34 cross entropy loss and 0.90 roc-auc on holdout set



## Round 5
Round 5 code is the submission for  20210316T161002 of Perspecta-PurdueRutgers team This submission has 0.32 cross entropy and 0.93 roc-auc and 0.26 cross entropy and 0.95 roc-auc on holdout set.

To detect whether a model is trojaned, please run `sample_normal_embs_abs_submit.py`. The classifier file `rf_lr_abs4.pkl` which `sample_normal_embs_abs_submit.py` depends is too large so I do not upload on github. To train this classifier, first run `sample_normal_embs_abs.py` to generate samples and then run `classify_normal_embs_abs1_5.py` to generate classifier `rf_lr_abs4.pkl`.
