Round 5 code is the submission for 20210316T161002 of Perspecta-PurdueRutgers team This submission has 0.32 cross entropy and 0.93 roc-auc and 0.26 cross entropy and 0.95 roc-auc on holdout set.

To detect whether a model is trojaned, please run sample_normal_embs_abs_submit.py. The classifier file rf_lr_abs4.pkl which sample_normal_embs_abs_submit.py depends is too large so I do not upload on github. To train this classifier, first run sample_normal_embs_abs.py to generate samples and then run classify_normal_embs_abs1_5.py to generate classifier rf_lr_abs4.pkl.
