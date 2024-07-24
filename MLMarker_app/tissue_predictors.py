import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
import time

class TissuePredictor:
    def __init__(self, X_train_path, X_test_path, y_train_path, y_test_path, dict_train_label_weight_path):
        # Load the data
        self.X_train = pd.read_csv(X_train_path)
        self.X_test = pd.read_csv(X_test_path)
        self.y_train = pd.read_csv(y_train_path)
        self.y_test = pd.read_csv(y_test_path)
        dict_train_label_weight = pd.read_csv(dict_train_label_weight_path)
        self.dict_train_label_weights = {item[0]: item[1] for item in dict_train_label_weight.values.tolist()}
        self.train_weights = list(self.dict_train_label_weights.values())
        self.num_classes = len(np.unique(self.y_train))
        print('Data loaded')

    def cv_comparison(self, models, names, X, y, cv):
        cv_scores = pd.DataFrame()
        accs, f1s, precs, recs, f1s_w, precs_w, recs_w = [], [], [], [], [], [], []

        for model, name in zip(models, names):
            print(name)
            start = time.time()
            acc = np.round(cross_val_score(model, X, y, scoring='accuracy', cv=cv), 4)
            accs.append(acc)
            acc_avg = round(np.mean(acc[~np.isnan(acc)]), 4)
            f1 = np.round(cross_val_score(model, X, y, scoring='f1_macro', cv=cv), 4)
            f1s.append(f1)
            f1_avg = round(np.mean(f1[~np.isnan(f1)]), 4)
            prec = np.round(cross_val_score(model, X, y, scoring='precision_macro', cv=cv), 4)
            precs.append(prec)
            prec_avg = round(np.mean(prec[~np.isnan(prec)]), 4)
            rec = np.round(cross_val_score(model, X, y, scoring='recall_macro', cv=cv), 4)
            recs.append(rec)
            rec_avg = round(np.mean(rec[~np.isnan(rec)]), 4)
            f1_w = np.round(cross_val_score(model, X, y, scoring='f1_weighted', cv=cv), 4)
            f1s_w.append(f1_w)
            f1_w_avg = round(np.mean(f1_w[~np.isnan(f1_w)]), 4)
            prec_w = np.round(cross_val_score(model, X, y, scoring='precision_weighted', cv=cv), 4)
            precs_w.append(prec_w)
            prec_w_avg = round(np.mean(prec_w[~np.isnan(prec_w)]), 4)
            rec_w = np.round(cross_val_score(model, X, y, scoring='recall_weighted', cv=cv), 4)
            recs_w.append(rec_w)
            rec_w_avg = round(np.mean(rec_w[~np.isnan(rec_w)]), 4)
            cv_scores[name] = [acc_avg, f1_avg, prec_avg, rec_avg, f1_w_avg, prec_w_avg, rec_w_avg]
            print(time.time() - start)
        cv_scores.index = ['Accuracy', 'f1_macro', 'precision_macro', 'recall_macro', 'f1_weighted', 'precision_weighted', 'recall_weighted']
        return cv_scores, accs, f1s, precs, recs, f1s_w, precs_w, recs_w

    def train_models(self, output_prefix):
        xgb_unbal = XGBClassifier(random_state=42, objective='multi:softprob', eval_metric='mlogloss', num_class=self.num_classes, n_jobs=-1)
        xgb = XGBClassifier(random_state=42, objective='multi:softprob', eval_metric='mlogloss', num_class=self.num_classes, weight=self.train_weights, n_jobs=-1)
        svm_unbal = SVC(random_state=42)
        svm = SVC(random_state=42, class_weight=self.dict_train_label_weights)

        models_xgb = [xgb_unbal, xgb]
        names_xgb = ['XGBClassifier unbalanced', 'XGBClassifier dict balanced']
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
        comp_xgb, _, _, _, _, _, _, _ = self.cv_comparison(models_xgb, names_xgb, self.X_train, self.y_train, cv=cv)
        comp_xgb.to_csv(f'{output_prefix}_XGB.csv', sep='/')

        models_svm = [svm_unbal, svm]
        names_svm = ['SVM unbalanced', 'SVM']
        comp_svm, _, _, _, _, _, _, _ = self.cv_comparison(models_svm, names_svm, self.X_train, self.y_train, cv=cv)
        comp_svm.to_csv(f'{output_prefix}_SVM.csv', sep='/')
