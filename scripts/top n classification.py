# !/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Longxiao
# Date: 2025/10/23
import pandas as pd
from sklearn.model_selection import KFold
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import glob

top_n = 1  # For naming purposes only

def balance_dataset(data, labels, ids):

    combined_data = pd.concat([data, labels], axis=1)
    combined_data['ID'] = ids.values

    class_0 = combined_data[combined_data.iloc[:, -2] == 0]
    class_1 = combined_data[combined_data.iloc[:, -2] == 1]

    min_samples = min(len(class_0), len(class_1))

    if len(class_0) > min_samples:
        class_0_balanced = resample(class_0, replace=False, n_samples=min_samples, random_state=42)
    else:
        class_0_balanced = class_0

    if len(class_1) > min_samples:
        class_1_balanced = resample(class_1, replace=False, n_samples=min_samples, random_state=42)
    else:
        class_1_balanced = class_1

    balanced_data = pd.concat([class_0_balanced, class_1_balanced])

    X_balanced = balanced_data.iloc[:, :-2]  # 特征
    y_balanced = balanced_data.iloc[:, -2]  # 标签
    id_balanced = balanced_data['ID']  # ID

    return X_balanced, y_balanced, id_balanced


top_n_tables_folder = r'C:\Users\lenovo\Desktop\top_1_features_classification_tables_hc'
results_folder = r'C:\Users\lenovo\Desktop\top_1_results_hc'

if not os.path.exists(results_folder):
    os.makedirs(results_folder)

top_n_files = glob.glob(os.path.join(top_n_tables_folder, f"top{top_n}_features_02_Healthy_VS_*.csv"))

for top_n_file in top_n_files:
    file_name = os.path.basename(top_n_file)
    label = file_name.replace(f"top{top_n}_features_02_Healthy_VS_", "").replace(".csv", "")
    print(f"\n{'=' * 60}")
    print(f"process : 02_Healthy VS {label}")
    df = pd.read_csv(top_n_file, header=None)
    feature_names = df.iloc[1:-1, 0].values

    features = df.iloc[1:-1, 1:].T
    features.columns = feature_names

    labels = df.iloc[-1, 1:]
    ids = df.iloc[0, 1:]

    binary_labels = (labels == label).astype(int)  # Healthy 0

    comparison_folder = os.path.join(results_folder, f"02_Healthy_VS_{label}")
    if not os.path.exists(comparison_folder):
        os.makedirs(comparison_folder)

    all_fold_results = []
    accuracy_list = []
    auc_list = []
    conf_matrix_list = []
    feature_importances = []
    features, binary_labels, ids = balance_dataset(features, binary_labels, ids)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_num = 0

    for train_index, test_index in kf.split(features):
        fold_num += 1

        X_train, X_test = features.iloc[train_index], features.iloc[test_index]
        y_train, y_test = binary_labels.iloc[train_index], binary_labels.iloc[test_index]
        id_train, id_test = ids.iloc[train_index], ids.iloc[test_index]

        X_train_final = X_train.to_numpy()
        X_test_final = X_test.to_numpy()
        y_train_final = y_train.to_numpy()
        y_test_final = y_test.to_numpy()

        model = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5,
                                   max_iter=10000, tol=1e-3, random_state=42)
        model.fit(X_train_final, y_train_final)

        y_pred = model.predict(X_test_final)
        y_pred_proba = model.predict_proba(X_test_final)[:, 1]  # 获取正类的预测概率

        accuracy = accuracy_score(y_test_final, y_pred)
        auc = roc_auc_score(y_test_final, y_pred_proba)
        conf_matrix = confusion_matrix(y_test_final, y_pred)

        accuracy_list.append(accuracy)
        auc_list.append(auc)
        conf_matrix_list.append(conf_matrix)

        feature_importance = model.coef_[0]
        feature_importances.append(np.abs(feature_importance))

        fold_results = pd.DataFrame({
            'Data_ID': id_test.values,
            'Fold': fold_num,
            'Actual_Label': y_test_final,
            'Predicted_Label': y_pred,
            'Predicted_Score': y_pred_proba
        })

        fold_results['Original_Label'] = fold_results['Actual_Label'].map({0: '02_Healthy', 1: label})

        fold_results_file = os.path.join(comparison_folder, f"fold_{fold_num}_predictions.csv")
        fold_results.to_csv(fold_results_file, index=False)

        all_fold_results.append(fold_results)

    if all_fold_results:
        all_results = pd.concat(all_fold_results, ignore_index=True)

        duplicate_ids = all_results[all_results.duplicated('Data_ID', keep=False)]
        if not duplicate_ids.empty:
            all_results['Unique_ID'] = all_results['Data_ID'] + '_fold' + all_results['Fold'].astype(str)
        else:
            all_results['Unique_ID'] = all_results['Data_ID']
        all_results_file = os.path.join(comparison_folder, "all_folds_predictions.csv")
        all_results.to_csv(all_results_file, index=False)


    print(f"finish : 02_Healthy VS {label}")