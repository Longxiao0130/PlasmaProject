# !/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Longxiao
# Date: 2025/10/23
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix


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

    X_balanced = balanced_data.iloc[:, :-2]
    y_balanced = balanced_data.iloc[:, -2]
    id_balanced = balanced_data['ID']

    return X_balanced, y_balanced, id_balanced


df = pd.read_csv(r'C:\Users\lenovo\Desktop\data with label.csv', header=0, index_col=0)
features = df.iloc[1:-1, :].T
labels = df.iloc[-1, :]
ids = df.iloc[0, :]

results_folder = r'C:\Users\lenovo\Desktop\lr_classification with 2 class_hc'
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

labels_to_compare = ['03_Esophageal', '04_Gastric', '05_Colon', '06_Rectal', '07_Pancreatic', '080910_merged']

for label in labels_to_compare:
    last_row = df.iloc[-1]
    column_mask = last_row.isin(['02_Healthy', label])
    filtered_df = df.loc[:, column_mask]

    filtered_features = filtered_df.iloc[1:-1, :]
    # print('filtered_features',filtered_features)
    filtered_labels = filtered_df.iloc[-1, :]
    # print('filtered_labels',filtered_labels)
    filtered_ids = filtered_df.columns
    # print('filtered_ids',filtered_ids)

    binary_labels = (filtered_labels == label).astype(int)  # cancer 1,Healthy 0
    filtered_features = filtered_features.T

    comparison_folder = os.path.join(results_folder, f"02_Healthy_VS_{label}")
    if not os.path.exists(comparison_folder):
        os.makedirs(comparison_folder)

    all_fold_results = []
    accuracy_list = []
    # auc_list = []
    conf_matrix_list = []
    feature_importances = []
    filtered_features, binary_labels, filtered_ids = balance_dataset(filtered_features, binary_labels, filtered_ids)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_num = 0

    for train_index, test_index in kf.split(filtered_features):
        fold_num += 1

        X_train, X_test = filtered_features.iloc[train_index], filtered_features.iloc[test_index]
        y_train, y_test = binary_labels.iloc[train_index], binary_labels.iloc[test_index]
        id_train, id_test = filtered_ids.iloc[train_index], filtered_ids.iloc[test_index]

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
        # auc_list.append(auc)
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

    avg_feature_importance = np.mean(feature_importances, axis=0)
    feature_importance_df = pd.DataFrame({
        'Feature': filtered_features.columns,
        'Original_Index': [features.columns.get_loc(col) + 3 for col in filtered_features.columns],
        'Importance': avg_feature_importance
    }).sort_values(by='Importance', ascending=False)

    feature_importance_file = os.path.join(results_folder, f"feature_importance_02_Healthy_VS_{label}.csv")
    feature_importance_df.to_csv(feature_importance_file, index=False)

print("Finish！")