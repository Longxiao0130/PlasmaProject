#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Longxiao
# Date: 2025/10/23
import pandas as pd
import os
import glob

feature_importance_folder = r'C:\Users\lenovo\Desktop\lr_classification with 2 class_hc'
data_file = r'C:\Users\lenovo\Desktop\data with label.csv'
output_folder = r'C:\Users\lenovo\Desktop\top_1_features_classification_tables_hc'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

top_n = 1

data_with_label = pd.read_csv(data_file, header=None)
feature_files = glob.glob(os.path.join(feature_importance_folder, "feature_importance_02_Healthy_VS_*.csv"))

statistics = []

for feature_file in feature_files:
    file_name = os.path.basename(feature_file)
    label = file_name.replace("feature_importance_02_Healthy_VS_", "").replace(".csv", "")
    print(f"\n{'=' * 60}")
    print(f"process : 02_Healthy VS {label}")

    importance_df = pd.read_csv(feature_file)
    top_features = importance_df.nlargest(top_n, 'Importance')
    top_feature_names = top_features['Feature'].tolist()

    feature_col = data_with_label.iloc[:, 0].astype(str).str.strip()
    matched_info = []
    for feature in top_feature_names:
        match = feature_col[feature_col == feature]
        if not match.empty:
            matched_info.append((match.index[0], feature))

    matched_count = len(matched_info)

    # 筛选样本列（02_Healthy和当前标签）
    labels_row = data_with_label.iloc[-1, :].astype(str).str.strip()
    healthy_mask = labels_row == '02_Healthy'
    label_mask = labels_row == label
    selected_columns_mask = healthy_mask | label_mask

    selected_cols = selected_columns_mask[selected_columns_mask].index.tolist()

    healthy_samples = sum(healthy_mask)
    label_samples = sum(label_mask)

    print(f"02_Healthy({healthy_samples}), {label}({label_samples})")

    if matched_count == 0:
        print("No matching features")
        continue

    final_data = []

    header_row = ['Feature'] + data_with_label.iloc[0, selected_cols].tolist()
    final_data.append(header_row)

    for row_idx, feature_name in matched_info:
        feature_row = [feature_name] + data_with_label.iloc[row_idx, selected_cols].tolist()
        final_data.append(feature_row)

    label_row = ['Label'] + data_with_label.iloc[-1, selected_cols].tolist()
    final_data.append(label_row)

    new_table = pd.DataFrame(final_data)

    output_path = os.path.join(output_folder, f"top{top_n}_features_02_Healthy_VS_{label}.csv")
    new_table.to_csv(output_path, index=False, header=False)

print(f"\nsave: {output_folder}")