#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Longxiao
# Date: 2025/10/29
import pandas as pd
import os

importance_folder = r"C:\Users\lenovo\Desktop\lr_classification with 2 class_hc"
data_file = r"C:\Users\lenovo\Desktop\dat_total_vol_table_norm_v8.csv"

cancer_types = ['03_Esophageal', '04_Gastric', '05_Colon', '06_Rectal', '07_Pancreatic', '080910_merged']

data_df = pd.read_csv(data_file)
flitered_importance = 0.50

for cancer in cancer_types:
    importance_file = os.path.join(importance_folder, f"feature_importance_02_Healthy_VS_{cancer}.csv")

    if not os.path.exists(importance_file):
        print(f"The file does not exist : {importance_file}")
        continue

    print(f"\nprocess : {cancer}")

    filtered_genes = data_df[
        (data_df['Cluster'] == cancer) &
        (data_df['point_type'] == '2_right_sig')
        ]['GeneID'].unique()

    print(f"GeneID : {len(filtered_genes)}")

    importance_df = pd.read_csv(importance_file)
    print(f"Number of original features : {len(importance_df)}")

    max_importance = importance_df['Importance'].max()
    importance_df['relative_importance'] = (importance_df['Importance'] / max_importance) #* 100

    # Screening Criteria:
    # 1. The content of the feature column is among the GeneIDs filtered for the current cancer type
    # 2. Relative importance is greater than 50%
    filtered_importance = importance_df[
        (importance_df['Feature'].isin(filtered_genes)) &
        (importance_df['relative_importance'] > flitered_importance)
        ]

    print(f"Number of features meeting the criteria: {len(filtered_importance)}")

    if len(filtered_importance) > 0:

        output_file = os.path.join(importance_folder, f"filtered_{flitered_importance}_importance_02_Healthy_VS_{cancer}.csv")
        filtered_importance.to_csv(output_file, index=False)
        print(f"save : {output_file}")

        for _, row in filtered_importance.iterrows():
            print(f"  {row['Feature']}: {row['relative_importance']:.2f}")
    else:
        print("No features meeting the criteria were found")

print("\nFinish！")