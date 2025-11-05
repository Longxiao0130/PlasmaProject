#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Longxiao
# Date: 2025/10/24
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve, auc
import os

plt.rcParams['font.family'] = 'Times New Roman'

file_paths = [
    [
        r"C:\Users\lenovo\Desktop\lr_classification with 2 class_hc\all_proteins_results_hc\02_Healthy_VS_03_Esophageal\all_folds_predictions.csv",
        r"C:\Users\lenovo\Desktop\lr_classification with 2 class_hc\top_10_results_hc\02_Healthy_VS_03_Esophageal\all_folds_predictions.csv",
        r"C:\Users\lenovo\Desktop\lr_classification with 2 class_hc\top_3_results_hc\02_Healthy_VS_03_Esophageal\all_folds_predictions.csv",
        r"C:\Users\lenovo\Desktop\lr_classification with 2 class_hc\top_1_results_hc\02_Healthy_VS_03_Esophageal\all_folds_predictions.csv",
        r"C:\Users\lenovo\Desktop\lr_classification with 2 class_hc\filtered_0.5_results_hc\02_Healthy_VS_03_Esophageal\all_folds_predictions.csv",
    ],
    [
        r"C:\Users\lenovo\Desktop\lr_classification with 2 class_hc\all_proteins_results_hc\02_Healthy_VS_04_Gastric\all_folds_predictions.csv",
        r"C:\Users\lenovo\Desktop\lr_classification with 2 class_hc\top_10_results_hc\02_Healthy_VS_04_Gastric\all_folds_predictions.csv",
        r"C:\Users\lenovo\Desktop\lr_classification with 2 class_hc\top_3_results_hc\02_Healthy_VS_04_Gastric\all_folds_predictions.csv",
        r"C:\Users\lenovo\Desktop\lr_classification with 2 class_hc\top_1_results_hc\02_Healthy_VS_04_Gastric\all_folds_predictions.csv",
        r"C:\Users\lenovo\Desktop\lr_classification with 2 class_hc\filtered_0.5_results_hc\02_Healthy_VS_04_Gastric\all_folds_predictions.csv",
    ],
    [
        r"C:\Users\lenovo\Desktop\lr_classification with 2 class_hc\all_proteins_results_hc\02_Healthy_VS_05_Colon\all_folds_predictions.csv",
        r"C:\Users\lenovo\Desktop\lr_classification with 2 class_hc\top_10_results_hc\02_Healthy_VS_05_Colon\all_folds_predictions.csv",
        r"C:\Users\lenovo\Desktop\lr_classification with 2 class_hc\top_3_results_hc\02_Healthy_VS_05_Colon\all_folds_predictions.csv",
        r"C:\Users\lenovo\Desktop\lr_classification with 2 class_hc\top_1_results_hc\02_Healthy_VS_05_Colon\all_folds_predictions.csv",
        r"C:\Users\lenovo\Desktop\lr_classification with 2 class_hc\filtered_0.5_results_hc\02_Healthy_VS_05_Colon\all_folds_predictions.csv",
    ],
    [
        r"C:\Users\lenovo\Desktop\lr_classification with 2 class_hc\all_proteins_results_hc\02_Healthy_VS_06_Rectal\all_folds_predictions.csv",
        r"C:\Users\lenovo\Desktop\lr_classification with 2 class_hc\top_10_results_hc\02_Healthy_VS_06_Rectal\all_folds_predictions.csv",
        r"C:\Users\lenovo\Desktop\lr_classification with 2 class_hc\top_3_results_hc\02_Healthy_VS_06_Rectal\all_folds_predictions.csv",
        r"C:\Users\lenovo\Desktop\lr_classification with 2 class_hc\top_1_results_hc\02_Healthy_VS_06_Rectal\all_folds_predictions.csv",
        r"C:\Users\lenovo\Desktop\lr_classification with 2 class_hc\filtered_0.5_results_hc\02_Healthy_VS_06_Rectal\all_folds_predictions.csv",
    ],
    [
        r"C:\Users\lenovo\Desktop\lr_classification with 2 class_hc\all_proteins_results_hc\02_Healthy_VS_07_Pancreatic\all_folds_predictions.csv",
        r"C:\Users\lenovo\Desktop\lr_classification with 2 class_hc\top_10_results_hc\02_Healthy_VS_07_Pancreatic\all_folds_predictions.csv",
        r"C:\Users\lenovo\Desktop\lr_classification with 2 class_hc\top_3_results_hc\02_Healthy_VS_07_Pancreatic\all_folds_predictions.csv",
        r"C:\Users\lenovo\Desktop\lr_classification with 2 class_hc\top_1_results_hc\02_Healthy_VS_07_Pancreatic\all_folds_predictions.csv",
        r"C:\Users\lenovo\Desktop\lr_classification with 2 class_hc\filtered_0.5_results_hc\02_Healthy_VS_07_Pancreatic\all_folds_predictions.csv",
    ],
    [
        r"C:\Users\lenovo\Desktop\lr_classification with 2 class_hc\all_proteins_results_hc\02_Healthy_VS_080910_merged\all_folds_predictions.csv",
        r"C:\Users\lenovo\Desktop\lr_classification with 2 class_hc\top_10_results_hc\02_Healthy_VS_080910_merged\all_folds_predictions.csv",
        r"C:\Users\lenovo\Desktop\lr_classification with 2 class_hc\top_3_results_hc\02_Healthy_VS_080910_merged\all_folds_predictions.csv",
        r"C:\Users\lenovo\Desktop\lr_classification with 2 class_hc\top_1_results_hc\02_Healthy_VS_080910_merged\all_folds_predictions.csv",
        r"C:\Users\lenovo\Desktop\lr_classification with 2 class_hc\filtered_0.5_results_hc\02_Healthy_VS_080910_merged\all_folds_predictions.csv",
    ]
]

fig, axes = plt.subplots(1, 6, figsize=(24, 4))
colors = ["#87CEFA" , "#35a435", "#ff8d28", "#2a7db7","#d83031" ,"#9370DB", "#ff8d28"]
labels = ["All Proteins","Top 10", "Top 3", "Top 1","protein panel"]


for i, paths in enumerate(file_paths):
    for j, file_path in enumerate(paths):

        df = pd.read_csv(file_path)

        label_part = file_path.split('\\')[-2]
        labelss = label_part.split('_VS_')
        print(labelss)
        # y_true = df['Actual_Label'].apply(lambda x: 1 if x != '02_Healthy' else 0)
        y_true = df['Actual_Label']
        y_scores = df['Predicted_Score']

        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        axes[i].plot(fpr, tpr, color=colors[j], lw=2, label=f'{labels[j]}(AUC = {roc_auc:.3f})')   #
        axes[i].plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
        axes[i].set_xlim([0.0, 1.0])
        axes[i].set_ylim([0.0, 1.05])
        axes[i].set_xlabel('1−Specificity')
        axes[i].set_ylabel('Sensitivity')
        # axes[i].set_title(file_path.split('\\')[-3].split('_vs_')[1], fontsize=12, fontweight='bold')
        axes[i].set_title(file_path.split('\\')[-2].split('_vs_')[0], fontsize=12, fontweight='bold')
        axes[i].legend(loc="lower right")

        # 隐藏子图的边框和纵轴刻度线
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['right'].set_visible(False)
        if i == 0:
            axes[i].spines['left'].set_position(('outward', 4))
        else:
            axes[i].spines['left'].set_visible(False)

        axes[i].tick_params(axis='y', which='both', length=0)

fig.text(0.5, 0.05, '1-Specificity', ha='center', va='center', fontsize=12, fontweight='bold')
fig.text(0.08, 0.5, 'Sensitivity', ha='center', va='center', rotation='vertical', fontsize=12, fontweight='bold')

plt.tight_layout(rect=[0.1, 0.1, 0.9, 0.9])

plt.savefig(r'C:\Users\lenovo\Desktop\co_curves.pdf')
plt.savefig(r'C:\Users\lenovo\Desktop\co_curves.png', dpi=300)
