#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Longxiao
# Date: 2025/10/24

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.colors as mcolors

plt.rcParams['font.family'] = 'Times New Roman'

file_paths = [
    r"C:\Users\lenovo\Desktop\lr_classification with 2 class_hc\all_proteins_results_hc\02_Healthy_VS_03_Esophageal\all_folds_predictions.csv",
    r"C:\Users\lenovo\Desktop\lr_classification with 2 class_hc\all_proteins_results_hc\02_Healthy_VS_04_Gastric\all_folds_predictions.csv",
    r"C:\Users\lenovo\Desktop\lr_classification with 2 class_hc\all_proteins_results_hc\02_Healthy_VS_05_Colon\all_folds_predictions.csv",
    r"C:\Users\lenovo\Desktop\lr_classification with 2 class_hc\all_proteins_results_hc\02_Healthy_VS_06_Rectal\all_folds_predictions.csv",
    r"C:\Users\lenovo\Desktop\lr_classification with 2 class_hc\all_proteins_results_hc\02_Healthy_VS_07_Pancreatic\all_folds_predictions.csv",
    r"C:\Users\lenovo\Desktop\lr_classification with 2 class_hc\all_proteins_results_hc\02_Healthy_VS_080910_merged\all_folds_predictions.csv"
]

fig, axes = plt.subplots(1, 6, figsize=(19, 4))


def create_color_matrix(cm, diagonal_color='#4d99ca', off_diagonal_color='#e5eff9'):

    n_classes = cm.shape[0]
    color_matrix = np.zeros(cm.shape)  # 先全部填充为0 (浅蓝色)
    np.fill_diagonal(color_matrix, 1)  # 将对角线填充为1 (深蓝色)
    return color_matrix

for i, file_path in enumerate(file_paths):

    df = pd.read_csv(file_path)

    df = df.dropna(subset=['Actual_Label', 'Predicted_Label'])

    label_part = file_path.split('\\')[-2]
    labels = label_part.split('_VS_')
    print(labels)

    cm = confusion_matrix(df['Actual_Label'], df['Predicted_Label'], labels=[0, 1])

    title = labels[1]

    color_matrix = create_color_matrix(cm)

    sns.heatmap(color_matrix, annot=cm, fmt='d',
                cmap=mcolors.ListedColormap(['#e5eff9', '#4d99ca']),
                ax=axes[i], cbar=False,
                annot_kws={'size': 14, 'weight': 'bold'},
                linewidths=1, linecolor='white',
                vmin=0, vmax=1,  
                xticklabels=False, yticklabels=False) 

    axes[i].set_title(title, fontsize=12, fontweight='bold')
    axes[i].set_xticks([0.5, 1.5])
    axes[i].set_xticklabels(['Healthy', 'Cancer'])
    axes[i].set_yticks([0.5, 1.5])

    if i == 0:
        axes[i].set_yticklabels(['Healthy', 'Cancer'], rotation=0)
    else:
        axes[i].set_yticklabels([])

fig.text(0.04, 0.5, 'True Class', ha='center', va='center', rotation='vertical', fontsize=12, fontweight='bold')

fig.text(0.5, 0.05, 'Predicted Class', ha='center', va='center', fontsize=12, fontweight='bold')

plt.tight_layout(rect=[0.05, 0.1, 0.95, 0.9])

plt.savefig(r'C:\Users\lenovo\Desktop\confusion_matrices.pdf', bbox_inches='tight', dpi=300)
plt.savefig(r'C:\Users\lenovo\Desktop\confusion_matrices.png', dpi=300, bbox_inches='tight')