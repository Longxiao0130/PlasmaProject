#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Longxiao
# Date: 2025/10/24

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

plt.rcParams['font.family'] = 'Times New Roman'

file_paths = [
    r"C:\Users\lenovo\Desktop\lr_classification with 2 class_hc\all_proteins_results_hc\02_Healthy_VS_03_Esophageal\all_folds_predictions.csv",
    r"C:\Users\lenovo\Desktop\lr_classification with 2 class_hc\all_proteins_results_hc\02_Healthy_VS_04_Gastric\all_folds_predictions.csv",
    r"C:\Users\lenovo\Desktop\lr_classification with 2 class_hc\all_proteins_results_hc\02_Healthy_VS_05_Colon\all_folds_predictions.csv",
    r"C:\Users\lenovo\Desktop\lr_classification with 2 class_hc\all_proteins_results_hc\02_Healthy_VS_06_Rectal\all_folds_predictions.csv",
    r"C:\Users\lenovo\Desktop\lr_classification with 2 class_hc\all_proteins_results_hc\02_Healthy_VS_07_Pancreatic\all_folds_predictions.csv",
    r"C:\Users\lenovo\Desktop\lr_classification with 2 class_hc\all_proteins_results_hc\02_Healthy_VS_080910_merged\all_folds_predictions.csv"
]

colors = ["#E64B35FF", "#4DBBD5FF", "#00A087FF", "#3C5488FF", "#F39B7FFF", "#8491B4FF"]

fig, axes = plt.subplots(1, 6, figsize=(20, 4), sharey=True)

for i, file_path in enumerate(file_paths):
    df = pd.read_csv(file_path)

    file_name = file_path.split('\\')[-2]
    parts2 = file_name.split('_VS_')
    title = parts2[1]

    palette = {0: 'lightgray', 1: colors[i]}

    sns.boxplot(
        x='Actual_Label',
        y='Predicted_Score',
        hue='Actual_Label',  # 添加 hue 参数
        data=df,
        ax=axes[i],
        palette=palette,
        fliersize=0,  # 关闭异常值的绘制
        linewidth=1,  # 箱线的宽度
        width=0.3,  # 箱型图的宽度
        legend=False  # 不显示图例
    )

    sns.stripplot(
        x='Actual_Label',
        y='Predicted_Score',
        hue='Actual_Label',  
        data=df,
        ax=axes[i],
        jitter=True,  
        size=5,  
        palette=palette,  
        edgecolor='black',  
        linewidth=0.5,
        legend=False  
    )

    axes[i].set_title(title, fontsize=12, fontweight='bold')

    axes[i].set_xticks([0, 1])  
    axes[i].set_xticklabels(['Healthy', 'Cancer']) 

    axes[i].set_xlabel('')  
    axes[i].set_ylabel('') 

    if i != 0:
        axes[i].set_ylabel('')

fig.text(0.5, 0.05, 'True Class', ha='center', va='center', fontsize=12, fontweight='bold')
fig.text(0.04, 0.5, 'Probability cancer', ha='center', va='center', rotation='vertical', fontsize=12, fontweight='bold')

plt.tight_layout(rect=[0.05, 0.1, 0.95, 0.9])

plt.savefig(r'C:\Users\lenovo\Desktop\boxplots.pdf', bbox_inches='tight', dpi=300)
plt.savefig(r'C:\Users\lenovo\Desktop\boxplots.png', dpi=300, bbox_inches='tight')
