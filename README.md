# EPVS-Radiomics-ML
Python code for statistical analysis and ML modeling associated with the manuscript on EPVS radiomics, cognitive impairment, and sleep disturbance

# Radiomics & Statistical Analysis Pipeline
A complete data analysis pipeline for EPVS feature selection, machine learning interpretation, and statistical validation, including Python-based radiomics/SHAP analysis and R-based statistical analysis scripts. All analysis scripts are provided with **corresponding supporting data tables** for direct run.

## 📂 Project Overview
This repository provides an analysis workflow for EPVS research, covering feature preprocessing, feature selection, machine learning model construction and evaluation, SHAP feature interpretation, and a set of statistical validation analyses. All codes are well-commented, reproducible, and come with matching data files.

### ✅ Supported Analyses & Scripts
All executable scripts are provided with **corresponding data tables** (data files are placed in the same directory as the scripts for direct calling).

### 🐍 Python Analysis Scripts
| Script Name | Core Function |
|-------------|---------------|
| `Radiomics pipeline.py` | Complete EPVS feature preprocessing, feature selection, and modeling pipeline |
| `SHAP analysis.py` | SHAP (SHapley Additive exPlanations) feature importance analysis, visualization of feature contribution for machine learning models (supports waterfall/bar/summary/force/decision/heatmap plots) |

### 📊 R Statistical Analysis Scripts
| Script Name | Core Function |
|-------------|---------------|
| `Effect size calculation.R` | Quantitative calculation of effect size  |
| `confusion matrix.R` | Confusion matrix construction  |
| `Correlation analysis.R` | Pearson/Spearman correlation analysis between variables, correlation coefficient and significance test |
| `Partial correlation analysis.R` | Partial correlation analysis (control confounding variables to explore the direct correlation between target variables) |

### 📋 Corresponding Data Tables (One-to-One Matching)
All data files are provided in standard `.xlsx`/`.csv` format, placed in the same directory as the scripts, and can be directly read and used without additional preprocessing.
| Data File Name | Data Content & Application Scenario |
|----------------|-------------------------------------|
| `Clinical scales and EPVS features.xlsx` | Integrated dataset containing all clinical scale scores and complete EPVS features (basic dataset for all analyses) |
| `Selected features in four tasks.xlsx` | Final selected features from four classification experiments |
| `MoCA.csv` | EPVS dataset for distinguishing cognitive impairment and normal cognition, **dedicated to the demo of Radiomics pipeline.py** |
| `MoCA_6features.csv` | 6 optimal features selected from the MoCA classification experiment, **dedicated to the demo of SHAP analysis.py** |
| `Correlation analysis.csv` | Final selected features of each classification experiment, **dedicated to Correlation analysis.R** |
| `Partial correlation analysis.csv` | Special dataset for partial correlation research, **dedicated to Partial correlation analysis.R**; explore the correlations between EPVS metrics (total volume, total number) and intracranial volume, with age and gender as covariates |
