# -*- coding: utf-8 -*-
"""
SHAP Feature Importance Analysis Script
Adapted to MoCA screened feature set, outputs 6 types of SHAP visualization plots for training/testing/full dataset
"""
import os
import xgboost
import shap
import pandas as pd
import matplotlib.pyplot as plt

# ====================== ✅ Core Configuration Area (Only modify paths here, no changes needed for other parts) ======================
# Data file path
DATA_PATH = "/home/jiaojiao/maui/RC_Collaboratives/project/ChengduZhongyi_MR/20250722_SHAP/MoCA_6features.csv"
# Root path for saving results (subfolders will be created automatically)
SAVE_ROOT = "/home/jiaojiao/maui/RC_Collaboratives/project/ChengduZhongyi_MR/20250722_SHAP/MoCA"
# Plot parameter configuration (fixed 300 dpi high resolution, PDF vector graphics, suitable for paper publication)
PLOT_CONFIG = {
    "dpi": 300,
    "format": "pdf",
    "bbox_inches": "tight",
    "font_family": "Arial"  # Common font for academic papers, change to 'SimHei' for Chinese support
}
# Specify sample index (for single sample visualization: Waterfall Plot / Force Plot)
TRAIN_SAMPLE_IDX = 6   # The 7th sample of training set (index 6)
TEST_SAMPLE_IDX = 15   # The 16th sample of test set (index 15)

# ====================== ✅ Environment initialization and data loading ======================
def init_env_and_load_data():
    """Initialize folder directory + Load dataset + Split training/test/full dataset"""
    # Automatically create save directory to avoid path non-existence error
    if not os.path.exists(SAVE_ROOT):
        os.makedirs(SAVE_ROOT, exist_ok=True)
    # Set matplotlib backend to avoid plotting pop-up windows/errors
    plt.switch_backend('Agg')
    plt.rcParams['font.sans-serif'] = [PLOT_CONFIG["font_family"]]
    plt.rcParams['axes.unicode_minus'] = False  # Fix negative sign display issue
    
    # Load data and verify file existence
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data file does not exist: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"✅ Data loaded successfully | Data shape: {df.shape} | Feature columns: {df.drop(['Label','Group'],axis=1).columns.tolist()}")
    
    # Split dataset: Training set / Testing set / Full dataset
    df_train = df[df['Group'] == 'Train'].reset_index(drop=True)
    df_test = df[df['Group'] == 'Test'].reset_index(drop=True)
    
    # Separate features and labels
    data_dict = {
        "full": {"X": df.drop(columns=['Label', 'Group']), "y": df['Label']},
        "train": {"X": df_train.drop(columns=['Label', 'Group']), "y": df_train['Label']},
        "test": {"X": df_test.drop(columns=['Label', 'Group']), "y": df_test['Label']}
    }
    return data_dict, df_train, df_test

# ====================== ✅ Train model + Initialize SHAP explainer ======================
def train_model_and_init_shap(X_train, y_train):
    """Train XGBoost regression model + Create SHAP explainer (consistent with original model parameters)"""
    model = xgboost.XGBRegressor(random_state=42)  # Fix random seed for reproducible results
    model.fit(X_train, y_train)
    explainer = shap.Explainer(model)  # Core SHAP explainer
    tree_explainer = shap.TreeExplainer(model)  # Special explainer for Decision Plot
    print("✅ XGB model training completed | SHAP explainer initialized successfully")
    return model, explainer, tree_explainer

# ====================== ✅ Core Function: SHAP analysis for training set (output all 6 types of plots) ======================
def shap_analysis_train(X_train, explainer, tree_explainer):
    """SHAP analysis for training set: Waterfall Plot + Bar Plot + Summary Plot + Force Plot + Decision Plot + Heatmap Plot"""
    print("\n🔍 Start SHAP analysis for training set...")
    shap_values = explainer.shap_values(X_train)
    shap_values2 = explainer(X_train)  # Explanation object (adapted for bar/heatmap plots)
    exp_value = tree_explainer.expected_value
    
    # 1. Single sample Waterfall Plot
    shap.plots.waterfall(shap_values2[TRAIN_SAMPLE_IDX], show=False)
    plt.savefig(os.path.join(SAVE_ROOT, "MoCA_waterfall_train.pdf"), 
                dpi=PLOT_CONFIG["dpi"], format=PLOT_CONFIG["format"], bbox_inches=PLOT_CONFIG["bbox_inches"])
    plt.close()
    print("✅ Training set Waterfall Plot saved")
    
    # 2. Feature importance Bar Plot
    shap.plots.bar(shap_values2, show=False)
    plt.savefig(os.path.join(SAVE_ROOT, "MoCA_bar_train.pdf"), 
                dpi=PLOT_CONFIG["dpi"], format=PLOT_CONFIG["format"], bbox_inches=PLOT_CONFIG["bbox_inches"])
    plt.close()
    print("✅ Training set Bar Plot saved")
    
    # 3. Feature clustering Summary Plot
    shap.summary_plot(shap_values, X_train, show=False)
    plt.savefig(os.path.join(SAVE_ROOT, "MoCA_summary_train.pdf"), 
                dpi=PLOT_CONFIG["dpi"], format=PLOT_CONFIG["format"], bbox_inches=PLOT_CONFIG["bbox_inches"])
    plt.close()
    print("✅ Training set Summary Plot saved")
    
    # 4. Single sample Force Plot
    shap.force_plot(exp_value, shap_values[TRAIN_SAMPLE_IDX, :], X_train.iloc[TRAIN_SAMPLE_IDX, :], 
                    matplotlib=True, show=False)
    plt.savefig(os.path.join(SAVE_ROOT, "MoCA_forceplot_train.pdf"), 
                dpi=PLOT_CONFIG["dpi"], format=PLOT_CONFIG["format"], bbox_inches=PLOT_CONFIG["bbox_inches"])
    plt.close()
    print("✅ Training set Force Plot saved")
    
    # 5. Decision Path Plot
    features = X_train.iloc[range(65)]  # Take the first 65 samples, consistent with the original code
    features_display = X_train.loc[features.index]
    shap_decision_values = tree_explainer.shap_values(features)[:]
    shap.decision_plot(exp_value, shap_decision_values, features_display, show=False)
    plt.savefig(os.path.join(SAVE_ROOT, "MoCA_decision_train.pdf"), 
                dpi=PLOT_CONFIG["dpi"], format=PLOT_CONFIG["format"], bbox_inches=PLOT_CONFIG["bbox_inches"])
    plt.close()
    print("✅ Training set Decision Plot saved")
    
    # 6. SHAP Heatmap Plot
    shap.plots.heatmap(shap_values2, show=False)
    plt.savefig(os.path.join(SAVE_ROOT, "MoCA_heatmap_train.pdf"), 
                dpi=PLOT_CONFIG["dpi"], format=PLOT_CONFIG["format"], bbox_inches=PLOT_CONFIG["bbox_inches"])
    plt.close()
    print("✅ Training set Heatmap Plot saved")

# ====================== ✅ Core Function: SHAP analysis for testing set (output all 6 types of plots) ======================
def shap_analysis_test(X_test, explainer, tree_explainer):
    """SHAP analysis for test set: Waterfall Plot + Bar Plot + Summary Plot + Force Plot + Decision Plot + Heatmap Plot"""
    print("\n🔍 Start SHAP analysis for test set...")
    shap_values = explainer.shap_values(X_test)
    shap_values2 = explainer(X_test)
    exp_value = tree_explainer.expected_value
    
    # 1. Single sample Waterfall Plot
    shap.plots.waterfall(shap_values2[TEST_SAMPLE_IDX], show=False)
    plt.savefig(os.path.join(SAVE_ROOT, "MoCA_waterfall_test.pdf"),** PLOT_CONFIG)
    plt.close()
    print("✅ Test set Waterfall Plot saved")
    
    # 2. Feature importance Bar Plot
    shap.plots.bar(shap_values2, show=False)
    plt.savefig(os.path.join(SAVE_ROOT, "MoCA_bar_test.pdf"), **PLOT_CONFIG)
    plt.close()
    print("✅ Test set Bar Plot saved")
    
    # 3. Feature clustering Summary Plot
    shap.summary_plot(shap_values, X_test, show=False)
    plt.savefig(os.path.join(SAVE_ROOT, "MoCA_summary_test.pdf"),** PLOT_CONFIG)
    plt.close()
    print("✅ Test set Summary Plot saved")
    
    # 4. Single sample Force Plot
    shap.force_plot(exp_value, shap_values[TEST_SAMPLE_IDX, :], X_test.iloc[TEST_SAMPLE_IDX, :], 
                    matplotlib=True, show=False)
    plt.savefig(os.path.join(SAVE_ROOT, "MoCA_forceplot_test.pdf"), **PLOT_CONFIG)
    plt.close()
    print("✅ Test set Force Plot saved")
    
    # 5. Decision Path Plot (take the first 17 samples, consistent with the original code)
    features = X_test.iloc[range(17)]
    features_display = X_test.loc[features.index]
    shap_decision_values = tree_explainer.shap_values(features)[:]
    shap.decision_plot(exp_value, shap_decision_values, features_display, show=False)
    plt.savefig(os.path.join(SAVE_ROOT, "MoCA_decision_test.pdf"),** PLOT_CONFIG)
    plt.close()
    print("✅ Test set Decision Plot saved")
    
    # 6. SHAP Heatmap Plot
    shap.plots.heatmap(shap_values2, show=False)
    plt.savefig(os.path.join(SAVE_ROOT, "MoCA_heatmap_test.pdf"), **PLOT_CONFIG)
    plt.close()
    print("✅ Test set Heatmap Plot saved")

# ====================== ✅ Core Function: SHAP analysis for full dataset (output feature importance Bar Plot only) ======================
def shap_analysis_full(X_full, explainer):
    """SHAP analysis for full dataset: Output feature importance Bar Plot only (consistent with original code)"""
    print("\n🔍 Start SHAP analysis for full dataset...")
    shap_values2 = explainer(X_full)
    
    # Feature importance Bar Plot (full dataset)
    shap.plots.bar(shap_values2, show=False)
    plt.savefig(os.path.join(SAVE_ROOT, "MoCA_bar.pdf"),** PLOT_CONFIG)
    plt.close()
    print("✅ Full dataset Bar Plot saved")

# ====================== ✅ Main Function: One-click execution of all SHAP analysis ======================
def main():
    try:
        # 1. Initialize environment and load data
        data_dict, df_train, df_test = init_env_and_load_data()
        X_train, y_train = data_dict["train"]["X"], data_dict["train"]["y"]
        X_test, y_test = data_dict["test"]["X"], data_dict["test"]["y"]
        X_full, y_full = data_dict["full"]["X"], data_dict["full"]["y"]
        
        # 2. Train model and initialize SHAP explainer
        model, explainer, tree_explainer = train_model_and_init_shap(X_train, y_train)
        
        # 3. Execute SHAP analysis (Training set → Test set → Full dataset)
        shap_analysis_train(X_train, explainer, tree_explainer)
        shap_analysis_test(X_test, explainer, tree_explainer)
        shap_analysis_full(X_full, explainer)
        
        print("\n🎉 All SHAP analysis completed! Results saved to: ", SAVE_ROOT)
        
    except Exception as e:
        print(f"\n❌ Program running error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
