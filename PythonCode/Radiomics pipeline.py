import os
import configparser
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.metrics import (
    roc_auc_score, accuracy_score, confusion_matrix,
    recall_score, precision_score, f1_score
)
# Replace skrebate with mrmr library for feature selection
from mrmr import mrmr_classif
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import DataConversionWarning

# ====================== Configuration Reader  ======================
class ExperimentConfig:
    """Experiment configuration reading class, decouples configuration with reference to reader_config.py"""
    def __init__(self, config_path=None):
        self.cf = configparser.ConfigParser()        
        self.default_config = {
            'data': {'test_size': 0.2, 'random_state': 42},
            'feature_selection': {'n_features': 6},
            'model': {
                # General parameters for cross-validation
                'cv_split': 6,
                # Gaussian Process parameters
                'gp_length_scale': 1.0,
                'gp_n_restarts_optimizer': 10,
                # Decision Tree parameters
                'dt_criterion': 'gini',
                'dt_splitter': 'best',
                'dt_max_depth': 2,
                'dt_min_samples_split': 2,
                'dt_min_samples_leaf': 1,
                # Logistic Regression parameters
                'lr_penalty': 'l2',
                'lr_C': 1.0,
                'lr_tol': 1e-4,
                'lr_solver': 'liblinear',
                'lr_max_iter': 1000,
                # Random Forest parameters
                'rf_criterion': 'gini',
                'rf_n_estimators': 100,
                'rf_max_depth': 2,
                'rf_min_samples_split': 2,
                'rf_min_samples_leaf': 1
            }
        }
        if config_path and os.path.exists(config_path):
            self.cf.read(config_path, encoding='utf-8')

    def get(self, section, key, default=None):
        """Safely read configuration items, return default value if configuration is missing"""
        try:
            return self.cf.get(section, key)
        except (configparser.NoSectionError, configparser.NoOptionError):
            return self.default_config.get(section, {}).get(key, default)

    def getfloat(self, section, key, default=None):
        return float(self.get(section, key, default))

    def getint(self, section, key, default=None):
        return int(self.get(section, key, default))

    def getbool(self, section, key, default=None):
        return self.cf.getboolean(section, key, fallback=default)

# ====================== Data Loading and Splitting (refined parameters + exception handling) ======================
def load_and_split_data(csv_path, config: ExperimentConfig):
    """
    Load MoCA.csv and split into training/testing sets by configured ratio
    :param csv_path: Path to MoCA.csv file
    :param config: Instance of experiment configuration
    :return: X_train, X_test, y_train, y_test, feature_names
    :raises: FileNotFoundError, RuntimeError, ValueError
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"MoCA.csv not found at path: {csv_path}")
    if csv_path.split('.')[-1] != 'csv':
        raise RuntimeError(f"File format error: {csv_path} must be CSV file")
    
    try:
        df = pd.read_csv(csv_path, encoding='utf-8')
    except Exception as e:
        raise RuntimeError(f"Read CSV failed: {str(e)}")
    
    if df.empty:
        raise ValueError("Input CSV is empty, no valid data")
    if df.shape[1] < 2:
        raise ValueError("CSV must contain at least label column + 1 feature column")
    
    # Separate features and labels (maintain DataFrame format for mrmr_classif compatibility)
    X = df.iloc[:, 1:]
    y = df.iloc[:, 0].values
    feature_names = df.columns[1:].tolist()
    
    if len(np.unique(y)) < 2:
        raise ValueError("Label column only contains single class, can't split train/test")
    
    test_size = config.getfloat('data', 'test_size', 0.2)
    random_state = config.getint('data', 'random_state', 42)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y,
        shuffle=True
    )
    
    if X_train.shape[0] < 8 or X_test.shape[0] < 8:
        warnings.warn(f"Sample size too small: train={X_train.shape[0]}, test={X_test.shape[0]}", UserWarning)
    
    return X_train, X_test, y_train, y_test, feature_names

# ====================== Z-score Normalization (refined preprocessing logic) ======================
def z_score_normalization(X_train, X_test):
    """
    Perform Z-score normalization on training set and apply the same rules to testing set,
    fill missing values to avoid runtime errors
    :param X_train: Training set features (DataFrame)
    :param X_test: Test set features (DataFrame)
    :return: X_train_norm, X_test_norm, scaler
    :raises: RuntimeError
    """
    try:
        scaler = StandardScaler()
        # Fill missing values with training set mean (no data leakage) to avoid normalization errors
        X_train_filled = X_train.fillna(X_train.mean())
        X_test_filled = X_test.fillna(X_train.mean())
        
        X_train_norm = pd.DataFrame(
            scaler.fit_transform(X_train_filled),
            columns=X_train.columns,
            index=X_train.index
        )
        X_test_norm = pd.DataFrame(
            scaler.transform(X_test_filled),
            columns=X_test.columns,
            index=X_test.index
        )
        
        if X_train_norm.isnull().any().any():
            raise RuntimeError("Normalization generated NaN values")
        
        return X_train_norm, X_test_norm, scaler
    except Exception as e:
        raise RuntimeError(f"Z-score normalization failed: {str(e)}")

# ====================== mRMR Feature Selection (refined parameters + result verification) ======================
def mrmr_feature_selection(X_train_norm, y_train, config: ExperimentConfig):
    """
    Select Top-N features based on mrmr_classif
    :param X_train_norm: Normalized training set features (DataFrame)
    :param y_train: Training set labels (1D array)
    :param config: Instance of experiment configuration
    :return: X_train_selected, selected_indices, selected_feature_names
    :raises: ValueError
    """
    n_features = config.getint('feature_selection', 'n_features', 6)
    
    if n_features < 1 or n_features > X_train_norm.shape[1]:
        raise ValueError(f"n_features={n_features} out of range (1~{X_train_norm.shape[1]})")
    
    try:
        selected_feature_names = mrmr_classif(
            X=X_train_norm, 
            y=y_train, 
            K=n_features,
            show_progress=False
        )
    except Exception as e:
        raise RuntimeError(f"mRMR feature selection failed: {str(e)}")
    
    selected_indices = [X_train_norm.columns.get_loc(col) for col in selected_feature_names]
    X_train_selected = X_train_norm[selected_feature_names]
    
    if X_train_selected.empty:
        raise ValueError("mRMR selection returned empty feature set")
    
    return X_train_selected, selected_indices, selected_feature_names

# ====================== ✅ Core Update: Model Training (integrated GridSearch hyperparameter tuning + full parameter alignment) ======================
def train_models(X_train_selected, y_train, config: ExperimentConfig):
    """
    Train the specified 4 classifiers
    :param X_train_selected: Training set after feature selection (DataFrame)
    :param y_train: Training set labels (1D array)
    :param config: Instance of experiment configuration
    :return: Dictionary of trained models (key=model name, value=optimized trained model)
    """
    # ✅ Unified extraction of configuration parameters, 1:1 alignment with ML_model base class default values
    random_state = config.getint('data', 'random_state', 42)
    cv_split = config.getint('model', 'cv_split', 6)
    
    # Gaussian Process parameters
    gp_n_restarts = config.getint('model', 'gp_n_restarts_optimizer', 10)
    # Decision Tree parameters
    dt_criterion = config.get('model', 'dt_criterion', 'gini')
    dt_splitter = config.get('model', 'dt_splitter', 'best')
    # Logistic Regression parameters
    lr_penalty = config.get('model', 'lr_penalty', 'l2')
    lr_C = config.getfloat('model', 'lr_C', 1.0)
    lr_tol = config.getfloat('model', 'lr_tol', 1e-4)
    lr_solver = config.get('model', 'lr_solver', 'liblinear')
    lr_max_iter = config.getint('model', 'lr_max_iter', 1000)
    # Random Forest parameters
    rf_criterion = config.get('model', 'rf_criterion', 'gini')

    # ✅ Define base configuration + hyperparameter grid for 4 models, fully replicate base class code design
    models_dict = {
        # Gaussian Process Classifier | align with kernel list + hyperparameter range
        "GaussianProcess": {
            "base_model": GaussianProcessClassifier(
                random_state=random_state,
                n_restarts_optimizer=gp_n_restarts
            ),
            "param_grid": {
                "kernel": [C(1.0, (1e-5, 1e5)) * RBF(ls, (1e-5, 1e5)) for ls in np.linspace(1, 20, 20)],
                "n_restarts_optimizer": np.linspace(1, 30, 30).astype(int)
            }
        },
        # Decision Tree Classifier | align with all parameters + hyperparameter list
        "DecisionTree": {
            "base_model": DecisionTreeClassifier(
                random_state=random_state,
                class_weight='balanced'
            ),
            "param_grid": {
                "criterion": ['gini', 'entropy'],
                "splitter": ['best', 'random'],
                "max_depth": np.linspace(2, 32, 30).astype(int),
                "min_samples_split": np.linspace(2, 32, 30).astype(int),
                "min_samples_leaf": np.linspace(2, 12, 10).astype(int)
            }
        },
        # Logistic Regression Classifier | align with regularization + hyperparameter range
        "LogisticRegression": {
            "base_model": LogisticRegression(
                max_iter=lr_max_iter,
                random_state=random_state,
                class_weight='balanced',
                solver=lr_solver
            ),
            "param_grid": {
                "penalty": ['l1', 'l2'],
                "C": np.logspace(-2, 10, 12),
                "tol": np.linspace(1e-5, 1e-3, 100)
            }
        },
        # Random Forest Classifier | align with tree parameters + hyperparameter range
        "RandomForest": {
            "base_model": RandomForestClassifier(
                random_state=random_state,
                class_weight='balanced',
                n_jobs=-1
            ),
            "param_grid": {
                "criterion": ['gini', 'entropy'],
                "n_estimators": np.linspace(10, 1010, 1000).astype(int),
                "max_depth": np.linspace(2, 32, 30).astype(int),
                "min_samples_split": np.linspace(2, 32, 30).astype(int),
                "min_samples_leaf": np.linspace(2, 12, 10).astype(int)
            }
        }
    }

    # ✅ Core logic for model training + hyperparameter tuning with precise exception capture
    trained_models = {}
    print(f"\n📊 Start model training with GridSearchCV (cv={cv_split}, scoring=roc_auc)...")
    for model_name, model_info in models_dict.items():
        try:
            grid_search = GridSearchCV(
                estimator=model_info["base_model"],
                param_grid=model_info["param_grid"],
                scoring='roc_auc',
                cv=cv_split,
                n_jobs=-1,
                verbose=0,
                error_score='raise'
            )
            grid_search.fit(X_train_selected, y_train)
            best_model = grid_search.best_estimator_
            trained_models[model_name] = best_model
            
            # Print training logs for experiment review
            print(f"✅ {model_name} | Best CV AUC: {grid_search.best_score_:.4f}")
            print(f"   Best Params: {grid_search.best_params_}\n")

        except Exception as e:
            raise RuntimeError(f"❌ Train {model_name} failed: {str(e)}") from e

    return trained_models

# ====================== Model Evaluation ==================================================
def evaluate_models(trained_models, X_test_selected, y_test):
    """
    Evaluate models on test set, calculate full metrics including AUC/Accuracy/Sensitivity/Specificity,
    robust to zero-division errors
    :param trained_models: Dictionary of trained models
    :param X_test_selected: Test set after feature selection
    :param y_test: Test set labels
    :return: Evaluation result dictionary (including confusion matrix)
    """
    metrics = {}
    for name, model in trained_models.items():
        try:
            y_pred = model.predict(X_test_selected)
            y_pred_proba = model.predict_proba(X_test_selected)[:, 1] if hasattr(model, "predict_proba") else None
            
            cm = confusion_matrix(y_test, y_pred)
            if cm.shape != (2, 2):
                raise RuntimeError(f"Confusion matrix shape error: {cm.shape} (expected 2x2)")
            tn, fp, fn, tp = cm.ravel()
            
            # Core metric calculation with zero-division protection
            auc = roc_auc_score(y_test, y_pred_proba) if (y_pred_proba is not None and len(np.unique(y_test)) == 2) else "N/A"
            accuracy = accuracy_score(y_test, y_pred)
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            precision = precision_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            metrics[name] = {
                "AUC": auc,
                "Accuracy": accuracy,
                "Sensitivity": sensitivity,
                "Specificity": specificity,
                "Precision": precision,
                "F1-Score": f1,
                "ConfusionMatrix": {"TN": tn, "FP": fp, "FN": fn, "TP": tp}
            }
        except Exception as e:
            metrics[name] = {"Error": f"Evaluate failed: {str(e)}"}
    
    return metrics

# ====================== Result Saving (serialization + structured storage) ======================
def save_results(evaluation_results, selected_features, save_path="./experiment_results"):
    """
    Save feature selection results + model evaluation results to CSV files, create directory automatically
    :param evaluation_results: Evaluation result dictionary
    :param selected_features: List of selected features
    :param save_path: Directory for result saving
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    
    # Save selected features
    feat_df = pd.DataFrame({"Selected_Features": selected_features})
    feat_df.to_csv(os.path.join(save_path, "selected_features.csv"), index=False, encoding='utf-8')
    
    # Save model evaluation results
    metrics_list = []
    for model_name, metric_dict in evaluation_results.items():
        if "Error" in metric_dict:
            metrics_list.append({"Model": model_name, "Error": metric_dict["Error"]})
        else:
            metrics_list.append({
                "Model": model_name,
                "AUC": metric_dict["AUC"],
                "Accuracy": metric_dict["Accuracy"],
                "Sensitivity": metric_dict["Sensitivity"],
                "Specificity": metric_dict["Specificity"],
                "Precision": metric_dict["Precision"],
                "F1-Score": metric_dict["F1-Score"],
                "TN": metric_dict["ConfusionMatrix"]["TN"],
                "FP": metric_dict["ConfusionMatrix"]["FP"],
                "FN": metric_dict["ConfusionMatrix"]["FN"],
                "TP": metric_dict["ConfusionMatrix"]["TP"]
            })
    metrics_df = pd.DataFrame(metrics_list)
    metrics_df.to_csv(os.path.join(save_path, "model_evaluation.csv"), index=False, encoding='utf-8')
    
    print(f"\n📁 All results saved to directory: {save_path}")

# ====================== Main Process (end-to-end engineering encapsulation) ======================
def main(csv_path, config_path=None):
    """Main function for the entire experiment process, one-click execution of 
    data processing → feature selection → model training → evaluation → saving"""
    config = ExperimentConfig(config_path)
    
    try:
        # 1. Data loading and splitting
        print("="*60)
        print("📌 Step 1: Loading and splitting dataset")
        X_train, X_test, y_train, y_test, feature_names = load_and_split_data(csv_path, config)
        print(f"✅ Data split done | Train: {X_train.shape}, Test: {X_test.shape}")
        
        # 2. Z-score normalization
        print("\n" + "="*60)
        print("📌 Step 2: Z-score normalization (fill missing values)")
        X_train_norm, X_test_norm, scaler = z_score_normalization(X_train, X_test)
        print("✅ Normalization completed")
        
        # 3. mRMR feature selection
        print("\n" + "="*60)
        print("📌 Step 3: mRMR feature selection (Top-N)")
        X_train_selected, selected_indices, selected_feature_names = mrmr_feature_selection(
            X_train_norm, y_train, config
        )
        X_test_selected = X_test_norm.iloc[:, selected_indices]
        print(f"✅ Selected {len(selected_feature_names)} features: {selected_feature_names}")
        
        # 4. Model training (including GridSearch hyperparameter tuning)
        print("\n" + "="*60)
        print("📌 Step 4: Model training with hyperparameter optimization")
        trained_models = train_models(X_train_selected, y_train, config)
        print("✅ All models training completed")
        
        # 5. Model evaluation
        print("\n" + "="*60)
        print("📌 Step 5: Model evaluation on test set")
        evaluation_results = evaluate_models(trained_models, X_test_selected, y_test)
        print("✅ Evaluation completed")
        
        # 6. Formatted output of evaluation results
        print("\n" + "="*60)
        print("📊 Final Evaluation Results (Test Set)")
        print("="*60)
        for model_name, metric_dict in evaluation_results.items():
            print(f"\n【{model_name}】")
            if "Error" in metric_dict:
                print(f"  ❌ {metric_dict['Error']}")
            else:
                for k, v in metric_dict.items():
                    if k != "ConfusionMatrix":
                        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
                cm = metric_dict["ConfusionMatrix"]
                print(f"  Confusion Matrix: TN={cm['TN']}, FP={cm['FP']}, FN={cm['FN']}, TP={cm['TP']}")
        
        # 7. Save experiment results
        print("\n" + "="*60)
        print("📌 Step 6: Saving experiment results")
        save_results(evaluation_results, selected_feature_names)
        
        return evaluation_results
    
    except Exception as e:
        raise RuntimeError(f"\n❌ Experiment failed: {str(e)}")

# ====================== Program Entry ======================
if __name__ == "__main__":
    # ✅ Replace with the actual path of your MoCA.csv
    CSV_PATH = "/home/jiaojiao/maui/RC_Collaboratives/project/ChengduZhongyi_MR/20240207_PVSandScales/MoCA.csv"
    # Optional: configuration file path, use default parameters if set to None
    CONFIG_PATH = None  # Example: CONFIG_PATH = "experiment_config.ini"
    
    try:
        main(CSV_PATH, CONFIG_PATH)
    except Exception as e:
        print(f"\n❌ Program exit with error: {str(e)}")
        exit(1)
