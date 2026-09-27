import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import joblib

from data_preprocessing import prepare_train_test_data, FEATURE_COLS, TARGET_COL

plt.rcParams['font.sans-serif'] = 'DejaVu Sans'

def train_and_evaluate_all():
    os.makedirs('figures', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # 1. Load preprocessed data
    X_train, X_test, X_train_s, X_test_s, y_train, y_test, scaler = prepare_train_test_data()
    
    models = {
        'Logistic Regression': {
            'model': LogisticRegression(random_state=42, max_iter=1000),
            'param_grid': {
                'C': [0.01, 0.1, 1.0, 10.0],
                'solver': ['liblinear', 'lbfgs']
            },
            'use_scaled': True
        },
        'K-Nearest Neighbors': {
            'model': KNeighborsClassifier(),
            'param_grid': {
                'n_neighbors': [3, 5, 7, 9, 11],
                'weights': ['uniform', 'distance']
            },
            'use_scaled': True
        },
        'Decision Tree': {
            'model': DecisionTreeClassifier(random_state=42),
            'param_grid': {
                'max_depth': [3, 5, 7, 10, None],
                'min_samples_split': [2, 5, 10],
                'criterion': ['gini', 'entropy']
            },
            'use_scaled': False
        },
        'Random Forest (Benchmark)': {
            'model': RandomForestClassifier(random_state=42),
            'param_grid': {
                'n_estimators': [50, 100],
                'max_depth': [5, 10, None]
            },
            'use_scaled': False
        }
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    fitted_models = {}
    y_probs = {}
    y_preds = {}
    
    print("=== Starting Model Training & Grid Search Tuning ===")
    
    for name, config in models.items():
        X_tr = X_train_s if config['use_scaled'] else X_train
        X_te = X_test_s if config['use_scaled'] else X_test
        
        grid = GridSearchCV(
            estimator=config['model'],
            param_grid=config['param_grid'],
            cv=cv,
            scoring='f1',
            n_jobs=-1
        )
        grid.fit(X_tr, y_train)
        
        best_clf = grid.best_estimator_
        fitted_models[name] = best_clf
        
        pred = best_clf.predict(X_te)
        prob = best_clf.predict_proba(X_te)[:, 1]
        
        y_preds[name] = pred
        y_probs[name] = prob
        
        acc = accuracy_score(y_test, pred)
        prec = precision_score(y_test, pred)
        rec = recall_score(y_test, pred)
        f1 = f1_score(y_test, pred)
        auc = roc_auc_score(y_test, prob)
        
        results[name] = {
            'best_params': grid.best_params_,
            'cv_best_f1': float(grid.best_score_),
            'Accuracy': float(acc),
            'Precision': float(prec),
            'Recall': float(rec),
            'F1-Score': float(f1),
            'ROC-AUC': float(auc)
        }
        
        print(f"\n--- {name} ---")
        print(f"Best Params: {grid.best_params_}")
        print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | ROC-AUC: {auc:.4f}")
    
    # Save comparison dataframe
    results_df = pd.DataFrame(results).T
    results_df.to_csv('models/model_comparison_results.csv')
    
    # 2. Select Best Foundational Model
    foundational_names = ['Logistic Regression', 'K-Nearest Neighbors', 'Decision Tree']
    best_foundational_name = max(foundational_names, key=lambda m: results[m]['F1-Score'])
    best_model = fitted_models[best_foundational_name]
    
    print(f"\n==========================================")
    print(f"SELECTED BEST FOUNDATIONAL MODEL: {best_foundational_name}")
    print(f"F1-Score: {results[best_foundational_name]['F1-Score']:.4f}")
    print(f"==========================================")
    
    joblib.dump(best_model, 'models/best_model.joblib')
    
    metadata = {
        'best_model_name': best_foundational_name,
        'best_params': results[best_foundational_name]['best_params'],
        'metrics': results[best_foundational_name],
        'requires_scaling': models[best_foundational_name]['use_scaled'],
        'feature_names': FEATURE_COLS
    }
    
    with open('models/model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=4)
        
    # Figures
    # 1. Confusion Matrices Comparison
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes = axes.flatten()
    for idx, (name, pred) in enumerate(y_preds.items()):
        cm = confusion_matrix(y_test, pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                    xticklabels=['Routine (0)', 'Threat (1)'],
                    yticklabels=['Routine (0)', 'Threat (1)'])
        axes[idx].set_title(f"{name}\nF1: {results[name]['F1-Score']:.3f} | AUC: {results[name]['ROC-AUC']:.3f}", fontsize=10, fontweight='bold')
        axes[idx].set_xlabel('Predicted Label')
        axes[idx].set_ylabel('True Label')
    plt.tight_layout()
    plt.savefig('figures/confusion_matrices.png', dpi=300)
    plt.close()
    
    # 2. ROC Curves
    plt.figure(figsize=(8, 6))
    for name, prob in y_probs.items():
        fpr, tpr, _ = roc_curve(y_test, prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {results[name]['ROC-AUC']:.3f})", linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.7)
    plt.xlabel('False Positive Rate', fontsize=11)
    plt.ylabel('True Positive Rate (Recall)', fontsize=11)
    plt.title('ROC Curves Comparison for Email Threat Classification', fontsize=13, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.tight_layout()
    plt.savefig('figures/roc_curves.png', dpi=300)
    plt.close()
    
    # 3. Feature Importance / Coefficients
    plt.figure(figsize=(9, 5))
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
        feat_df = pd.DataFrame({'Feature': FEATURE_COLS, 'Importance': importances}).sort_values('Importance', ascending=False)
        sns.barplot(x='Importance', y='Feature', data=feat_df, hue='Feature', legend=False, palette='viridis')
        plt.title(f'Feature Importances ({best_foundational_name})', fontsize=13, fontweight='bold')
    elif hasattr(best_model, 'coef_'):
        importances = np.abs(best_model.coef_[0])
        feat_df = pd.DataFrame({'Feature': FEATURE_COLS, 'Importance': importances}).sort_values('Importance', ascending=False)
        sns.barplot(x='Importance', y='Feature', data=feat_df, hue='Feature', legend=False, palette='magma')
        plt.title(f'Absolute Coefficient Weight ({best_foundational_name})', fontsize=13, fontweight='bold')
    plt.xlabel('Weight / Importance Value')
    plt.tight_layout()
    plt.savefig('figures/feature_importance.png', dpi=300)
    plt.close()
    
    # 4. Class Distribution & EDA plots
    clean_df = pd.read_csv('data/processed_email_threat_dataset.csv')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    
    sns.countplot(x='is_threat', data=clean_df, ax=axes[0], hue='is_threat', legend=False, palette=['#2ecc71', '#e74c3c'])
    axes[0].set_title('Target Class Balance (0: Routine, 1: Threat)', fontsize=11, fontweight='bold')
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(['Routine (Ham)', 'Threat (Phishing)'])
    
    sns.boxplot(x='is_threat', y='keyword_indicators', data=clean_df, ax=axes[1], hue='is_threat', legend=False, palette=['#2ecc71', '#e74c3c'])
    axes[1].set_title('Keyword Indicators by Class', fontsize=11, fontweight='bold')
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(['Routine (Ham)', 'Threat (Phishing)'])
    
    plt.tight_layout()
    plt.savefig('figures/class_distribution.png', dpi=300)
    plt.close()
    
    # 5. Correlation Matrix
    plt.figure(figsize=(9, 7))
    corr = clean_df[FEATURE_COLS + [TARGET_COL]].corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5)
    plt.title('Correlation Matrix of Email Features & Threat Status', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('figures/correlation_matrix.png', dpi=300)
    plt.close()
    
    print("\nAll evaluation figures successfully generated in figures/ directory!")
    return results_df

if __name__ == '__main__':
    train_and_evaluate_all()
