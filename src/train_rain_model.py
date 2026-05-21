import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_DIR / "outputs"
MPL_CACHE_DIR = PROJECT_DIR / ".matplotlib-cache"
MPL_CACHE_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = PROJECT_DIR / "data" / "processed" / "weather_project_dataset.csv"
OUTPUT_DIR.mkdir(exist_ok=True)

def prepare_data(df: pd.DataFrame):
    """Create model features and the rain-tomorrow target."""
    # 1. Basic preprocessing
    df["date"] = pd.to_datetime(df["date"])
    
    # 2. Feature engineering: extract month and temperature range
    df["month"] = df["date"].dt.month
    df["temp_range"] = df["max_temp"] - df["min_temp"]
    
    # 3. Select features
    features = ["min_temp", "max_temp", "temp_range", "rainfall", "sunshine", "rain_today", "month"]
    X = df[features]
    y = df["rain_tomorrow"]
    
    return X, y

def main() -> None:
    # Load and prepare data
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing processed dataset: {DATA_PATH}. "
            "Run `python src/merge_weather_data.py` first."
        )

    df = pd.read_csv(DATA_PATH)
    X, y = prepare_data(df)
    visualize_eda_plots(df)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # --- Data Scaling ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- 1. Logistic Regression ---
    lr_model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    lr_model.fit(X_train_scaled, y_train)
    lr_pred = lr_model.predict(X_test_scaled)
    lr_accuracy = accuracy_score(y_test, lr_pred)

    # --- 2. Random Forest (Current Champion) ---
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_pred)

    print("-" * 50)
    print(f"Logistic Regression Accuracy: {lr_accuracy:.4f}")
    print(f"Random Forest Accuracy:       {rf_accuracy:.4f}")
    print("-" * 50)

    # Select the overall best model
    if rf_accuracy > lr_accuracy:
        best_model = rf_model
        best_name = "Random Forest"
        best_X_test = X_test
        best_pred = rf_pred
    else:
        best_model = lr_model
        best_name = "Logistic Regression"
        best_X_test = X_test_scaled
        best_pred = lr_pred
    
    print(f"Best Model selected: {best_name}")
    print(classification_report(y_test, best_pred))

    # --- Professional Diagnostic Visualizations (Optimized Top 3) ---
    visualize_diagnostic_plots(best_model, best_name, best_X_test, y_test, X.columns)

    # --- Sample prediction (Restored & Optimized) ---
    sample_data = {
        "min_temp": 14.0, "max_temp": 24.0, "temp_range": 10.0,
        "rainfall": 0.0, "sunshine": 30.0, "rain_today": 0, "month": 4
    }
    sample_df = pd.DataFrame([sample_data])
    sample_input = sample_df if best_name == "Random Forest" else scaler.transform(sample_df)
    
    prediction = best_model.predict(sample_input)[0]
    probability = best_model.predict_proba(sample_input)[0][1]

    print("\nSample prediction (April data):")
    print(f"Will it rain tomorrow? {'Yes' if prediction == 1 else 'No'}")
    print(f"Probability of rain tomorrow: {probability:.4f}")

################## data visualization ##################

def visualize_diagnostic_plots(model, model_name, X_test, y_test, feature_names) -> None:
    """Generates the 'Top 3' essential diagnostic plots."""
    print("Generating essential diagnostic plots...")
    
    # 1. Confusion Matrix (Critical for seeing errors)
    plt.figure(figsize=(8, 6))
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=["No Rain", "Rain"], 
                yticklabels=["No Rain", "Rain"])
    plt.title(f"Confusion Matrix ({model_name})")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png")
    plt.close()

    # 2. ROC Curve (Standard for overall quality)
    plt.figure(figsize=(8, 6))
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic ({model_name})')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "roc_curve.png")
    plt.close()

    # 3. Feature Importance (Why the model works)
    plt.figure(figsize=(10, 6))
    if hasattr(model, 'coef_'): importance = model.coef_[0]
    elif hasattr(model, 'feature_importances_'): importance = model.feature_importances_
    
    sns.barplot(x=importance, y=feature_names, hue=feature_names, palette="viridis", legend=False)
    plt.title(f"Feature Importance ({model_name})")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_importance.png")
    plt.close()
    
    print("Essential plots saved: Confusion Matrix, ROC Curve, Feature Importance.")


def visualize_eda_plots(df: pd.DataFrame) -> None:
    """Generate simple exploratory charts for the dataset."""
    print("Generating exploratory data plots...")

    plt.figure(figsize=(7, 5))
    sns.countplot(x="rain_tomorrow", data=df, hue="rain_tomorrow", palette="Blues", legend=False)
    plt.xticks([0, 1], ["No Rain", "Rain"])
    plt.title("Rain Tomorrow Distribution")
    plt.xlabel("Rain Tomorrow")
    plt.ylabel("Daily Records")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "rain_distribution.png")
    plt.close()

    corr_df = df.copy()
    corr_df["date"] = pd.to_datetime(corr_df["date"])
    corr_df["month"] = corr_df["date"].dt.month
    corr_df["temp_range"] = corr_df["max_temp"] - corr_df["min_temp"]
    numeric_columns = [
        "min_temp",
        "max_temp",
        "temp_range",
        "rainfall",
        "sunshine",
        "rain_today",
        "month",
        "rain_tomorrow",
    ]

    plt.figure(figsize=(9, 7))
    sns.heatmap(corr_df[numeric_columns].corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Weather Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "correlation_heatmap.png")
    plt.close()

    print("Exploratory plots saved: Rain Distribution, Correlation Heatmap.")

if __name__ == "__main__":
    main()
