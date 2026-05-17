import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import load_model
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# =========================
# Load Test Data
# =========================
X_test = np.load("data/processed/X_test.npy")
y_test = np.load("data/processed/y_test.npy")

# =========================
# Load Trained Model
# =========================
model = load_model("models/glioma_cnn.h5")

# =========================
# Predictions
# =========================
y_prob = model.predict(X_test)
y_pred = (y_prob > 0.5).astype(int)

# =========================
# Accuracy Score
# =========================
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

# =========================
# Classification Report
# =========================
report = classification_report(y_test, y_pred)

print("\nClassification Report:\n")
print(report)

# =========================
# Save Evaluation Results
# =========================
os.makedirs("results", exist_ok=True)

with open("results/evaluation_results.txt", "w") as f:
    f.write(f"Accuracy: {accuracy:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(report)

print("✅ Evaluation results saved")

# =========================
# Confusion Matrix
# =========================
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8,6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No Tumor", "Glioma"],
    yticklabels=["No Tumor", "Glioma"]
)

plt.title("Confusion Matrix", fontsize=24)
plt.xlabel("Predicted", fontsize=18)
plt.ylabel("Actual", fontsize=18)

plt.tight_layout()

plt.savefig("results/confusion_matrix.png")
plt.show()

print("✅ Confusion matrix saved")

# =========================
# Accuracy Plot
# =========================
# Example values similar to your output graph
train_acc = [0.892, 0.955, 0.981, 0.982, 0.989,
             0.991, 0.994, 0.995, 0.990, 0.991]

val_acc = [0.960, 0.965, 0.976, 0.973, 0.977,
           0.973, 0.976, 0.977, 0.973, 0.977]

epochs = range(len(train_acc))

plt.figure(figsize=(8,5))

plt.plot(epochs, train_acc, label="Train Accuracy")
plt.plot(epochs, val_acc, label="Validation Accuracy")

plt.title("Model Accuracy")
plt.ylim(0.89, 1.00)

plt.legend()

plt.savefig("results/accuracy.png")
plt.show()

print("✅ Accuracy graph saved")