import os
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from model_architecture import build_model

X_train = np.load("data/processed/X_train.npy")
y_train = np.load("data/processed/y_train.npy")
X_test = np.load("data/processed/X_test.npy")
y_test = np.load("data/processed/y_test.npy")

model = build_model()
model.compile(
    optimizer=Adam(),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

os.makedirs("models", exist_ok=True)

checkpoint = ModelCheckpoint(
    "models/glioma_cnn.h5",
    monitor="val_accuracy",
    save_best_only=True
)

model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=10,
    batch_size=16,
    callbacks=[checkpoint]
)

print("✅ Model trained and saved")
