# =========================
# Imports
# =========================
import os
import random
import shutil
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# =========================
# Setup
# =========================
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)

# Paths
BASE_DIR = r"C:\Users\USER\Downloads\dog vs cat\dogvscat"
INPUT_DIR = os.path.join(BASE_DIR, "train")
WORK_DIR = os.path.join(BASE_DIR, "working")
SAMPLE_DIR = os.path.join(WORK_DIR, "sample")

# Parameters
IMG_HEIGHT, IMG_WIDTH = 150, 150
BATCH_SIZE = 32
EPOCHS = 5

# =========================
# Data Preparation
# =========================
def prepare_data(src, size=2000):
    folders = ['train/cats', 'train/dogs', 'validation/cats', 'validation/dogs']
    for folder in folders:
        os.makedirs(os.path.join(SAMPLE_DIR, folder), exist_ok=True)

    cats = os.listdir(os.path.join(src, 'cats'))
    dogs = os.listdir(os.path.join(src, 'dogs'))
    random.shuffle(cats)
    random.shuffle(dogs)

    train_n, val_n = int(size * 0.4), int(size * 0.1)

    for i in range(train_n):
        shutil.copy(os.path.join(src, 'cats', cats[i]), os.path.join(SAMPLE_DIR, 'train/cats'))
        shutil.copy(os.path.join(src, 'dogs', dogs[i]), os.path.join(SAMPLE_DIR, 'train/dogs'))

    for i in range(val_n):
        shutil.copy(os.path.join(src, 'cats', cats[train_n+i]), os.path.join(SAMPLE_DIR, 'validation/cats'))
        shutil.copy(os.path.join(src, 'dogs', dogs[train_n+i]), os.path.join(SAMPLE_DIR, 'validation/dogs'))

if not os.path.exists(SAMPLE_DIR):
    prepare_data(INPUT_DIR)

# Data generators
train_aug = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.15,
    horizontal_flip=True
)


val_aug = ImageDataGenerator(rescale=1./255)


train_gen = train_aug.flow_from_directory(
    os.path.join(SAMPLE_DIR, 'train'),
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

val_gen = val_aug.flow_from_directory(
    os.path.join(SAMPLE_DIR, 'validation'),
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

# =========================
# Model 1: CNN
# =========================
cnn = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

callbacks = [
    EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, min_lr=1e-4)
]

history = cnn.fit(
    train_gen,
    steps_per_epoch=train_gen.samples // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=val_gen,
    validation_steps=val_gen.samples // BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

y_true = val_gen.classes
y_pred = (cnn.predict(val_gen) > 0.5).astype("int32")
cnn_acc = accuracy_score(y_true, y_pred)

# =========================
# Model 2: Logistic Regression
# =========================
X_train, y_train = next(train_aug.flow_from_directory(
    os.path.join(SAMPLE_DIR, 'train'),
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=train_gen.samples,
    class_mode='binary',
    shuffle=False
))
    
X_val, y_val = next(val_aug.flow_from_directory(
    os.path.join(SAMPLE_DIR, 'validation'),
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=val_gen.samples,
    class_mode='binary',
    shuffle=False
))


X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_val_flat = X_val.reshape(X_val.shape[0], -1)

log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_flat, y_train)
log_acc = log_reg.score(X_val_flat, y_val)

# =========================
# Model 3: Decision Tree
# =========================
dt = DecisionTreeClassifier(max_depth=10)
dt.fit(X_train_flat, y_train)
dt_acc = dt.score(X_val_flat, y_val)

# =========================
# Visualization
# =========================
def plot_bar(models, accuracies, title):
    plt.figure(figsize=(7, 5))
    plt.bar(models, accuracies, color=["blue", "green", "orange"])
    plt.title(title)
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 1)
    for i, v in enumerate(accuracies):
        plt.text(i, v + 0.02, f"{v*100:.0f}%", ha='center', fontsize=10, fontweight='bold')
    plt.gca().set_yticks([i/10 for i in range(0, 11)])
    plt.gca().set_yticklabels([f"{i*10}%" for i in range(0, 11)])
    plt.show()

# CNN Training Progress
plt.figure(figsize=(8, 5))
plt.plot(history.history['accuracy'], label="Train Accuracy")
plt.plot(history.history['val_accuracy'], label="Validation Accuracy")
plt.title("CNN Training Progress")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# Logistic Regression
plot_bar(["Logistic Regression"], [log_acc], "Logistic Regression Accuracy")

# Decision Tree
plot_bar(["Decision Tree"], [dt_acc], "Decision Tree Accuracy")

# Comparison of All Models
plot_bar(["CNN", "Logistic Regression", "Decision Tree"],
         [cnn_acc, log_acc, dt_acc],
         "Comparison of All Models (Dog vs Cat)")
