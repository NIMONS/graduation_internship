import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# -------------------------------
# 1️⃣ Chuẩn bị dữ liệu
# -------------------------------
train_dir = "dataset/train"
test_dir = "dataset/test"

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

num_classes = len(train_generator.class_indices)
print(f"Số lớp phân loại: {num_classes}")

# -------------------------------
# 2️⃣ Xây dựng mô hình MobileNetV3
# -------------------------------
base_model = MobileNetV3Large(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# Đóng băng các lớp của mô hình gốc
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x)  # Dropout cao hơn một chút để tránh overfitting
predictions = Dense(num_classes, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=predictions)

# -------------------------------
# 3️⃣ Biên dịch mô hình
# -------------------------------
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -------------------------------
# 4️⃣ Huấn luyện mô hình
# -------------------------------
EPOCHS = 20

history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=EPOCHS
)

# -------------------------------
# 5️⃣ Lưu mô hình
# -------------------------------
os.makedirs("models", exist_ok=True)
model.save("models/fruit_freshness_mobilenetv3.h5")
print("✅ Đã lưu mô hình tại: models/fruit_freshness_mobilenetv3.h5")

# -------------------------------
# 6️⃣ Vẽ biểu đồ Accuracy và Loss
# -------------------------------
acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(acc, label="Train Accuracy")
plt.plot(val_acc, label="Validation Accuracy")
plt.legend()
plt.title("Độ chính xác (Accuracy)")

plt.subplot(1, 2, 2)
plt.plot(loss, label="Train Loss")
plt.plot(val_loss, label="Validation Loss")
plt.legend()
plt.title("Hàm mất mát (Loss)")

plt.tight_layout()
plt.show()

# -------------------------------
# 7️⃣ Đánh giá mô hình
# -------------------------------
test_loss, test_acc = model.evaluate(test_generator)
print(f"🎯 Độ chính xác trên tập kiểm thử: {test_acc:.2f}")
