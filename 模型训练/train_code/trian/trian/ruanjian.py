import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# ✅ 模拟训练数据
# 假设心率 60-100，血氧 90-100
heart_rate = np.random.uniform(60, 100, 1000)
spo2 = np.random.uniform(90, 100, 1000)
X = np.stack([heart_rate, spo2], axis=1)

# 简单的标签规则（你可以换成更复杂的）
y = ((heart_rate > 85) & (spo2 < 95)).astype(np.float32)  # 心率高且血氧低 → 有风险

# ✅ 创建简单的神经网络模型
model = keras.Sequential([
    layers.Input(shape=(2,)),
    layers.Dense(8, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X, y, epochs=50, batch_size=16)

# ✅ 保存为 H5
model.save('heart_disease_model.h5')
print("✅ 模型已保存为 heart_disease_model.h5")
