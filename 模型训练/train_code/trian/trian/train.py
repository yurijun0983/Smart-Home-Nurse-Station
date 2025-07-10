import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# 加载数据
df = pd.read_csv('health_dataset.csv')

# ✅ 心脏病输入（只用心率和血氧）
X_heart = df[['heart_rate', 'spo2']].values
y_heart = df['heart_disease'].values

# EEG输入（4维，用于睡眠质量和癫痫）
X_eeg = df[['eeg1', 'eeg2', 'eeg3', 'eeg4']].values
y_sleep = to_categorical(df['sleep_quality'].values, num_classes=3)
y_epi = df['epilepsy'].values

# ✅ 睡眠质量模型
model_sleep = Sequential([
    Dense(64, activation='relu', input_shape=(4,)),
    Dense(32, activation='relu'),
    Dense(3, activation='softmax')
])
model_sleep.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model_sleep.fit(X_eeg, y_sleep, epochs=20, batch_size=64, validation_split=0.1)
model_sleep.save('sleep_quality_model.h5')

# ✅ 心脏病模型（2维输入）
model_heart = Sequential([
    Dense(64, activation='relu', input_shape=(2,)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
model_heart.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_heart.fit(X_heart, y_heart, epochs=20, batch_size=64, validation_split=0.1)
model_heart.save('heart_disease_model.h5')

# ✅ 癫痫模型
model_epi = Sequential([
    Dense(64, activation='relu', input_shape=(4,)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
model_epi.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_epi.fit(X_eeg, y_epi, epochs=20, batch_size=64, validation_split=0.1)
model_epi.save('epilepsy_model.h5')

print("✅ 无标准化模型（心脏病用心率+血氧）训练完成")
