import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# 载入数据
df = pd.read_csv('health_dataset.csv')

# 载入模型
model_heart = load_model('heart_disease_model.h5')
model_sleep = load_model('sleep_quality_model.h5')
model_epi = load_model('epilepsy_model.h5')

# 准备输入特征
X_heart = df[['heart_rate', 'systolic', 'diastolic', 'spo2']].values
X_eeg = df[['eeg1', 'eeg2', 'eeg3', 'eeg4']].values

# 归一化（同训练时用的Scaler，最好保存并加载；这里简单重新fit仅作演示）
scaler_heart = StandardScaler()
X_heart_scaled = scaler_heart.fit_transform(X_heart)

scaler_eeg = StandardScaler()
X_eeg_scaled = scaler_eeg.fit_transform(X_eeg)

# 预测
heart_preds = model_heart.predict(X_heart_scaled)
sleep_preds = model_sleep.predict(X_eeg_scaled)
epi_preds = model_epi.predict(X_eeg_scaled)

# 对心脏病和癫痫用0.5阈值二分类判定
heart_results = ['有风险' if p >= 0.5 else '无风险' for p in heart_preds.flatten()]
epi_results = ['有风险' if p >= 0.5 else '无风险' for p in epi_preds.flatten()]

# 睡眠质量是3分类，取概率最大类别
sleep_results = np.argmax(sleep_preds, axis=1)
sleep_map = {0: '好', 1: '中', 2: '差'}
sleep_results = [sleep_map[c] for c in sleep_results]

# 打印结果
for i in range(len(df)):
    print(f"样本{i+1}：❤️ 心脏病：{heart_results[i]} | ⚡ 癫痫：{epi_results[i]} | 💤 睡眠质量：{sleep_results[i]}")

