import numpy as np
import pandas as pd

np.random.seed(42)
N = 10000

# 特征生成（同之前说明）
heart_rate = np.clip(np.random.normal(75, 10, N), 50, 110)
systolic = np.clip(np.random.normal(120, 15, N), 90, 160)
diastolic = np.clip(np.random.normal(80, 8, N), 50, 100)
spo2 = np.clip(np.random.normal(98, 1.5, N), 85, 100)
eeg = np.clip(np.random.normal(50, 15, (N, 4)), 0, 100)

# 插入异常值（可略）
num_outliers = int(0.01 * N)
outlier_idx = np.random.choice(N, num_outliers, replace=False)
for i in outlier_idx:
    choice = np.random.choice(['hr','sys','dia','spo2','eeg'])
    if choice == 'hr':
        heart_rate[i] = np.random.uniform(30, 50)
    elif choice == 'sys':
        systolic[i] = np.random.uniform(160, 200)
    elif choice == 'dia':
        diastolic[i] = np.random.uniform(100, 120)
    elif choice == 'spo2':
        spo2[i] = np.random.uniform(80, 90)
    elif choice == 'eeg':
        ch = np.random.randint(0,4)
        eeg[i, ch] = np.random.uniform(100, 150)

# 标签构造
heart_disease = ((systolic > 130) | (diastolic > 85) | (spo2 < 96)).astype(int)

epilepsy = np.zeros(N, dtype=int)
epi_idx = np.random.choice(N, int(0.05*N), replace=False)
epilepsy[epi_idx] = 1
for i in epi_idx:
    ch = np.random.randint(0,4)
    eeg[i, ch] = np.random.uniform(80, 100)

sleep_quality = np.zeros(N, dtype=int)
for i in range(N):
    score = 0
    if heart_rate[i] > 85: score += 1
    if (systolic[i] > 130) or (diastolic[i] > 85): score += 1
    if spo2[i] < 96: score += 1
    if eeg[i].max() > 80: score += 1
    if score == 0:
        sleep_quality[i] = 2
    elif score <= 2:
        sleep_quality[i] = 1
    else:
        sleep_quality[i] = 0

# 合成DataFrame
df = pd.DataFrame({
    'heart_rate': heart_rate,
    'systolic': systolic,
    'diastolic': diastolic,
    'spo2': spo2,
    'eeg1': eeg[:,0],
    'eeg2': eeg[:,1],
    'eeg3': eeg[:,2],
    'eeg4': eeg[:,3],
    'sleep_quality': sleep_quality,
    'heart_disease': heart_disease,
    'epilepsy': epilepsy
})

# 保存CSV
df.to_csv('health_dataset.csv', index=False)
print("CSV数据集生成完毕！")
