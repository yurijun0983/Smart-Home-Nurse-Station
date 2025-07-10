import pandas as pd
import socket
import struct
import time

# UDP目标IP和端口
TARGET_IP = "192.168.10.113"
TARGET_PORT = 1234

# 载入数据
df = pd.read_csv('health_dataset.csv')

# 提取8个特征列（float）
features = ['heart_rate','spo2', 'eeg1', 'eeg2', 'eeg3', 'eeg4']

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

for idx, row in df.iterrows():
    # 组装8个float
    data = [float(row[col]) for col in features]
    # 打包为二进制，8个float，格式是8个f
    msg = struct.pack('6f', *data)

    # 发送UDP包
    sock.sendto(msg, (TARGET_IP, TARGET_PORT))
    print(f"[Send] 样本{idx+1} 数据已发送: {data}")

    time.sleep(1)  # 每秒发送一条

sock.close()
