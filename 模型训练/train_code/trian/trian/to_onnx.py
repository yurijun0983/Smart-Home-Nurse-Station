import tf2onnx
import tensorflow as tf

# ✅ 加载模型
model = tf.keras.models.load_model("heart_disease_model.h5")

# ✅ 转换为ONNX，固定输入shape [1, 2]
spec = (tf.TensorSpec((1, 2), tf.float32, name="input"),)

# ✅ 转换
onnx_model, _ = tf2onnx.convert.from_keras(
    model,
    input_signature=spec,
    opset=13,  # 推荐13或更高
    output_path="heart_disease_model.onnx"
)

print("✅ 已保存为heart_disease_model.onnx")
