import nncase

def convert_onnx_to_kmodel(onnx_path, kmodel_path, input_shape):
    opts = nncase.CompileOptions()
    opts.input_shape = [input_shape]         # 需要是二维列表，例如 [[1,2]]
    opts.input_types = ["float32"]            # 一般都是 float32
    opts.target = "k230"

    compiler = nncase.Compiler(opts)

    with open(onnx_path, "rb") as f:
        onnx_data = f.read()
        compiler.import_onnx(onnx_data, nncase.ImportOptions())
        compiler.compile()

    with open(kmodel_path, "wb") as f:
        f.write(compiler.gencode_tobytes())
    print(f"已保存 {kmodel_path}")

if __name__ == "__main__":
    convert_onnx_to_kmodel("heart_disease_model.onnx", "heart_disease_model.kmodel", [1, 2])
    convert_onnx_to_kmodel("epilepsy_model.onnx", "epilepsy_model.kmodel", [1, 4])
    convert_onnx_to_kmodel("sleep_quality_model.onnx", "sleep_quality_model.kmodel", [1, 6])
