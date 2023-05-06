import torch
import torchvision.models as models
import os
import sys

from convnext import convnext_tiny

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_pth = "/code/zzg/project/seal_project/license_classification/checkpoint/convnext_1205/stitch_auto_cut_ls/convnext_1205-53-best-0.92216.pth"

model_name = "convnext"

if model_name == "resnet50":
    model = models.resnet50()
    model.fc = torch.nn.Linear(2048, 2)
    model.load_state_dict(torch.load(model_pth))   #pytorch模型加载

if model_name == "convnext":
    model = convnext_tiny()
    model.head = torch.nn.Linear(768, 2)
    model.load_state_dict(torch.load(model_pth)) 
# set the model to inference mode
model.eval()

batch_size = 1  #批处理大小
input_shape = (3, 224, 320)   #输入数据
x = torch.randn(batch_size, *input_shape)		# 生成张量

tag = os.path.basename(model_pth).replace(".pth", "")

out_dir = "onnx"
os.makedirs(out_dir, exist_ok=True)
export_onnx_file = out_dir + "/{}.onnx".format(tag)					# 目的ONNX文件名

torch.onnx.export(model,
                    x,
                    export_onnx_file,
                    opset_version=13,
                    do_constant_folding=True,	# 是否执行常量折叠优化
                    input_names=["input"],		# 输入名
                    output_names=["output"],	# 输出名
                    dynamic_axes={"input": {0:"batch_size"},  # 批处理变量
                                  "output": {0:"batch_size"}})
# x = torch.randn(batch_size,*input_shape)
# torch_out = torch.onnx._export(torch_model, x, "test.onnx", export_params=True)
print("---finished!---")