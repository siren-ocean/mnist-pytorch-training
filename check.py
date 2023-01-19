import cv2
import torch

import onnxruntime
from torchvision import transforms
import numpy as np
from ConvNet import ConvNet

img = cv2.imread("image/0.png")
img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
img = transforms.ToTensor()(img)

input_data = [img]
input_data = torch.stack(input_data, 0)
# print(input_data.shape)


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


# 输出onnx模型数据
ort_session = onnxruntime.InferenceSession("model/mnist.onnx")
ort_input = {ort_session.get_inputs()[0].name: to_numpy(input_data)}
onnx_out = np.array(ort_session.run(None, ort_input))
print(onnx_out[0])
print(onnx_out.shape)
print("===================================")

# 输出pytorch模型数据
model = ConvNet()
model.load_state_dict(torch.load('model/mnist.pt', map_location=torch.device('cpu')))
model.eval()
torch.no_grad()
output = model(input_data)
pytorch_output = output.detach().numpy()
print(pytorch_output)
print(pytorch_output.shape)
print("===================================")

# 模型数据对比
pytorch_output = pytorch_output.flatten()
onnx_out = onnx_out.flatten()
pytor = np.array(pytorch_output, dtype="float32")  # need to float32
onn = np.array(onnx_out, dtype="float32")  # need to float32
np.testing.assert_almost_equal(pytor, onn, decimal=5)  # 精确到小数点后5位，验证是否正确
print("The result is correct !!!")
