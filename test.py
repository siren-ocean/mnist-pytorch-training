import os
import torch
from PIL import Image
from ConvNet import ConvNet
import matplotlib.pyplot as plt
from torchvision import transforms

path = './image/'
images = []
labels = []

for name in sorted(os.listdir(path)):
    img = Image.open(path + name).convert('L')
    img = transforms.ToTensor()(img)
    images.append(img)
    labels.append(int(name[0]))
images = torch.stack(images, 0)

# %% 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvNet()
model.load_state_dict(torch.load('model/mnist.pt', device))
model.eval()

# %% 测试模型
with torch.no_grad():
    output = model(images)

# %% 打印结果
pred = output.argmax(1)
true = torch.LongTensor(labels)
print(pred)
print(true)

# %% 绘制
plt.figure(figsize=(10, 4))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.title(f'pred {pred[i]} | true {true[i]}')
    plt.axis('off')
    plt.imshow(images[i].squeeze(0), cmap='gray')
plt.show()
