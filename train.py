import torch
import torch.nn as nn
from torchvision import datasets, transforms

from ConvNet import ConvNet


BATCH_SIZE = 512 if torch.cuda.is_available() else 12
EPOCHS = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=BATCH_SIZE, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=BATCH_SIZE, shuffle=True)

model = ConvNet().to(device)
optimizer = torch.optim.Adam(model.parameters())
lossf = nn.CrossEntropyLoss()


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = lossf(output, target)
        loss.backward()
        optimizer.step()
        if (batch_idx + 1) % 30 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch + 1, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def save_model(model, device, test_loader, epoch):
    model.eval()
    test_loss = 0
    correct = 0

    if epoch == 0:
        global max_acc
        max_acc = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += lossf(output, target)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_acc = correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    if test_acc > max_acc:
        torch.save(model.state_dict(), 'model/mnist.pt')
        max_acc = test_acc


def transform2onnx():
    model.load_state_dict(torch.load("model/mnist.pt"))
    dummy_input = torch.randn(1, 1, 28, 28).to(device)
    input_names = ["input_0"]
    output_names = ["output_0"]
    torch.onnx.export(model, dummy_input, 'model/mnist.onnx', verbose=True, input_names=input_names,
                      output_names=output_names)


for epoch in range(EPOCHS):
    train(model, device, train_loader, optimizer, epoch)
    save_model(model, device, test_loader, epoch)

transform2onnx()
