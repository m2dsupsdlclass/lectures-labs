import torch
import torch.nn.functional as F
import torch.nn as nn
from matplotlib.cm import get_cmap
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets
from torchvision.transforms import transforms

import matplotlib.pyplot as plt

english_labels = ["T-shirt/top",
                  "Trouser",
                  "Pullover",
                  "Dress",
                  "Coat",
                  "Sandal",
                  "Shirt",
                  "Sneaker",
                  "Bag",
                  "Ankle boot"]

cuda = False
batch_size = 32
lr = 0.01
momentum = 0.9
log_interval = 10
epochs = 6

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

train_data = datasets.FashionMNIST('data', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                   ]))
train_loader = DataLoader(train_data, batch_size=128, shuffle=False, **kwargs)

# Lets's compute the average mean and std of the train images. We will
# use them for normalizing data later on.
n_samples_seen = 0.
mean = 0
std = 0
for train_batch, train_target in train_loader:
    batch_size = train_batch.shape[0]
    train_batch = train_batch.view(batch_size, -1)
    this_mean = torch.mean(train_batch, dim=1)
    this_std = torch.sqrt(
        torch.mean((train_batch - this_mean[:, None]) ** 2, dim=1))
    mean += torch.sum(this_mean, dim=0)
    std += torch.sum(this_std, dim=0)
    n_samples_seen += batch_size

mean /= n_samples_seen
std /= n_samples_seen

train_data = datasets.FashionMNIST('data', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean=mean,
                                                            std=std)]))

test_data = datasets.FashionMNIST('data', train=False, download=True,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=mean,
                                                           std=std)]))

train_loader = DataLoader(train_data, batch_size=32, shuffle=True,
                          **kwargs)

test_loader = DataLoader(test_data, batch_size=32, shuffle=False,
                         **kwargs)


class VGGCell(nn.Module):
    def __init__(self, in_channel, out_channel, depth, max_pooling=True):
        super(VGGCell, self).__init__()
        self.convs = nn.ModuleList()
        for i in range(depth):
            if i == 0:
                self.convs.append(nn.Conv2d(in_channel, out_channel,
                                            kernel_size=(3, 3),
                                            padding=1))
            else:
                self.convs.append(nn.Conv2d(out_channel, out_channel,
                                            kernel_size=(3, 3),
                                            padding=1))
        self.max_pooling = max_pooling

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        if self.max_pooling:
            x = F.max_pool2d(x, kernel_size=(2, 2))
        return x


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        vgg1 = VGGCell(1, 32, 1, max_pooling=True)
        vgg2 = VGGCell(32, 64, 1, max_pooling=True)
        self.vggs = nn.ModuleList([vgg1, vgg2])
        self.dropout_2d = nn.Dropout2d(p=0.25)
        self.fc1 = nn.Linear(7 * 7 * 64, 256)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        for vgg in self.vggs:
            x = self.dropout_2d(vgg(x))
        x = x.view(-1, 7 * 7 * 64)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# Let's test our model on CPU
model = Model()
img, target = train_data[0]
# n_channel, width, height
print(img.shape)

fig, ax = plt.subplots(1, 1)
ax.imshow(img[0].numpy(), cmap=get_cmap('gray'))
plt.show()

# First dimension should contain batch_size
img = img[None, :]
img = Variable(img)
pred = model(img)
print(target, english_labels[target])
print(pred)

if cuda:
    model.cuda()

optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                       T_max=5,
                                                       last_epoch=-1)


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if cuda:
            data, target = data.cuda(), target.cuda()
        batch_size = data.shape[0]
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0] * batch_size

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch + 1, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))
    train_loss /= len(test_loader.dataset)
    return train_loss


def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0]
        # sum up batch loss
        _, pred = output.data.max(dim=1)
        # get the index of the max log-probability
        correct += torch.sum(pred == target.data.long())

    test_loss /= len(test_loader.dataset)
    test_accuracy = correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f},'
          ' Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * test_accuracy))
    return test_loss, test_accuracy


logs = {'epoch': [], 'train_loss': [], 'test_loss': [],
        'test_accuracy': [], 'lr': []}

for epoch in range(epochs):
    train_loss = train(epoch)
    test_loss, test_accuracy = test()
    logs['epoch'].append(epoch)
    logs['train_loss'].append(train_loss)
    logs['test_loss'].append(test_loss)
    logs['test_accuracy'].append(test_accuracy)
    logs['lr'].append(optimizer.param_groups[0]['lr'])
    scheduler.step(epoch)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 12))
ax1.plot(logs['epoch'], list(zip(logs['train_loss'],
                                 logs['test_loss'],
                                 logs['test_accuracy'])))
ax1.legend(['Train loss', 'Test loss', 'Test accuracy'])
ax2.plot(logs['epoch'], logs['lr'],
         label='Learning rate')
ax2.legend()

# Let's see what our model can do
test_img, true_target = test_data[42]

fig, ax = plt.subplots(1, 1)
ax.imshow(test_img[0].numpy(), cmap=get_cmap('gray'))
plt.show()

test_img = test_img[None, :]
if cuda:
    test_img = test_img.cuda()
test_img = Variable(test_img, volatile=True)
pred = model(test_img)
_, target = torch.max(pred, dim=1)
target = target.data[0]
print(english_labels[target])
