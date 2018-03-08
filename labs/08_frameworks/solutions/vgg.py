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

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        vgg1 = VGGCell(1, 32, 2, max_pooling=True)
        vgg2 = VGGCell(32, 64, 3, max_pooling=False)
        vgg3 = VGGCell(64, 128, 3, max_pooling=True)
        vgg4 = VGGCell(128, 256, 3, max_pooling=False)
        self.vggs = nn.ModuleList([vgg1, vgg2, vgg3, vgg4])
        self.dropout_2d = nn.Dropout2d(p=0.25)
        self.fc1 = nn.Linear(7 * 7 * 256, 256)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        for vgg in self.vggs:
            x = self.dropout_2d(vgg(x))
        x = x.view(-1, 7 * 7 * 256)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def reset_parameters(self):
        for vgg in self.vggs:
            vgg.reset_parameters()
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()