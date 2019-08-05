import torch
import torch.nn as nn

class G(nn.Module):
    def __init__(self, n_channel_input, n_channel_output, n_filters):
        super(G, self).__init__()
        self.conv1 = nn.Conv2d(n_channel_input, n_filters, 4, 2, 1)
        self.conv2 = nn.Conv2d(n_filters, n_filters * 2, 4, 2, 1)
        self.conv3 = nn.Conv2d(n_filters * 2, n_filters * 4, 4, 2, 1)
        self.conv4 = nn.Conv2d(n_filters * 4, n_filters * 8, 4, 2, 1)
        self.conv5 = nn.Conv2d(n_filters * 8, n_filters * 8, 4, 2, 1)
        self.conv6 = nn.Conv2d(n_filters * 8, n_filters * 8, 4, 2, 1)

        self.deconv1 = nn.ConvTranspose2d(n_filters * 8, n_filters * 8, 4, 2, 1)
        self.deconv2 = nn.ConvTranspose2d(n_filters * 8 * 2, n_filters * 8, 4, 2, 1)
        self.deconv3 = nn.ConvTranspose2d(n_filters * 8 * 2, n_filters * 4, 4, 2, 1)
        self.deconv4 = nn.ConvTranspose2d(n_filters * 4 * 2, n_filters * 2, 4, 2, 1)
        self.deconv5 = nn.ConvTranspose2d(n_filters * 2 * 2, n_filters, 4, 2, 1)
        self.deconv6 = nn.ConvTranspose2d(n_filters * 2, n_channel_output, 4, 2, 1)

        self.batch_norm = nn.BatchNorm2d(n_filters)
        self.batch_norm2 = nn.BatchNorm2d(n_filters * 2)
        self.batch_norm4 = nn.BatchNorm2d(n_filters * 4)
        self.batch_norm8 = nn.BatchNorm2d(n_filters * 8)
        
        self.leaky_relu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)

        self.dropout = nn.Dropout(0.2)
        self.tanh = nn.Tanh()

    def forward(self, x):
        encoder1 = self.conv1(x)
        encoder2 = self.batch_norm2(self.conv2(self.leaky_relu(encoder1)))
        encoder3 = self.batch_norm4(self.conv3(self.leaky_relu(encoder2)))
        encoder4 = self.batch_norm8(self.conv4(self.leaky_relu(encoder3)))
        encoder5 = self.batch_norm8(self.conv5(self.leaky_relu(encoder4)))
        encoder6 = self.conv6(self.leaky_relu(encoder5))

        decoder1 = self.dropout(self.batch_norm8(self.deconv1(self.relu(encoder6))))
        decoder1 = torch.cat((decoder1, encoder5), 1)
        
        decoder2 = self.dropout(self.batch_norm8(self.deconv2(self.relu(decoder1))))
        decoder2 = torch.cat((decoder2, encoder4), 1)
        
        decoder3 = self.dropout(self.batch_norm4(self.deconv3(self.relu(decoder2))))
        decoder3 = torch.cat((decoder3, encoder3), 1)
        
        decoder4 = self.batch_norm2(self.deconv4(self.relu(decoder3)))
        decoder4 = torch.cat((decoder4, encoder2), 1)

        decoder5 = self.batch_norm(self.deconv5(self.relu(decoder4)))
        decoder5 = torch.cat((decoder5, encoder1), 1)
        
        decoder6 = self.deconv6(self.relu(decoder5))
        output = self.tanh(decoder6)
        return output

class D(nn.Module):
    def __init__(self, n_channel_input, n_channel_output, n_filters):
        super(D, self).__init__()
        self.conv1 = nn.Conv2d(n_channel_input + n_channel_output, n_filters, 4, 2, 1)
        self.conv2 = nn.Conv2d(n_filters, n_filters * 2, 4, 2, 1)
        self.conv3 = nn.Conv2d(n_filters * 2, n_filters * 4, 4, 2, 1)
        self.conv4 = nn.Conv2d(n_filters * 4, n_filters * 8, 4, 1, 1)
        self.conv5 = nn.Conv2d(n_filters * 8, n_filters * 16, 4, 1, 1)
        self.conv6 = nn.Conv2d(n_filters * 16, 1, 4, 1, 1)

        self.batch_norm2 = nn.BatchNorm2d(n_filters * 2)
        self.batch_norm4 = nn.BatchNorm2d(n_filters * 4)
        self.batch_norm8 = nn.BatchNorm2d(n_filters * 8)
        self.batch_norm16 = nn.BatchNorm2d(n_filters * 16)

        self.leaky_relu = nn.LeakyReLU(0.2, True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        encoder1 = self.conv1(x)
        encoder2 = self.batch_norm2(self.conv2(self.leaky_relu(encoder1)))
        encoder3 = self.batch_norm4(self.conv3(self.leaky_relu(encoder2)))
        encoder4 = self.batch_norm8(self.conv4(self.leaky_relu(encoder3)))
        encoder5 = self.batch_norm16(self.conv5(self.leaky_relu(encoder4)))
        encoder6 = self.conv6(self.leaky_relu(encoder5))
        output =  self.sigmoid(encoder6)
        return output