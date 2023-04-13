import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 4, stride, 1, bias=True, padding_mode="reflect"),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.conv(x)


class Discriminator(nn.Module):
    def __init__(self, in_ch=3, features=[64, 128, 256, 512]):
        super().__init__()
        self.first_layer = nn.Sequential(
            nn.Conv2d(
                in_ch,
                features[0],
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="reflect"
            ),
            nn.LeakyReLU(0.2),
        )

        hidden_layers = []
        in_ch = features[0]
        for f in features[1: ]:
            hidden_layers.append(BasicBlock(in_ch, f, stride=1 if f == features[-1] else 2))
            in_ch = f
        hidden_layers.append(nn.Conv2d(in_ch, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect"))
        self.model = nn.Sequential(*hidden_layers)

    def forward(self, x):
        x = self.first_layer(x)
        return torch.sigmoid(self.model(x))


if __name__ == "__main__":
    # test if our model works well, generating 5 example with 3 channels and size of 256 *256
    x = torch.randn((5, 3, 256, 256))
    model = Discriminator(in_ch=3)
    prediction = model(x)
    print(prediction.shape)