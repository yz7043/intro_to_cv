import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, is_down=True, use_identity=True, **kwargs):
        super().__init__()
        if is_down:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, padding_mode="reflect", **kwargs),
                nn.InstanceNorm2d(out_ch),
                nn.ReLU(inplace=True) if use_identity else nn.Identity()
            )
        else:
            self.conv = nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, **kwargs),
                nn.InstanceNorm2d(out_ch),
                nn.ReLU(inplace=True) if use_identity else nn.Identity()
            )

    def forward(self, x):
        return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(ch, ch, kernel_size=3, padding=1),
            ConvBlock(ch, ch, use_identity=False, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    def __init__(self, img_ch, num_features=64, num_residuals=9):
        super().__init__()
        self.first_layer = nn.Sequential(
            nn.Conv2d(img_ch, num_features, kernel_size=7, stride=1, padding=3, padding_mode="reflect"),
            nn.ReLU(inplace=True)
        )
        self.down_blocks = nn.ModuleList(
            [
                ConvBlock(num_features, num_features*2, is_down=True, kernel_size=3, stride=2, padding=1),
                ConvBlock(num_features*2, num_features*4, is_down=True, kernel_size=3, stride=2, padding=1),
            ]
        )

        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(num_features*4) for _ in range(num_residuals)]
        )
        self.up_blocks = nn.ModuleList(
            [
                ConvBlock(num_features*4, num_features*2, is_down=False, kernel_size=3, stride=2, padding=1,
                          output_padding=1),
                ConvBlock(num_features*2, num_features, is_down=False, kernel_size=3, stride=2, padding=1,
                          output_padding=1),
            ]
        )
        self.last_layer = nn.Conv2d(num_features, img_ch, kernel_size=7, stride=1, padding=3, padding_mode="reflect")

    def forward(self, x):
        x = self.first_layer(x)
        for layer in self.down_blocks:
            x = layer(x)
        x = self.residual_blocks(x)
        for layer in self.up_blocks:
            x = layer(x)
        x = self.last_layer(x)
        return torch.tanh(x)


if __name__ == "__main__":
    # test if our model works well, generating 5 example with 3 channels and size of 256 *256
    img_ch = 3
    img_size = 256
    # the output photo from generator should have the same size as the input image
    x = torch.randn((2, img_ch, img_size, img_size))
    gen = Generator(img_ch, 9)
    print(gen(x).shape)