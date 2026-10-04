import torch
import torch.nn as nn

class CNNScratch(nn.Module):
    """
    CNN built completely from scratch for glaucoma stage classification.
    Expected input: 
        [batch_size,3,416,416]
    Output:
        [batch_size, num_classes]
    """

    def __init__(self, num_classes=3):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                padding=1,
                bias = False
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2,stride=2)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1,
                bias = False
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2,stride=2)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1,
                bias = False
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2,stride=2)
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                padding=1,
                bias = False
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2,stride=2)
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1,1))

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256,num_classes)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu"
                )
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight,1)
                nn.init.constant_(module.bias,0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)

                if module.bias is not None:
                    nn.init.constant_(module.bias,0)
                


    def forward(self,x):
        x= self.block1(x)
        x= self.block2(x)
        x= self.block3(x)
        x= self.block4(x)

        x=self.global_pool(x)
        x=torch.flatten(x,start_dim =1)
        x=self.classifier(x)

        return x
