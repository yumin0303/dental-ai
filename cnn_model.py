import torch.nn as nn
import torchvision.models as models


class DentalCNN(nn.Module):
    """
    ResNet18 기반 치아 교정 필요도 분류기.
    - layer1~3: 완전 동결 (ImageNet 일반 특징 유지)
    - layer4: 학습 가능 (치아 이미지 특화)
    - classifier: 학습 가능
    """

    def __init__(self, pretrained=True):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        base = models.resnet18(weights=weights)

        # layer1~3 동결
        frozen_layers = list(base.children())[:7]
        for layer in frozen_layers:
            for param in layer.parameters():
                param.requires_grad = False

        # layer4는 학습 (치아 특화 fine-tuning)
        self.backbone = nn.Sequential(*list(base.children())[:-1])

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.classifier(self.backbone(x))
