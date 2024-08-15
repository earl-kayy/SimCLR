import torch
import torch.nn as nn
import torch.nn.functional as F

# 필요한 모듈 임포트
from torchvision import models

# Encoder 클래스 정의
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.resnet = models.resnet50(weights=None)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, 1024)

    def forward(self, x):
        return self.resnet(x)

# Enc_PH 클래스 정의 : Pretext Task
class Enc_PH(nn.Module):
    def __init__(self, model, feat_dim=128):
        super(Enc_PH, self).__init__()
        self.dim_in = model.resnet.fc.out_features
        self.encoder = model
        self.head = nn.Sequential(
            nn.Linear(self.dim_in, self.dim_in),
            nn.ReLU(inplace=True),
            nn.Linear(self.dim_in, feat_dim)
        )

    def forward(self, x):
        feat = self.encoder(x)
        feat = F.normalize(self.head(feat), dim=1)
        return feat

# Enc_Classifier 클래스 정의 : Downstream Task
class Enc_Classifier(nn.Module):
    def __init__(self, model):
        super(Enc_Classifier, self).__init__()
        self.encoder = model.encoder
        self.mlp = nn.Sequential(
            nn.Linear(model.encoder.resnet.fc.out_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.mlp(x)
        return x
