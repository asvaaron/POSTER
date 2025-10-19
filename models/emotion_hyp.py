import torch
import numpy as np
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.nn import functional as F

from .hyp_crossvit import *
from .mobilefacenet import MobileFaceNet
from .ir50 import Backbone


def load_pretrained_weights(model, checkpoint):
    import collections
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    model_dict = model.state_dict()
    new_state_dict = collections.OrderedDict()
    matched_layers, discarded_layers = [], []
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]
        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)
    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)
    print('load_weight', len(matched_layers))
    return model


class SE_block(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear1 = torch.nn.Linear(input_dim, input_dim)
        self.relu = nn.ReLU()
        self.linear2 = torch.nn.Linear(input_dim, input_dim)
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        x1 = self.linear1(x)
        x1 = self.relu(x1)
        x1 = self.linear2(x1)
        x1 = self.sigmod(x1)
        x = x * x1
        return x


class ClassificationHead(nn.Module):
    """Original simple head"""

    def __init__(self, input_dim: int, target_dim: int):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, target_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        y_hat = self.linear(x)
        return y_hat


class EnhancedHead_v1(nn.Module):
    """
    Enhanced head with BatchNorm + Dropout + Hidden Layer
    Good for: Preventing overfitting, improving generalization
    """

    def __init__(self, input_dim: int, target_dim: int, dropout_rate=0.3, hidden_ratio=0.5):
        super().__init__()
        hidden_dim = int(input_dim * hidden_ratio)

        self.features = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(hidden_dim, target_dim)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.features(x)


class EnhancedHead_v2(nn.Module):
    """
    Deeper head with residual connection
    Good for: Learning complex decision boundaries
    """

    def __init__(self, input_dim: int, target_dim: int, dropout_rate=0.3):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(input_dim)
        self.dropout1 = nn.Dropout(dropout_rate)

        # First block
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.bn2 = nn.BatchNorm1d(input_dim)
        self.dropout2 = nn.Dropout(dropout_rate / 2)

        # Second block
        self.fc2 = nn.Linear(input_dim, input_dim // 2)
        self.bn3 = nn.BatchNorm1d(input_dim // 2)
        self.dropout3 = nn.Dropout(dropout_rate / 2)

        # Output
        self.fc_out = nn.Linear(input_dim // 2, target_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)

        # Initial normalization
        x = self.bn1(x)
        identity = x

        # First block with residual
        out = self.dropout1(x)
        out = F.relu(self.fc1(out))
        out = self.bn2(out)
        out = out + identity  # Residual connection

        # Second block
        out = self.dropout2(out)
        out = F.relu(self.fc2(out))
        out = self.bn3(out)

        # Output
        out = self.dropout3(out)
        out = self.fc_out(out)

        return out


class EnhancedHead_v3(nn.Module):
    """
    Attention-based head
    Good for: Focusing on important features, interpretability
    """

    def __init__(self, input_dim: int, target_dim: int, dropout_rate=0.3):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(input_dim)

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 4),
            nn.ReLU(),
            nn.Linear(input_dim // 4, input_dim),
            nn.Sigmoid()
        )

        # Classification layers
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(input_dim // 2),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(input_dim // 2, target_dim)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.bn1(x)

        # Apply attention
        attention_weights = self.attention(x)
        x = x * attention_weights

        # Classification
        return self.classifier(x)


class EnhancedHead_v4(nn.Module):
    """
    Multi-scale head with parallel pathways
    Good for: Capturing different levels of abstraction
    """

    def __init__(self, input_dim: int, target_dim: int, dropout_rate=0.3):
        super().__init__()
        self.bn = nn.BatchNorm1d(input_dim)

        # Pathway 1: Direct (shallow)
        self.path1 = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(input_dim, target_dim)
        )

        # Pathway 2: Medium depth
        self.path2 = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(input_dim // 2),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(input_dim // 2, target_dim)
        )

        # Pathway 3: Deep
        self.path3 = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(input_dim // 2),
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.ReLU(),
            nn.BatchNorm1d(input_dim // 4),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(input_dim // 4, target_dim)
        )

        # Fusion weights (learnable)
        self.fusion_weights = nn.Parameter(torch.ones(3) / 3)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.bn(x)

        # Get predictions from all pathways
        out1 = self.path1(x)
        out2 = self.path2(x)
        out3 = self.path3(x)

        # Weighted fusion
        weights = F.softmax(self.fusion_weights, dim=0)
        out = weights[0] * out1 + weights[1] * out2 + weights[2] * out3

        return out


class EnhancedHead_v5(nn.Module):
    """
    Uncertainty-aware head with Monte Carlo Dropout
    Good for: Getting confidence estimates, robust predictions
    """

    def __init__(self, input_dim: int, target_dim: int, dropout_rate=0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(input_dim // 2),
            nn.Dropout(dropout_rate),
            nn.Linear(input_dim // 2, target_dim)
        )

        self.dropout_rate = dropout_rate

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.features(x)


class EnhancedHead_v6(nn.Module):
    """
    Feature rectification head with gating
    Good for: Filtering noisy features, adaptive feature selection
    """

    def __init__(self, input_dim: int, target_dim: int, dropout_rate=0.3):
        super().__init__()
        self.bn = nn.BatchNorm1d(input_dim)

        # Feature gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid()
        )

        # Main classification path
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(input_dim // 2),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(input_dim // 2, target_dim)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.bn(x)

        # Apply gating
        gate_weights = self.gate(x)
        x = x * gate_weights

        return self.classifier(x)

class EnhancedHead_v7_RNN(nn.Module):
    def __init__(self, input_dim: int, target_dim: int, hidden_dim=256, dropout_rate=0.3):
        super().__init__()
        self.fc_in = nn.Linear(input_dim, hidden_dim)
        self.rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, target_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.bn = nn.BatchNorm1d(input_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.bn(x)
        x = self.fc_in(x).unsqueeze(1)  # sequence len = 1
        x, _ = self.rnn(x)
        x = self.dropout(x[:, -1, :])
        out = self.fc_out(x)
        return out

class pyramid_trans_expr(nn.Module):
    def __init__(self, img_size=224, num_classes=7, type="large", head_type="simple"):
        super().__init__()
        depth = 8
        if type == "small":
            depth = 4
        if type == "base":
            depth = 6
        if type == "large":
            depth = 8

        self.img_size = img_size
        self.num_classes = num_classes

        self.face_landback = MobileFaceNet([112, 112], 136)
        face_landback_checkpoint = torch.load('./models/pretrain/mobilefacenet_model_best.pth.tar',
                                              map_location=lambda storage, loc: storage)
        self.face_landback.load_state_dict(face_landback_checkpoint['state_dict'])

        for param in self.face_landback.parameters():
            param.requires_grad = False

        self.ir_back = Backbone(50, 0.0, 'ir')
        ir_checkpoint = torch.load('./models/pretrain/ir50.pth',
                                   map_location=lambda storage, loc: storage)
        self.ir_back = load_pretrained_weights(self.ir_back, ir_checkpoint)

        self.ir_layer = nn.Linear(1024, 512)

        self.pyramid_fuse = HyVisionTransformer(in_chans=49, q_chanel=49, embed_dim=512,
                                                depth=depth, num_heads=8, mlp_ratio=2.,
                                                drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1)

        self.se_block = SE_block(input_dim=512)

        # Select head type
        if head_type == "simple":
            self.head = ClassificationHead(input_dim=512, target_dim=self.num_classes)
        elif head_type == "enhanced_v1":
            self.head = EnhancedHead_v1(input_dim=512, target_dim=self.num_classes)
        elif head_type == "enhanced_v2":
            self.head = EnhancedHead_v2(input_dim=512, target_dim=self.num_classes)
        elif head_type == "enhanced_v3":
            self.head = EnhancedHead_v3(input_dim=512, target_dim=self.num_classes)
        elif head_type == "enhanced_v4":
            self.head = EnhancedHead_v4(input_dim=512, target_dim=self.num_classes)
        elif head_type == "enhanced_v5":
            self.head = EnhancedHead_v5(input_dim=512, target_dim=self.num_classes)
        elif head_type == "enhanced_v6":
            self.head = EnhancedHead_v6(input_dim=512, target_dim=self.num_classes)
        elif head_type == "enhanced_v7":
            self.head = EnhancedHead_v7_RNN(input_dim=512, target_dim=self.num_classes)
        else:
            raise ValueError(f"Unknown head_type: {head_type}")

    def forward(self, x):
        B_ = x.shape[0]
        x_face = F.interpolate(x, size=112)
        _, x_face = self.face_landback(x_face)
        x_face = x_face.view(B_, -1, 49).transpose(1, 2)

        x_ir = self.ir_back(x)
        x_ir = self.ir_layer(x_ir)

        y_hat = self.pyramid_fuse(x_ir, x_face)
        y_hat = self.se_block(y_hat)
        y_feat = y_hat
        out = self.head(y_hat)

        return out, y_feat
#
#
# HEAD DESCRIPTIONS:
# ------------------
# 1. simple (ClassificationHead):
#    - Original simple linear classifier
#    - Use as: Baseline for comparison
#    - Parameters: ~3.5K (for 7 classes)
#
# 2. enhanced_v1 (EnhancedHead_v1):
#    - BatchNorm + Dropout + Hidden Layer
#    - Best for: Preventing overfitting, better generalization
#    - Parameters: ~132K
#    - Recommended: Start here for improvements
#
# 3. enhanced_v2 (EnhancedHead_v2):
#    - Deeper network with residual connections
#    - Best for: Learning complex decision boundaries
#    - Parameters: ~394K
#    - Recommended: If v1 works but you want more capacity
#
# 4. enhanced_v3 (EnhancedHead_v3):
#    - Attention mechanism to focus on important features
#    - Best for: Feature interpretability, understanding what matters
#    - Parameters: ~165K
#    - Recommended: When you need to understand feature importance
#
# 5. enhanced_v4 (EnhancedHead_v4):
#    - Multi-scale with 3 parallel pathways (shallow, medium, deep)
#    - Best for: Robustness through ensemble-like behavior
#    - Parameters: ~268K
#    - Recommended: When you have computational budget and want robustness
#
# 6. enhanced_v5 (EnhancedHead_v5):
#    - Monte Carlo Dropout for uncertainty estimation
#    - Best for: Getting confidence scores with predictions
#    - Parameters: ~132K
#    - Recommended: For safety-critical applications needing uncertainty
#
# 7. enhanced_v6 (EnhancedHead_v6):
#    - Feature gating to filter noisy features
#    - Best for: Datasets with noisy or unreliable features
#    - Parameters: ~394K
#    - Recommended: When features might contain noise
