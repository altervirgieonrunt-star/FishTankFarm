"""
PINN 网络模型
- 标准 MLP：输入环境特征 → 输出 DO 预测
- 集成物理残差计算
"""
import torch
import torch.nn as nn


class PINN(nn.Module):
    """
    Physics-Informed Neural Network for DO Prediction
    
    输入: [t_norm, 水温, 气温, 光照时长, ...]  (n_features 维)
    输出: DO 预测值 (1维)
    """
    def __init__(self, n_features: int, hidden_layers: list, activation: str = "tanh"):
        super().__init__()
        
        layers = []
        prev_dim = n_features
        
        for h_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, h_dim))
            if activation == "tanh":
                layers.append(nn.Tanh())
            elif activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "gelu":
                layers.append(nn.GELU())
            elif activation == "silu":
                layers.append(nn.SiLU())
            prev_dim = h_dim
        
        # 输出层：1 维
        layers.append(nn.Linear(prev_dim, 1))
        
        self.net = nn.Sequential(*layers)
        
        # Xavier 初始化
        self._init_weights()
    
    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, n_features)
        Returns:
            DO_pred: (batch,)
        """
        return self.net(x).squeeze(-1)
