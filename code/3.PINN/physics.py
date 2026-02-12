"""
物理层：溶氧动力学方程（PyTorch 可微分实现）

核心方程:
    dDO/dt = K_La * (DO_sat(T) - DO) - R_fish(T) - R_bio + P_photo(light)

其中:
    DO_sat(T) = 14.62 - 0.3898*T + 0.006969*T^2 - 5.897e-5*T^3
    R_fish(T) = R_fish_base * (1 + alpha_T * (T - T_ref))
    P_photo   = P_photo_rate * light_hours
"""
import torch
import torch.nn as nn


def do_saturation(T: torch.Tensor) -> torch.Tensor:
    """
    饱和溶氧浓度 (Benson & Krause, 1984 简化版)
    
    Args:
        T: 水温 (℃)
    Returns:
        DO_sat (mg/L)
    """
    return 14.62 - 0.3898 * T + 0.006969 * T**2 - 5.897e-5 * T**3


class PhysicsParams(nn.Module):
    """
    可学习的物理参数模块。
    所有参数在 log 空间中优化以保证正定性。
    """
    def __init__(self, init_values: dict, bounds: dict):
        super().__init__()
        self.bounds = bounds
        
        # 在 log 空间中初始化参数
        for name, val in init_values.items():
            # log_param = log(val) → param = exp(log_param) > 0
            log_val = torch.log(torch.tensor(float(val)))
            setattr(self, f"log_{name}", nn.Parameter(log_val))
    
    def get(self, name: str) -> torch.Tensor:
        """获取参数值（正定）"""
        log_param = getattr(self, f"log_{name}")
        val = torch.exp(log_param)
        
        # 软裁剪到合理范围
        if name in self.bounds:
            lo, hi = self.bounds[name]
            val = torch.clamp(val, min=lo, max=hi)
        return val
    
    def get_all(self) -> dict:
        """获取所有参数的当前值"""
        result = {}
        for attr_name in dir(self):
            if attr_name.startswith("log_"):
                param_name = attr_name[4:]
                result[param_name] = self.get(param_name).item()
        return result


def compute_ode_residual(
    DO_pred: torch.Tensor,     # 网络预测的 DO (N,)
    t: torch.Tensor,           # 时间 (N,1)，需要 requires_grad=True，与网络输入共享
    T_water: torch.Tensor,     # 水温 (N,)
    light_hours: torch.Tensor, # 光照时长 (N,)
    physics: PhysicsParams,    # 可学习物理参数
) -> torch.Tensor:
    """
    计算 ODE 残差:
        residual = dDO/dt - f(DO, T, light)
    
    其中 f = K_La*(DO_sat - DO) - R_fish - R_bio + P_photo
    
    使用自动微分计算 dDO/dt。
    t 必须是与网络输入共享的同一个张量 (shape [N,1], requires_grad=True)。
    
    Returns:
        residual (N,): ODE 残差向量
    """
    # 1. 自动微分：dDO/dt
    dDO_dt = torch.autograd.grad(
        outputs=DO_pred,
        inputs=t,
        grad_outputs=torch.ones_like(DO_pred),
        create_graph=True,
        retain_graph=True,
    )[0].squeeze(-1)  # (N,1) -> (N,)
    
    # 2. 物理项
    K_La = physics.get("K_La")
    R_fish_base = physics.get("R_fish_base")
    alpha_T = physics.get("alpha_T")
    T_ref = 25.0  # 固定参考温度
    R_bio = physics.get("R_bio")
    P_photo_rate = physics.get("P_photo_rate")
    
    # DO 饱和度
    DO_sat = do_saturation(T_water)
    
    # 复氧项
    reaeration = K_La * (DO_sat - DO_pred)
    
    # 鱼呼吸耗氧（温度修正）
    R_fish = R_fish_base * (1.0 + alpha_T * (T_water - T_ref))
    
    # 光合产氧
    P_photo = P_photo_rate * light_hours
    
    # ODE 右端项
    f_ode = reaeration - R_fish - R_bio + P_photo
    
    # 残差
    residual = dDO_dt - f_ode
    
    return residual


def compute_boundary_loss(
    DO_pred: torch.Tensor,
    T_water: torch.Tensor,
) -> torch.Tensor:
    """
    物理约束损失:
    1. DO >= 0 (不能为负)
    2. DO <= DO_sat(T) (不超过饱和值)
    """
    DO_sat = do_saturation(T_water)
    
    # 惩罚负值
    loss_neg = torch.mean(torch.relu(-DO_pred))
    
    # 惩罚超饱和
    loss_over = torch.mean(torch.relu(DO_pred - DO_sat * 1.1))  # 允许 10% 超饱和（光合作用）
    
    return loss_neg + loss_over
