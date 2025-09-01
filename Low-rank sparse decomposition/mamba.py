import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

# 可选依赖
try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    selective_scan_fn, mamba_inner_fn = None, None
    selective_state_update = None
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


class Mamba(nn.Module):
    """Mamba block implementation.
    
    Args:
        d_model (int): 输入 embedding 维度
        d_state (int): 状态大小
        d_conv (int): 卷积核大小
        expand (int): 通道扩展倍率
        dt_rank (int | "auto"): delta 投影 rank
        dt_init (str): "constant" 或 "random"
        use_fast_path (bool): 是否使用 fused kernel
        layer_idx (int): 层序号（推理缓存使用）
    """

    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        # 基本参数
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx

        # 投影层
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias, **factory_kwargs)

        # 深度可分离卷积
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            bias=conv_bias,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        # dt, B, C 投影
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * self.d_state, bias=False, **factory_kwargs)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)
        self._init_dt(dt_init, dt_scale, dt_min, dt_max, dt_init_floor, factory_kwargs)

        # 状态空间参数
        self.A_log, self.D = self._init_ssm(factory_kwargs, device)

    # ---------------- 初始化辅助函数 ----------------
    def _init_dt(self, dt_init, dt_scale, dt_min, dt_max, dt_init_floor, factory_kwargs):
        """初始化 delta 投影参数"""
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError(f"Unsupported dt_init={dt_init}")

        # 初始化 bias，使 softplus(dt_bias) ∈ [dt_min, dt_max]
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

    def _init_ssm(self, factory_kwargs, device):
        """初始化状态空间参数 A_log, D"""
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        )
        A_log = nn.Parameter(torch.log(A.contiguous()))
        A_log._no_weight_decay = True

        D = nn.Parameter(torch.ones(self.d_inner, device=device))
        D._no_weight_decay = True
        return A_log, D

    # ---------------- Forward ----------------
    def forward(self, hidden_states, inference_params=None):
        """forward: (B, L, D) -> (B, L, D)"""
        B, L, _ = hidden_states.shape

        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, B)
            if inference_params.seqlen_offset > 0:
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                return out

        xz = self._in_projection(hidden_states, L)

        if self.use_fast_path and causal_conv1d_fn and inference_params is None:
            return self._fast_forward(xz)
        else:
            return self._fallback_forward(xz, L, conv_state, ssm_state)

    def _in_projection(self, hidden_states, seqlen):
        """BLD -> 分裂 x, z 前的投影"""
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz += rearrange(self.in_proj.bias, "d -> d 1")
        return xz

    def _fast_forward(self, xz):
        """使用 fused kernel 的 forward"""
        A = -torch.exp(self.A_log.float())
        return mamba_inner_fn(
            xz,
            self.conv1d.weight,
            self.conv1d.bias,
            self.x_proj.weight,
            self.dt_proj.weight,
            self.out_proj.weight,
            self.out_proj.bias,
            A,
            None, None,  # B, C 依赖输入时才传
            self.D.float(),
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
        )

    def _fallback_forward(self, xz, seqlen, conv_state, ssm_state):
        """纯 PyTorch fallback forward"""
        x, z = xz.chunk(2, dim=1)
        x = self._apply_conv(x, seqlen, conv_state)

        # dt, B, C
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj.weight @ dt.t()
        dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
        B = rearrange(B, "(b l) n -> b n l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) n -> b n l", l=seqlen).contiguous()

        y = selective_scan_fn(
            x, dt, -torch.exp(self.A_log.float()), B, C, self.D.float(),
            z=z, delta_bias=self.dt_proj.bias.float(), delta_softplus=True,
            return_last_state=ssm_state is not None,
        )
        if ssm_state is not None:
            y, last_state = y
            ssm_state.copy_(last_state)

        y = rearrange(y, "b d l -> b l d")
        return self.out_proj(y)

    def _apply_conv(self, x, seqlen, conv_state):
        """卷积处理 + 状态更新"""
        if conv_state is not None:
            conv_state.copy_(F.pad(x, (self.d_conv - x.shape[-1], 0)))
        if causal_conv1d_fn is None:
            return self.act(self.conv1d(x)[..., :seqlen])
        return causal_conv1d_fn(
            x=x,
            weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
            bias=self.conv1d.bias,
            activation="silu",
        )
