import math
import torch
import torch.nn as nn
from functools import partial

class SS2DBase(nn.Module):
    def __init__(
        self,
        model_dim=96,
        state_dim=16,
        ssm_factor=2.0,
        dt_rank="auto",
        activation=nn.SiLU,
        conv_kernel=3,
        conv_bias=True,
        dropout_prob=0.0,
        bias_flag=False,
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_floor=1e-4,
        init_mode="v0",
        forward_style="v2",
        channel_first=False,
        **kwargs,
    ):
        super().__init__()
        self.K = 4
        self.model_dim = model_dim
        self.state_dim = state_dim
        self.inner_dim = int(ssm_factor * model_dim)
        self.dt_rank = math.ceil(model_dim / 16) if dt_rank == "auto" else dt_rank
        self.channel_first = channel_first
        self.use_conv = conv_kernel > 1

        # Forward selection
        self.forward_fn = partial(self.forward_core, backend="core")
        if forward_style.startswith("v0"):
            self._init_v0(**kwargs)
        else:
            self._init_v2(**kwargs)

        # Input projection
        proj_out_dim = self.inner_dim * 2
        self.input_proj = nn.Linear(model_dim, proj_out_dim, bias=bias_flag)
        self.activation = activation()

        # Depthwise conv
        if self.use_conv:
            self.dw_conv = nn.Conv2d(
                in_channels=self.inner_dim,
                out_channels=self.inner_dim,
                kernel_size=conv_kernel,
                padding=(conv_kernel - 1) // 2,
                groups=self.inner_dim,
                bias=conv_bias,
            )

        # Output projection
        self.output_norm = nn.LayerNorm(self.inner_dim)
        self.output_proj = nn.Linear(self.inner_dim, model_dim, bias=bias_flag)
        self.dropout_layer = nn.Dropout(dropout_prob) if dropout_prob > 0 else nn.Identity()

    def _init_v0(self, **kwargs):
        # Example placeholder for v0 initialization logic
        self.A_logs = nn.Parameter(torch.randn(self.K * self.inner_dim, self.state_dim))
        self.Ds = nn.Parameter(torch.ones(self.K * self.inner_dim))
        self.dt_weight = nn.Parameter(torch.randn(self.K, self.inner_dim, self.dt_rank))
        self.dt_bias = nn.Parameter(torch.randn(self.K, self.inner_dim))

    def _init_v2(self, **kwargs):
        # Example placeholder for v2 initialization logic
        self.A_logs = nn.Parameter(torch.zeros(self.K * self.inner_dim, self.state_dim))
        self.Ds = nn.Parameter(torch.ones(self.K * self.inner_dim))
        self.dt_weight = nn.Parameter(0.1 * torch.rand(self.K, self.inner_dim, self.dt_rank))
        self.dt_bias = nn.Parameter(0.1 * torch.rand(self.K, self.inner_dim))

    def forward_core(
        self, x, backend="core", **kwargs
    ):
        # Core processing logic (placeholder)
        B, C, H, W = x.shape
        # Apply some processing; e.g., selective scan
        # For demonstration, just return input
        return x

    def forward(self, x):
        # Input projection
        x_proj = self.input_proj(x)
        x_part, z_part = x_proj.chunk(2, dim=-1)
        z_part = self.activation(z_part)

        # Rearrange for conv if needed
        if not self.channel_first:
            x_part = x_part.permute(0, 3, 1, 2).contiguous()
        if self.use_conv:
            x_part = self.dw_conv(x_part)
        x_part = self.activation(x_part)

        # Core computation
        y = self.forward_fn(x_part)

        # Output projection
        y = self.output_norm(y)
        y = y * z_part
        out = self.dropout_layer(self.output_proj(y))
        return out
