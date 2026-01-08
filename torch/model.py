"""
InverseRenderNet_v2 PyTorch Implementation.

Pure PyTorch port of InverseRenderNet_v2 for inverse rendering.
Predicts albedo, normals, shadow, and SH lighting coefficients from a single image.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GroupNorm32(nn.GroupNorm):
    """GroupNorm with 32 groups."""
    def __init__(self, num_channels: int):
        super().__init__(num_groups=32, num_channels=num_channels, eps=1e-5, affine=True)


class ConvGNReLU(nn.Module):
    """Conv2d + GroupNorm + ReLU block."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False)
        self.gn = GroupNorm32(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        return self.relu(self.gn(self.conv(x)))


class ConvBlock(nn.Module):
    """Conv2d without activation (for final layers)."""
    def __init__(self, in_channels: int, out_channels: int, bias: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=bias)
        
    def forward(self, x):
        return self.conv(x)


class InverseRenderNet(nn.Module):
    """
    PyTorch port of InverseRenderNet_v2.
    
    Encoder-decoder with skip connections for predicting albedo, normals, and shadow.
    """
    
    def __init__(self, n_layers: int = 30, n_pools: int = 4, depth_base: int = 32):
        super().__init__()
        self.n_layers = n_layers
        self.n_pools = n_pools
        
        conv_layers = n_layers // 2 - 1
        deconv_layers = n_layers // 2
        nlayers_bef_pool = int(np.ceil((conv_layers - 1) / n_pools) - 1)
        max_depth = 512
        max_num_pool = int(np.log2(max_depth / depth_base))
        tail = int(conv_layers - nlayers_bef_pool * max_num_pool)
            
        self.nlayers_bef_pool = nlayers_bef_pool
        self.deconv_layers = deconv_layers
        
        # Encoder output channels
        f_out_conv = ([64] + 
            [int(depth_base * 2 ** (np.floor(i / nlayers_bef_pool))) 
             for i in range(1, conv_layers - tail + 1)] +
            [int(depth_base * 2 ** max_num_pool) 
             for i in range(conv_layers - tail + 1, conv_layers + 1)])
        
        f_in_conv = [3] + f_out_conv[:-1]
        
        # TF formula-based f_in for decoder output computation
        f_in_conv_tf = ([3] + 
            [int(depth_base * 2 ** (np.ceil(i / nlayers_bef_pool) - 1)) 
             for i in range(1, conv_layers - tail + 1)] +
            [int(depth_base * 2 ** max_num_pool) 
             for i in range(conv_layers - tail + 1, conv_layers + 1)])
        
        # Decoder channels
        f_out_am_deconv = f_in_conv_tf[:0:-1] + [3]
        f_out_nm_deconv = f_in_conv_tf[:0:-1] + [2]
        f_out_mask_deconv = f_in_conv_tf[:0:-1] + [1]
        
        f_in_deconv = [f_out_conv[-1]]
        for i in range(1, deconv_layers):
            prev_out = f_out_am_deconv[i-1]
            did_concat = (i % nlayers_bef_pool == 0 and i <= n_pools * nlayers_bef_pool)
            f_in_deconv.append(prev_out * 2 if did_concat else prev_out)
        
        # Build encoder
        self.encoder_layers = nn.ModuleList([
            ConvGNReLU(f_in_conv[i], f_out_conv[i]) for i in range(conv_layers + 1)
        ])
                
        # Build decoders
        self.albedo_decoder = self._build_decoder(f_in_deconv, f_out_am_deconv, deconv_layers, True)
        self.normal_decoder = self._build_decoder(f_in_deconv, f_out_nm_deconv, deconv_layers, False)
        self.shadow_decoder = self._build_decoder(f_in_deconv, f_out_mask_deconv, deconv_layers, True)
        
    def _build_decoder(self, f_in, f_out, num_layers, final_bias):
        layers = nn.ModuleList()
        for i in range(num_layers):
            if i == num_layers - 1:
                layers.append(ConvBlock(f_in[i], f_out[i], bias=final_bias))
            else:
                layers.append(ConvGNReLU(f_in[i], f_out[i]))
        return layers

    def _forward_encoder(self, x):
        skip_features = []
        for i, layer in enumerate(self.encoder_layers):
            tf_i = i + 1
            do_pool = ((tf_i - 1) % self.nlayers_bef_pool == 0 and 
                       tf_i <= self.n_pools * self.nlayers_bef_pool + 1 and tf_i != 1)
            if do_pool:
                skip_features.append(x)
                x = F.max_pool2d(layer(x), 2, 2)
            else:
                x = layer(x)
        return x, skip_features
    
    def _forward_decoder(self, x, skip_features, decoder):
        skip_idx = len(skip_features) - 1
        for i, layer in enumerate(decoder):
            tf_i = i + 1
            do_upsample = (tf_i % self.nlayers_bef_pool == 0 and 
                          tf_i <= self.n_pools * self.nlayers_bef_pool)
            if do_upsample and skip_idx >= 0:
                skip = skip_features[skip_idx]
                skip_idx -= 1
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
                x = torch.cat([layer(x), skip], dim=1)
            else:
                x = layer(x)
        return x
        
    def forward(self, x, mask):
        """
        Forward pass.
        
        Args:
            x: Input image (B, 3, H, W), normalized to [-1, 1]
            mask: Binary mask (B, 1, H, W)
            
        Returns:
            albedo, normal, shadow tensors
        """
        features, skip_features = self._forward_encoder(x)
        
        albedo_raw = self._forward_decoder(features, skip_features, self.albedo_decoder)
        albedo = torch.clamp(torch.tanh(albedo_raw) * mask, -0.9999, 0.9999)
        
        nm_raw = self._forward_decoder(features, skip_features, self.normal_decoder)
        nm_norm = torch.sqrt(nm_raw.pow(2).sum(dim=1, keepdim=True) + 1.0)
        normal = torch.cat([nm_raw / nm_norm, 1.0 / nm_norm], dim=1) * mask
        
        shadow_raw = self._forward_decoder(features, skip_features, self.shadow_decoder)
        shadow = torch.clamp(torch.tanh(shadow_raw) * mask, -0.9999, 0.9999)
        
        return albedo, normal, shadow


def load_weights(model, weight_path):
    """Load PyTorch weights."""
    state_dict = torch.load(weight_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict)
    print(f"Loaded weights from {weight_path}")
