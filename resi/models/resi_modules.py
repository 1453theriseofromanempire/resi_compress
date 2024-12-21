from turtle import forward
import torch
import torch.nn as nn
import sys

from resi.models.layers import ResidualBottleneckBlock, AttentionBlock
from resi.models.layers import conv, conv3x3, deconv

sys.path.append("/root/home/codes/resi_compress/src/latent-diffusion")
from ldm.modules.diffusionmodules.openaimodel import (
    ResBlock, 
    Downsample, 
    Upsample, 
    TimestepEmbedSequential
)
from ldm.modules.attention import SpatialTransformer as CrossAttenBlock
from ldm.modules.diffusionmodules.util import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)

class ResiEncoder(nn.Module):
    def __init__(self, N=192, M=320):
        super().__init__(self)

        self.N = int(N)
        self.M = int(M)

        self.g_img = nn.Sequential(
            conv(3, N),
            ResidualBottleneckBlock(N),
            ResidualBottleneckBlock(N),
            ResidualBottleneckBlock(N),
            conv(N, N),
            ResidualBottleneckBlock(N),
            ResidualBottleneckBlock(N),
            ResidualBottleneckBlock(N),
            AttentionBlock(N),
            conv(N, N),
            ResidualBottleneckBlock(N),
            ResidualBottleneckBlock(N),
            ResidualBottleneckBlock(N),
            conv(N, M),
            AttentionBlock(M),
        )

        self.h_z = nn.Sequential(
            conv3x3(M, N),
            nn.ReLU(inplace=True),
            conv(N, N),
            nn.ReLU(inplace=True),
            conv(N, N),
        )


    def forward(self, img, quant):

        z = self.g_img(img)

        z = torch.cat([z, quant], dim=1)

        return z


class HyperPriorModel(nn.Module):
    def __init__(self, N=192, M=320):
        super().__init__(self, )
        self.N = int(N)
        self.M = int(M)
        
        self.h_a = nn.Sequential(
            conv3x3(M, N),
            nn.ReLU(inplace=True),
            conv(N, N),
            nn.ReLU(inplace=True),
            conv(N, N),
        )

        self.h_s = nn.Sequential(
            deconv(N, N),
            nn.ReLU(inplace=True),
            deconv(N, N*3//2),
            nn.ReLU(inplace=True),
            conv3x3(N*3//2, 2*M),
        )
    
    def encode(self, z):

        return self.h_a(z)

    def decode(self, z_hat):

        return self.h_s(z_hat)


class ResiDecoder(nn.Module):
    def __init__(self, N=192, M=320):
        super().__init__(self,)

        self.N = int(N)
        self.M = int(M)

        self.h_s = nn.Sequential(
            deconv(N, N),
            nn.ReLU(inplace=True),
            deconv(N, N*3//2),
            nn.ReLU(inplace=True),
            conv3x3(N*3//2, N),
        )
    
    def forward(self, z_hat):

        return self.h_s(z_hat)


class FusionModel(nn.Module):
    def __init__(self, 
                 model_channels, 
                 in_channels, 
                 channel_mult=(1, 2,), 
                 dropout=0,
                 dims=2, 
                 num_heads=-1,
                 num_head_channels=-1,
                 conv_resample=True,
                 use_checkpoint=False, 
                 use_scale_shift_norm=False,
                 use_new_attention_order=False,
                 num_heads_upsample=-1,
                 transformer_depth=1,
                 context_dim=None, 
                ):
        super().__init__(self, )

        self.model_channels = model_channels
        self.in_channels = in_channels

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'


        time_embed_dim = model_channels * 4

        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )

        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        for level, mult in enumerate(channel_mult):
            layers = [
                ResBlock(
                    ch,
                    time_embed_dim,
                    dropout,
                    out_channels=mult * model_channels,
                    dims=dims,
                    use_checkpoint=use_checkpoint,
                    use_scale_shift_norm=use_scale_shift_norm,
                )
            ]
            ch = mult * model_channels                
            if num_head_channels == -1:
                dim_head = ch // num_heads
            else:
                num_heads = ch // num_head_channels
                dim_head = num_head_channels
            layers.append(
                CrossAttenBlock(
                    ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim
                )
            )
            self.input_blocks.append(TimestepEmbedSequential(*layers))
            self._feature_size += ch
            input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            ich = input_block_chans.pop()
            layers = [
                ResBlock(
                    ch + ich,
                    time_embed_dim,
                    dropout,
                    out_channels=model_channels * mult,
                    dims=dims,
                    use_checkpoint=use_checkpoint,
                    use_scale_shift_norm=use_scale_shift_norm,
                )
            ]
            ch = model_channels * mult
            if num_head_channels == -1:
                dim_head = ch // num_heads
            else:
                num_heads = ch // num_head_channels
                dim_head = num_head_channels
            layers.append(
                CrossAttenBlock(
                    ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim
                )
            )
            if level:
                out_ch = ch
                layers.append(
                    Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                )

            self.output_blocks.append(TimestepEmbedSequential(*layers))
            self._feature_size += ch
    
    def forward(self, quant, resi, lamda=None):
        
        if lamda:
            lamda_emb = timestep_embedding(lamda, self.model_channels, repeat_only=False)
            emb = self.time_embed(lamda_emb)
        else:
            lamda_emb = None
        
        hs = []
        for module in self.input_blocks:
            h = module(h, emb, resi)
            hs.append(h)
        # h = self.middle_block(h, emb, resi)
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb, resi)

        return h

