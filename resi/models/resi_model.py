from turtle import forward
import torch
import torch.nn as nn
import sys

from compressai.models.priors import CompressionModel
from compressai.entropy_models import EntropyBottleneck, GaussianConditional

sys.path.append("/root/home/codes/resi_compress/src/taming-transformers")
from taming.models.vqgan import VQModel
from taming.modules.diffusionmodules.model import Encoder, Decoder

from resi.models.resi_modules import FusionModel, ResiEncoder, ResiDecoder, HyperPriorModel


class ResiModel(CompressionModel):
    def __init__(self, 
                 ddconfig,
                 fusionconfig,
                ):
        super().__init__(self, entropy_bottleneck_channels=192)
        
        #TODO
        self.vqgan = VQModel()

        self.resi_encoder = ResiEncoder()
        self.hyper_model = HyperPriorModel()
        # self.resi_quant = 
        self.resi_decoder = ResiDecoder()

        self.fusion_model = FusionModel(**fusionconfig)

        self.decoder = Decoder(**ddconfig)
        
        self.gaussian_conditional = GaussianConditional(None)

    def forward(self, x):
        quant, emb_loss, info = self.vqgan.encode(x)

        z = self.resi_encoder(x, quant)

        h = self.hyper_model.encode(z)
        h_hat, h_likelihoods = self.entropy_bottleneck(h)
        latent_means, latent_scales = self.hyper_model.decode(h_hat).chunk(2, 1)

        z_hat, z_likelihoods = self.gaussian_conditional(z, latent_scales, means=latent_means)
        resi = self.resi_decoder(z_hat)

        y = self.fusion_model(quant, resi)

        x_hat = self.decoder(y)

        return {
            "x_hat": x_hat,
            "liklihoods": {"z": z_likelihoods, "h": h_likelihoods},
        }
    
    def encode(self, x):
        quant, emb_loss, info = self.vqgan.encode(x)

        z = self.resi_encoder(x, quant)

        h = self.resi_encoder.h_a(z)
        # z_hat = self.resi_quant(z)
        z_hat, z_likelihoods = self.entropy_bottleneck(h)
        
        return quant, z_hat

    def decode(self, quant, z_hat):
        resi = self.resi_decoder(z_hat)

        y = self.fusion_model(quant, resi)

        x_hat = self.decoder(y)

        return x_hat

