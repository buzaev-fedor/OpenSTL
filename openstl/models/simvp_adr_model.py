import torch
import torch.nn as nn
import torch.nn.functional as F

from openstl.modules import (ConvSC, ConvNeXtSubBlock, ConvMixerSubBlock, GASubBlock, gInception_ST,
                             HorNetSubBlock, MLPMixerSubBlock, MogaSubBlock, PoolFormerSubBlock,
                             SwinSubBlock, UniformerSubBlock, VANSubBlock, ViTSubBlock, TAUSubBlock)


class SimVP_ADR_Model(nn.Module):
    """SimVP-ADR Model

    Integration of SimVP (Simpler yet Better Video Prediction) with deepADRnet
    (Advection-Diffusion-Reaction Network) for spatiotemporal prediction.
    """

    def __init__(self, in_shape, hid_S=16, hid_T=256, N_S=4, N_T=4, model_type='gSTA',
                 mlp_ratio=8., drop=0.0, drop_path=0.0, spatio_kernel_enc=3,
                 spatio_kernel_dec=3, adr_layers=4, adr_hid_dim=128, 
                 act_inplace=True, device='cuda', **kwargs):
        super(SimVP_ADR_Model, self).__init__()
        
        T, C, H, W = in_shape  # T is pre_seq_length
        H, W = int(H / 2**(N_S/2)), int(W / 2**(N_S/2))  # downsample 1 / 2**(N_S/2)
        act_inplace = False

        
        # SimVP components
        self.enc = Encoder(C, hid_S, N_S, spatio_kernel_enc, act_inplace=act_inplace)
        self.dec = Decoder(hid_S, C, N_S, spatio_kernel_dec, act_inplace=act_inplace)

        # Choose middleware type
        model_type = 'gsta' if model_type is None else model_type.lower()
        if model_type == 'incepu':
            self.hid = MidIncepNet(T*hid_S, hid_T, N_T)
        else:
            self.hid = MidMetaNet(T*hid_S, hid_T, N_T,
                input_resolution=(H, W), model_type=model_type,
                mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path)
        
        # deepADRnet components
        self.use_adr = adr_layers > 0
        
        if self.use_adr:
            self.adr_processor = ADRProcessor(
                in_channels=hid_S,
                hid_channels=hid_S,
                nlayers=adr_layers,
                imsz=[H, W],
                device=device
            )


    def forward(self, x_raw, **kwargs):
        B, T, C, H, W = x_raw.shape
        x = x_raw.reshape(B*T, C, H, W)
        
        embed, skip = self.enc(x)
        _, C_, H_, W_ = embed.shape

        # Process with MidMetaNet
        z = embed.reshape(B, T, C_, H_, W_)
        hid = self.hid(z)
        
        # Apply ADR processing if enabled

        if self.use_adr:
            hid_shape = hid.shape
            hid_adr = torch.zeros_like(hid)

            for t in range(T):
                # Time parameter for ADR network
                t_tensor = torch.ones(B, device=hid.device) * t
                
                # Reshape for ADR processing
                hid_flat = hid[:, t, :, :, :]
            
                # Apply ADR processing
                hid_processed = self.adr_processor(hid_flat, t_tensor)
                hid_adr[:, t, :, :, :] = hid_processed
            
            # Reshape back
            hid = hid + hid_adr


        # if self.use_adr:
        #     hid_shape = hid.shape
        #     # Time parameter for ADR network
        #     t = torch.zeros(B, device=hid.device)
            
        #     # Reshape for ADR processing
        #     hid_flat = hid.reshape(B, T*C_, H_, W_)
            
        #     # Apply ADR processing
        #     hid_processed = self.adr_processor(hid_flat, t)
            
        #     # Reshape back
        #     hid = hid_processed.reshape(hid_shape)
        
        hid = hid.reshape(B*T, C_, H_, W_)

        # Decode
        Y = self.dec(hid, skip)
        Y = Y.reshape(B, T, C, H, W)
        
        return Y


def sampling_generator(N, reverse=False):
    samplings = [False, True] * (N // 2)
    if reverse: return list(reversed(samplings[:N]))
    else: return samplings[:N]


class Encoder(nn.Module):
    """3D Encoder for SimVP-ADR"""

    def __init__(self, C_in, C_hid, N_S, spatio_kernel, act_inplace=True):
        samplings = sampling_generator(N_S)
        super(Encoder, self).__init__()
        self.enc = nn.Sequential(
              ConvSC(C_in, C_hid, spatio_kernel, downsampling=samplings[0],
                     act_inplace=act_inplace),
            *[ConvSC(C_hid, C_hid, spatio_kernel, downsampling=s,
                     act_inplace=act_inplace) for s in samplings[1:]]
        )

    def forward(self, x):  # B*T, C, H, W
        enc1 = self.enc[0](x)
        latent = enc1
        for i in range(1, len(self.enc)):
            latent = self.enc[i](latent)
        return latent, enc1


class Decoder(nn.Module):
    """3D Decoder for SimVP-ADR"""

    def __init__(self, C_hid, C_out, N_S, spatio_kernel, act_inplace=True):
        samplings = sampling_generator(N_S, reverse=True)
        super(Decoder, self).__init__()
        self.dec = nn.Sequential(
            *[ConvSC(C_hid, C_hid, spatio_kernel, upsampling=s,
                     act_inplace=act_inplace) for s in samplings[:-1]],
              ConvSC(C_hid, C_hid, spatio_kernel, upsampling=samplings[-1],
                     act_inplace=act_inplace)
        )
        self.readout = nn.Conv2d(C_hid, C_out, 1)

    def forward(self, hid, enc1=None):
        for i in range(0, len(self.dec)-1):
            hid = self.dec[i](hid)
        
        Y = self.dec[-1](hid + enc1)
        Y = self.readout(Y)
        return Y


class MidIncepNet(nn.Module):
    """The hidden Translator of IncepNet for SimVP"""

    def __init__(self, channel_in, channel_hid, N2, incep_ker=[3,5,7,11], groups=8, **kwargs):
        super(MidIncepNet, self).__init__()
        assert N2 >= 2 and len(incep_ker) > 1
        self.N2 = N2
        enc_layers = [gInception_ST(
            channel_in, channel_hid//2, channel_hid, incep_ker=incep_ker, groups=groups)]
        for i in range(1,N2-1):
            enc_layers.append(
                gInception_ST(channel_hid, channel_hid//2, channel_hid,
                              incep_ker=incep_ker, groups=groups))
        enc_layers.append(
                gInception_ST(channel_hid, channel_hid//2, channel_hid,
                              incep_ker=incep_ker, groups=groups))
        dec_layers = [
                gInception_ST(channel_hid, channel_hid//2, channel_hid,
                              incep_ker=incep_ker, groups=groups)]
        for i in range(1,N2-1):
            dec_layers.append(
                gInception_ST(2*channel_hid, channel_hid//2, channel_hid,
                              incep_ker=incep_ker, groups=groups))
        dec_layers.append(
                gInception_ST(2*channel_hid, channel_hid//2, channel_in,
                              incep_ker=incep_ker, groups=groups))

        self.enc = nn.Sequential(*enc_layers)
        self.dec = nn.Sequential(*dec_layers)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B, T*C, H, W)

        # encoder
        skips = []
        z = x
        for i in range(self.N2):
            z = self.enc[i](z)
            if i < self.N2-1:
                skips.append(z)
        
        # decoder
        z = self.dec[0](z)
        for i in range(1,self.N2):
            z = self.dec[i](torch.cat([z, skips[-i]], dim=1))

        y = z.reshape(B, T, C, H, W)
        return y


class MetaBlock(nn.Module):
    """The hidden Translator of MetaFormer for SimVP-ADR"""

    def __init__(self, in_channels, out_channels, input_resolution=None, model_type=None,
                 mlp_ratio=8., drop=0.0, drop_path=0.0, layer_i=0):
        super(MetaBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        model_type = model_type.lower() if model_type is not None else 'gsta'
        
        print(f"Creating MetaBlock with type: {model_type}")

        if model_type == 'gsta':
            self.block = GASubBlock(
                in_channels, kernel_size=21, mlp_ratio=mlp_ratio,
                drop=drop, drop_path=drop_path, act_layer=nn.GELU)
        elif model_type == 'convmixer':
            self.block = ConvMixerSubBlock(in_channels, kernel_size=11, activation=nn.GELU)
        elif model_type == 'convnext':
            self.block = ConvNeXtSubBlock(
                in_channels, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path)
        elif model_type == 'hornet':
            self.block = HorNetSubBlock(in_channels, mlp_ratio=mlp_ratio, drop_path=drop_path)
        elif model_type in ['mlp', 'mlpmixer']:
            self.block = MLPMixerSubBlock(
                in_channels, input_resolution, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path)
        elif model_type in ['moga', 'moganet']:
            self.block = MogaSubBlock(
                in_channels, mlp_ratio=mlp_ratio, drop_rate=drop, drop_path_rate=drop_path)
        elif model_type == 'poolformer':
            self.block = PoolFormerSubBlock(
                in_channels, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path)
        elif model_type == 'swin':
            self.block = SwinSubBlock(
                in_channels, input_resolution, layer_i=layer_i, mlp_ratio=mlp_ratio,
                drop=drop, drop_path=drop_path)
        elif model_type == 'uniformer':
            block_type = 'MHSA' if in_channels == out_channels and layer_i > 0 else 'Conv'
            self.block = UniformerSubBlock(
                in_channels, mlp_ratio=mlp_ratio, drop=drop,
                drop_path=drop_path, block_type=block_type)
        elif model_type == 'van':
            self.block = VANSubBlock(
                in_channels, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path, act_layer=nn.GELU)
        elif model_type == 'vit':
            self.block = ViTSubBlock(
                in_channels, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path)
        elif model_type == 'tau':
            self.block = TAUSubBlock(
                in_channels, kernel_size=21, mlp_ratio=mlp_ratio,
                drop=drop, drop_path=drop_path, act_layer=nn.GELU)
        else:
            assert False and "Invalid model_type in SimVP-ADR"

        if in_channels != out_channels:
            self.reduction = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        z = self.block(x)
        
        if self.in_channels == self.out_channels:
            return z
        else:
            out = self.reduction(z)
            return out


class MidMetaNet(nn.Module):
    """The hidden Translator of MetaFormer for SimVP-ADR"""

    def __init__(self, channel_in, channel_hid, N2,
                 input_resolution=None, model_type=None,
                 mlp_ratio=4., drop=0.0, drop_path=0.1):
        super(MidMetaNet, self).__init__()
        assert N2 >= 2 and mlp_ratio > 1
        self.N2 = N2
        
        dpr = [  # stochastic depth decay rule
            x.item() for x in torch.linspace(1e-2, drop_path, self.N2)]

        # downsample
        enc_layers = [MetaBlock(
            channel_in, channel_hid, input_resolution, model_type,
            mlp_ratio, drop, drop_path=dpr[0], layer_i=0)]
        
        # middle layers
        for i in range(1, N2-1):
            enc_layers.append(MetaBlock(
                channel_hid, channel_hid, input_resolution, model_type,
                mlp_ratio, drop, drop_path=dpr[i], layer_i=i))
        
        # upsample
        enc_layers.append(MetaBlock(
            channel_hid, channel_in, input_resolution, model_type,
            mlp_ratio, drop, drop_path=drop_path, layer_i=N2-1))
        
        self.enc = nn.Sequential(*enc_layers)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B, T*C, H, W)

        z = x
        for i in range(self.N2):
            z = self.enc[i](z)

        y = z.reshape(B, T, C, H, W)
        return y


# ======= deepADRnet components =======

def CLP(dim_in, dim_out, shape=[64, 64], kernel_size=[3, 3]):
    """Convolution-LayerNorm-SiLU block from deepADRnet"""
    return nn.Sequential(
        nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, padding=kernel_size[0]//2),
        nn.LayerNorm(shape),
        nn.SiLU(),
        nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size, padding=kernel_size[0]//2)
    )


class ColorPreservingAdvection(nn.Module):
    """Color preserving advection module from deepADRnet"""
    
    def __init__(self, shape, device='cuda'):
        super(ColorPreservingAdvection, self).__init__()
        
        grid_h, grid_w = shape[0], shape[1]
        y, x = torch.meshgrid(torch.linspace(-1, 1, grid_h), torch.linspace(-1, 1, grid_w))
        self.grid = torch.stack((x, y), dim=-1).unsqueeze(0).unsqueeze(0)

    def forward(self, T, U, V):
        UV = torch.stack((U, V), dim=-1)
        grid = self.grid.to(T.device)
        transformation_grid = grid + UV
        Th = F.grid_sample(T, transformation_grid.squeeze(1), align_corners=True)
        return Th


class AdvectionColorBlock(nn.Module):
    """Advection color block from deepADRnet"""
    
    def __init__(self, channels, mesh_size, device='cuda'):
        super(AdvectionColorBlock, self).__init__()
        
        self.Adv = ColorPreservingAdvection(mesh_size, device)
        
        self.ConvU1 = nn.Conv2d(channels, channels, kernel_size=[3, 3], padding=1)
        self.LNU1 = nn.LayerNorm(normalized_shape=mesh_size)
        self.ConvU2 = nn.Conv2d(channels, channels, kernel_size=[3, 3], padding=1, bias=False)
        self.ConvU2.weight = nn.Parameter(1e-4*torch.randn(channels, channels, 3, 3))

        self.TimeEmbedU = nn.Parameter(1e-4*torch.randn(1, channels, mesh_size[0], mesh_size[1]))
        self.TNU = CLP(channels, channels, mesh_size)
        
        self.ConvV1 = nn.Conv2d(channels, channels, kernel_size=[3, 3], padding=1)
        self.LNV1 = nn.LayerNorm(normalized_shape=mesh_size)
        self.ConvV2 = nn.Conv2d(channels, channels, kernel_size=[3, 3], padding=1, bias=False)
        self.ConvV2.weight = nn.Parameter(1e-4*torch.randn(channels, channels, 3, 3))

        self.TimeEmbedV = nn.Parameter(1e-4*torch.randn(1, channels, mesh_size[0], mesh_size[1]))
        self.TNV = CLP(channels, channels, mesh_size)
        
    def forward(self, x, t):
        nw, nh = x.shape[3], x.shape[2]
        
        teU = t.reshape([-1, 1, 1, 1])*self.TimeEmbedU
        teU = self.TNU(teU) 
        teV = t.reshape([-1, 1, 1, 1])*self.TimeEmbedV
        teV = self.TNV(teV) 
        
        U = self.ConvU1(x) + teU
        U = self.LNU1(U)
        U = F.silu(U)
        U = self.ConvU2(U)
        
        V = self.ConvV1(x) + teV
        V = self.LNV1(V)
        V = F.silu(V)
        V = self.ConvV2(V)
        
        U, V = U/nw, V/nw

        xr = x.reshape(x.shape[0]*x.shape[1], 1, x.shape[2], x.shape[3])
        Ur = U.reshape(x.shape[0]*x.shape[1], 1, x.shape[2], x.shape[3])
        Vr = V.reshape(x.shape[0]*x.shape[1], 1, x.shape[2], x.shape[3])
        
        # Color Conserving Push Forward Operation: Advects pixels in xr by Ur and Vr along x and y axis respectively
        xr = self.Adv(xr, Ur, Vr)
        
        x = xr.reshape(x.shape)
        return x


class ADRProcessor(nn.Module):
    """ADR Processor combining blocks from deepADRnet"""
    
    def __init__(self, in_channels, hid_channels=128, nlayers=4, imsz=[64, 64], device='cuda'):
        super(ADRProcessor, self).__init__()
        
        self.nlayers = nlayers
        # self.Open = CLP(in_channels, hid_channels, imsz)
        
        self.Adv = nn.ModuleList()
        self.DR = nn.ModuleList()
        
        for i in range(nlayers):
            # Color conserving advection layer
            Advi = AdvectionColorBlock(hid_channels, imsz, device)
            # Diffusion and Reaction Layer: Double convolution layer with nonlinearity
            DRi = CLP(hid_channels, hid_channels, imsz, kernel_size=[5, 5])
            
            self.Adv.append(Advi)
            self.DR.append(DRi)
       
        # self.Close = nn.Conv2d(hid_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.h = 1/imsz[0]
                
    def forward(self, x, t):
        
        # Increase the dimensionality
        # z = self.Open(x)
        
        # Each residual network layer learns sequential advection, diffusion and reaction
        for i in range(self.nlayers):
            # Advection Layer: Learns the advection of color pixels at higher dimension
            dz = self.Adv[i](x, t)
            # dz = self.Adv[i](z, t)
            
            # Learns the diffusion and reaction of color pixels at higher dimension
            dz = self.DR[i](dz)
            
            # Residual Connection
            x = x + self.h*dz
        
        # Decrease the dimensionality
        # x = self.Close(z)
        
        return x 