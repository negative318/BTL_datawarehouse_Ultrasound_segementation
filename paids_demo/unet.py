import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.blocks import UnetrBasicBlock, UnetrUpBlock, UnetOutBlock

# ==========================================
# 1. Helper Blocks
# ==========================================

class ConvNeXtBottleneck(nn.Module):
    """
    A modern, large-kernel bottleneck replacing ASPP. 
    Uses a 7x7 depthwise convolution for a dense global receptive field,
    followed by an inverted bottleneck with GELU.
    """
    def __init__(self, in_channels, out_channels, final_channels, kernel_size=7):
        super().__init__()
        
        # 1. Large Kernel Depthwise Convolution
        # groups=in_channels makes it depthwise, drastically saving parameters
        self.dwconv = nn.Conv2d(
            in_channels, in_channels, 
            kernel_size=kernel_size, 
            padding=kernel_size // 2, 
            groups=in_channels, 
            bias=False
        )
        
        # 2. Modern Normalization (LayerNorm is preferred over InstanceNorm here)
        self.norm = nn.LayerNorm(in_channels, eps=1e-6)
        
        # 3. Pointwise expansion (Inverted Bottleneck)
        # Expands channels to learn complex representations (like Swin MLP)
        self.pwconv1 = nn.Linear(in_channels, out_channels * 4)
        self.act = nn.GELU()
        
        # 4. Pointwise projection back to final channels
        self.pwconv2 = nn.Linear(out_channels * 4, final_channels)
        
        # 5. Skip connection handling
        if in_channels != final_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, final_channels, kernel_size=1, bias=False),
                nn.InstanceNorm2d(final_channels)
            )
        else:
            self.skip = nn.Identity()

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        residual = self.skip(x)
        
        # Depthwise Conv
        x = self.dwconv(x)
        
        # PyTorch LayerNorm and Linear expect channels last: (N, C, H, W) -> (N, H, W, C)
        x = x.permute(0, 2, 3, 1)
        
        # MLP Block
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        
        # Back to channels first: (N, H, W, C) -> (N, C, H, W)
        x = x.permute(0, 3, 1, 2)
        
        x = self.dropout(x)
        return residual + x

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        nn.init.zeros_(self.fc[2].weight)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResSkipBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm2d(out_channels)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm2d(out_channels)
        
        if in_channels != out_channels:
            self.identity = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.InstanceNorm2d(out_channels)
            )
        else:
            self.identity = nn.Identity()

    def forward(self, x):
        residual = self.identity(x)
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out += residual
        return self.relu(out)

class AttentionUpBlock(nn.Module):
    def __init__(self, dec_dim, enc_dim, size, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        
        # Q comes from Encoder, K/V come from Decoder
        self.q_proj = nn.Conv2d(enc_dim, enc_dim // 2, kernel_size=1, bias=False)
        self.k_proj = nn.Conv2d(dec_dim, enc_dim // 2, kernel_size=1, bias=False)
        self.v_proj = nn.Conv2d(dec_dim, enc_dim, kernel_size=1, bias=False)
        
        # Post-process: 3x3 Conv to mix heads, restore local 2D context, and match enc_dim
        self.out_proj = nn.Sequential(
            nn.Conv2d(enc_dim, enc_dim, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(enc_dim)
        )
        
        self.fusion = nn.Sequential(
            nn.Conv2d(enc_dim * 2, enc_dim, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(enc_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(enc_dim, enc_dim, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(enc_dim),
            nn.ReLU(inplace=True)
        )

        self.depth_embed = nn.Parameter(torch.zeros(1, enc_dim // 2, size, 1))
        self.width_embed_L = nn.Parameter(torch.zeros(1, enc_dim // 2, 1, size))
        self.width_embed_T = nn.Parameter(torch.zeros(1, enc_dim // 2, 1, size))
        nn.init.trunc_normal_(self.depth_embed, std = 0.02)
        nn.init.trunc_normal_(self.width_embed_L, std = 0.02)
        nn.init.trunc_normal_(self.width_embed_T, std = 0.02)
        
    def qkv_preprocessing(self, q, k, v):
        B, _, q_H, q_W = q.shape
        _, _, kv_H, kv_W = k.shape
        pos_L = self.depth_embed + self.width_embed_L
        pos_T = self.depth_embed + self.width_embed_T
        q_L, q_T = q.chunk(2)
        k_L, k_T = k.chunk(2)
        v_L, v_T = v.chunk(2)
        q_L = q_L + pos_L
        q_T = q_T + pos_T
        q = torch.cat((q_L,q_T),dim=0)
        q = q.view(B, self.num_heads, -1, q_H*q_W).transpose(-1, -2)
        k_L = k_L + F.interpolate(pos_L, size=(kv_H, kv_W), mode='bilinear', align_corners=False)
        k_T = k_T + F.interpolate(pos_T, size=(kv_H, kv_W), mode='bilinear', align_corners=False)
        k_L = k_L.view(B//2, self.num_heads, -1, kv_H*kv_W).transpose(-1,-2)
        k_T = k_T.view(B//2, self.num_heads, -1, kv_H*kv_W).transpose(-1,-2)
        k = torch.cat((k_L, k_T),dim=2)
        k = torch.cat((k,k),dim=0)
        v_L = v_L.view(B//2, self.num_heads, -1, kv_H*kv_W).transpose(-1,-2)
        v_T = v_T.view(B//2, self.num_heads, -1, kv_H*kv_W).transpose(-1,-2)
        v = torch.cat((v_L, v_T),dim=2)
        v = torch.cat((v,v),dim=0)
        return q, k, v
    
    def forward(self, dec, enc):
        B, enc_C, enc_H, enc_W = enc.shape
        _, _, dec_H, dec_W = dec.shape
        # 1. Project Q, K, V
        Q_proj = self.q_proj(enc) # (B, dec_dim // 2, enc_H, enc_W)
        K_proj = self.k_proj(dec) # (B, dec_dim // 2, dec_H, dec_W)
        V_proj = self.v_proj(dec) # (B, dec_dim, dec_H, dec_W)

        # 2. Query preprocessing
        Q, K, V = self.qkv_preprocessing(Q_proj, K_proj, V_proj)

        # 3. Memory-efficient cross attention
        attn_out = F.scaled_dot_product_attention(Q, K, V)

        # 4. Reshape back to the ENCODER's spatial grid
        attn_out = attn_out.transpose(-1, -2).contiguous().view(B, -1, enc_H, enc_W)

        # 5. Post-process to mix heads and align channels
        dec_upsampled = self.out_proj(attn_out)

        # 5. CONCATENATE instead of add (Isolates semantics vs spatial details)
        concat_features = torch.cat([enc, dec_upsampled], dim=1)
        
        # 6. Deep non-linear fusion
        return self.fusion(concat_features)
# ==========================================
# 2. UNET
# ==========================================

class UNET_Seg(nn.Module):
    def __init__(self, seg_num_classes: int, in_chans: int = 1):
        super().__init__()
        # --- UNETR BLOCKS ---
        spatial_dims = 2
        decode_feature_size = 32
        norm_name = "instance"

        self.encoder1 = UnetrBasicBlock(spatial_dims, in_chans, decode_feature_size, 3, 1, norm_name, res_block=True)
        self.encoder2 = UnetrBasicBlock(spatial_dims, decode_feature_size, decode_feature_size*2, 3, 2, norm_name, res_block=True)
        self.encoder3 = UnetrBasicBlock(spatial_dims, 2 * decode_feature_size, 4 * decode_feature_size, 3, 2, norm_name, res_block=True)
        self.encoder4 = UnetrBasicBlock(spatial_dims, 4 * decode_feature_size, 8 * decode_feature_size, 3, 2, norm_name, res_block=True)
        self.encoder5 = UnetrBasicBlock(spatial_dims, 8 * decode_feature_size, 16 * decode_feature_size, 3, 2, norm_name, res_block=True)
        self.encoder10 = UnetrBasicBlock(spatial_dims, 16 * decode_feature_size, 32 * decode_feature_size, 3, 2, norm_name, res_block=True)

        # --- Residual Skip Connections ---
        self.skip_res1 = ResSkipBlock(decode_feature_size, decode_feature_size)
        self.skip_res2 = ResSkipBlock(2 * decode_feature_size, 2 * decode_feature_size)
        self.skip_res3 = ResSkipBlock(4 * decode_feature_size, 4 * decode_feature_size)
        self.skip_res4 = ResSkipBlock(8 * decode_feature_size, 8 * decode_feature_size)
        self.skip_res5 = ResSkipBlock(16 * decode_feature_size, 16 * decode_feature_size)

        self.bottleneck = ConvNeXtBottleneck(
            in_channels=32 * decode_feature_size, 
            out_channels=256, 
            final_channels=32 * decode_feature_size,
            kernel_size=7 # You can experiment with 9 or 11 if you need even larger context
        )
        
        # --- DECODER ---
        self.decoder5 = AttentionUpBlock(32 * decode_feature_size, 16 * decode_feature_size, 16)
        self.decoder4 = AttentionUpBlock(16 * decode_feature_size, 8 * decode_feature_size, 32)
        self.decoder3 = UnetrUpBlock(spatial_dims, 8 * decode_feature_size, 4 * decode_feature_size, 3, 2, norm_name, res_block=True)
        self.decoder2 = UnetrUpBlock(spatial_dims, 4 * decode_feature_size, 2 * decode_feature_size, 3, 2, norm_name, res_block=True)
        self.decoder1 = UnetrUpBlock(spatial_dims, 2 * decode_feature_size, decode_feature_size, 3, 2, norm_name, res_block=True)

        self.out = UnetOutBlock(spatial_dims, decode_feature_size, seg_num_classes)
        self.bottleneck_dim = 32 * decode_feature_size
        self.decode_feature_size = decode_feature_size 

    def encode(self, x):
        enc0 = self.encoder1(x)
        enc1 = self.encoder2(enc0)
        enc2 = self.encoder3(enc1)
        enc3 = self.encoder4(enc2)
        enc4 = self.encoder5(enc3)
        dec4 = self.encoder10(enc4)
        dec4 = self.bottleneck(dec4)
        #assert False, (enc0.shape, enc1.shape, enc2.shape, enc3.shape, enc4.shape, dec4.shape)
        return enc0, enc1, enc2, enc3, enc4, dec4

    def decode(self, enc0, enc1, enc2, enc3, enc4, dec4):
        enc4_skip = self.skip_res5(enc4)
        enc3_skip = self.skip_res4(enc3)
        enc2_skip = self.skip_res3(enc2)
        enc1_skip = self.skip_res2(enc1)
        enc0_skip = self.skip_res1(enc0)

        # Level 5
        dec3 = self.decoder5(dec4, enc4_skip)

        # Level 4
        dec2 = self.decoder4(dec3, enc3_skip)

        # Level 3
        dec1 = self.decoder3(dec2, enc2_skip)

        # Level 2
        dec0 = self.decoder2(dec1, enc1_skip) 

        # Level 1
        out = self.decoder1(dec0, enc0_skip)
        
        logits = self.out(out)
        return logits

# ==========================================
# 3. Enhanced Multi-Scale Orthogonal ViT
# ==========================================

class MultiScaleOrthogonalViT(nn.Module):
    def __init__(
        self, 
        scale_dims: list,       # Input channels e.g. [192, 320, 1088]
        hidden_dim: int,        # 256
        num_classes: int, 
        bottleneck_sizes: list, # Spatial resolutions e.g. [64, 32, 8]
        num_layers=2, 
        nhead=4,                # 4 Heads to save memory
        num_registers=4
    ):
        super().__init__()
        self.num_registers = num_registers
        self.hidden_dim = hidden_dim
        
        # 1. Projections
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_d, hidden_dim, kernel_size=3, padding = 1, bias=False),
                nn.InstanceNorm2d(hidden_dim),
                nn.ReLU(inplace=True)
            ) for in_d in scale_dims
        ])
        
        # 2. Scale Tokens
        self.scale_tokens = nn.Parameter(torch.zeros(1, len(scale_dims), hidden_dim))
        
        # 3. Orthogonal Embeddings (ParameterList for correct registration)
        self.emb_depth_list   = nn.ParameterList()
        self.emb_width_L_list = nn.ParameterList()
        self.emb_width_T_list = nn.ParameterList()
        
        for size in bottleneck_sizes:
            self.emb_depth_list.append(nn.Parameter(torch.zeros(1, hidden_dim, size, 1)))
            self.emb_width_L_list.append(nn.Parameter(torch.zeros(1, hidden_dim, 1, size)))
            self.emb_width_T_list.append(nn.Parameter(torch.zeros(1, hidden_dim, 1, size)))

        # 4. Transformer
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        if num_registers > 0:
            self.registers = nn.Parameter(torch.zeros(1, num_registers, hidden_dim))
            
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead, dim_feedforward=hidden_dim*2, 
            dropout=0.1, activation='gelu', batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_classes)
        )
        
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        if self.num_registers > 0:
            nn.init.trunc_normal_(self.registers, std=0.02)
        nn.init.trunc_normal_(self.scale_tokens, std=0.02)
        for p in self.emb_depth_list: nn.init.trunc_normal_(p, std=0.02)
        for p in self.emb_width_L_list: nn.init.trunc_normal_(p, std=0.02)
        for p in self.emb_width_T_list: nn.init.trunc_normal_(p, std=0.02)

    def forward(self, features_L: list, features_T: list):
        B = features_L[0].shape[0]
        all_tokens = []
        
        # Iterate through scales
        for i, (feat_L, feat_T) in enumerate(zip(features_L, features_T)):
            # Project to hidden_dim
            x_l = self.projections[i](feat_L) 
            x_t = self.projections[i](feat_T)
            
            # --- Dynamic Orthogonal Embedding ---
            # Retrieve learned parameters (1, Base_S, H) -> (1, H, Base_S)
            base_depth = self.emb_depth_list[i]
            base_width_L = self.emb_width_L_list[i]
            base_width_T = self.emb_width_T_list[i]

            pos_L = base_depth + base_width_L
            pos_T = base_depth + base_width_T

            # Add Embeddings
            x_l = x_l + pos_L
            x_t = x_t + pos_T
            
            # Flatten spatial dims: B, H, S*S -> B, S*S, H
            flat_l = x_l.flatten(2).transpose(1, 2)
            flat_t = x_t.flatten(2).transpose(1, 2)
            
            tokens = torch.cat([flat_l, flat_t], dim=1) 
            scale_embed = self.scale_tokens[:, i:i+1, :] 
            tokens = tokens + scale_embed
            
            all_tokens.append(tokens)
            
        x = torch.cat(all_tokens, dim=1)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        if self.num_registers > 0:
            regs = self.registers.expand(B, -1, -1)
            x = torch.cat((cls_tokens, regs, x), dim=1)
        else:
            x = torch.cat((cls_tokens, x), dim=1)
            
        x = self.transformer(x)
        return self.head(x[:, 0])
# ==========================================
# 4. Main Model Wrapper
# ==========================================

class UNetTwoView(nn.Module):
    def __init__(
        self,
        in_chns: int,
        seg_class_num: int,
        cls_class_num: int,
        img_size: int = 256
    ):
        super().__init__()

        self.seg_net = UNET_Seg(
            seg_num_classes=seg_class_num,
            in_chans=in_chns,
        )

        # --- Dimensions Logic ---
        # enc2 (Stage 2) -> 128 channels
        # enc3 (Stage 3) -> 256 channels
        # dec4 (Stage 5) -> 1024 channels
        base_dim_s1 = 128
        base_dim_s2 = 256
        base_dim_s3 = 1024
        
        self.seg_emb_dim = 64
        
        dim_scale1 = base_dim_s1 + self.seg_emb_dim
        dim_scale2 = base_dim_s2 + self.seg_emb_dim
        dim_scale3 = base_dim_s3 + self.seg_emb_dim

        # --- SE Blocks ---
        self.se_blocks = nn.ModuleList([
            SEBlock(channel=dim_scale1, reduction=16),
            SEBlock(channel=dim_scale2, reduction=16),
            SEBlock(channel=dim_scale3, reduction=16)
        ])

        # --- Resolution Logic (Patch Size 2) ---
        # 256 -> 64 (Scale 1), 32 (Scale 2), 8 (Scale 3)
        s1 = img_size // 4   # 64
        s2 = img_size // 8   # 32
        s3 = img_size // 32  # 8

        self.cls_classifier = MultiScaleOrthogonalViT(
            scale_dims=[dim_scale1, dim_scale2, dim_scale3],
            hidden_dim=256,         # <--- Memory optimized
            num_classes=cls_class_num,
            bottleneck_sizes=[s1, s2, s3],
            num_layers=2,
            nhead=4,                # <--- Memory optimized
            num_registers=4
        )

        self.fp_dropout = nn.Dropout2d(0.5)
        self.seg_embedding = nn.Embedding(3, self.seg_emb_dim) 

    def _seg_embed(self, seg_map):
        hard_seg_map = torch.argmax(seg_map, dim=1) 
        hard_seg_embed = self.seg_embedding(hard_seg_map) 
        hard_seg_embed = hard_seg_embed.permute(0, 3, 1, 2) 
        return hard_seg_embed

    def _fuse_feature(self, feature, seg_emb, scale_idx):
        seg_resized = F.interpolate(seg_emb, size=feature.shape[-2:], mode='bilinear', align_corners=False)
        fused = torch.cat((feature, seg_resized), dim=1)
        fused = self.se_blocks[scale_idx](fused)
        return fused

    def forward(self, x_long, x_trans, need_fp: bool = False):
        x = torch.cat((x_long, x_trans), dim=0)

        # 1. ENCODE
        enc0, enc1, enc2, enc3, enc4, dec4 = self.seg_net.encode(x)
        
        if need_fp:
            # Training with Feature Perturbation
            p_enc0 = torch.cat([enc0, self.fp_dropout(enc0)], dim=0)
            p_enc1 = torch.cat([enc1, self.fp_dropout(enc1)], dim=0)
            p_enc2 = torch.cat([enc2, self.fp_dropout(enc2)], dim=0)
            p_enc3 = torch.cat([enc3, self.fp_dropout(enc3)], dim=0)
            p_enc4 = torch.cat([enc4, self.fp_dropout(enc4)], dim=0)
            p_dec4 = torch.cat([dec4, self.fp_dropout(dec4)], dim=0)

            seg_logits = self.seg_net.decode(p_enc0, p_enc1, p_enc2, p_enc3, p_enc4, p_dec4)
            
            (seg_logits_LT, seg_logits_fp_LT) = seg_logits.chunk(2)
            (seg_logits_L, seg_logits_T) = seg_logits_LT.chunk(2)
            (seg_logits_fp_L, seg_logits_fp_T) = seg_logits_fp_LT.chunk(2)

            seg_emb_clean = self._seg_embed(seg_logits_LT)
            seg_emb_fp    = self._seg_embed(seg_logits_fp_LT)
            seg_emb_total = torch.cat([seg_emb_clean, seg_emb_fp], dim=0) 

            feat_s1 = self._fuse_feature(p_enc2, seg_emb_total, scale_idx=0)
            feat_s2 = self._fuse_feature(p_enc3, seg_emb_total, scale_idx=1)
            feat_s3 = self._fuse_feature(p_dec4, seg_emb_total, scale_idx=2)

            def split_feats(tensor):
                clean_lt, fp_lt = tensor.chunk(2)
                c_l, c_t = clean_lt.chunk(2)
                f_l, f_t = fp_lt.chunk(2)
                return c_l, c_t, f_l, f_t

            c_l_s1, c_t_s1, f_l_s1, f_t_s1 = split_feats(feat_s1)
            c_l_s2, c_t_s2, f_l_s2, f_t_s2 = split_feats(feat_s2)
            c_l_s3, c_t_s3, f_l_s3, f_t_s3 = split_feats(feat_s3)

            cls_logits = self.cls_classifier(
                features_L=[c_l_s1, c_l_s2, c_l_s3],
                features_T=[c_t_s1, c_t_s2, c_t_s3]
            )
            
            cls_logits_fp = self.cls_classifier(
                features_L=[f_l_s1, f_l_s2, f_l_s3],
                features_T=[f_t_s1, f_t_s2, f_t_s3]
            )

            return (seg_logits_L, seg_logits_fp_L), (seg_logits_T, seg_logits_fp_T), (cls_logits, cls_logits_fp)

        # Normal Inference
        seg_logits = self.seg_net.decode(enc0, enc1, enc2, enc3, enc4, dec4)
        (seg_logits_L, seg_logits_T) = seg_logits.chunk(2)
        
        seg_emb = self._seg_embed(seg_logits)
        
        feat_s1 = self._fuse_feature(enc2, seg_emb, scale_idx=0)
        feat_s2 = self._fuse_feature(enc3, seg_emb, scale_idx=1)
        feat_s3 = self._fuse_feature(dec4, seg_emb, scale_idx=2)
        
        def split_batch(t): return t.chunk(2)

        l_s1, t_s1 = split_batch(feat_s1)
        l_s2, t_s2 = split_batch(feat_s2)
        l_s3, t_s3 = split_batch(feat_s3)

        cls_logits = self.cls_classifier(
            features_L=[l_s1, l_s2, l_s3],
            features_T=[t_s1, t_s2, t_s3]
        )

        return seg_logits_L, seg_logits_T, cls_logits