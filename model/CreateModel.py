import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# Small utility blocks
# -------------------------
class ConvGNAct(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1, groups=32):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=False)
        gn_groups = min(groups, out_ch)
        self.gn = nn.GroupNorm(gn_groups, out_ch)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.act(self.gn(self.conv(x)))

class Residual(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            ConvGNAct(ch, ch, kernel=3, padding=1),
            ConvGNAct(ch, ch, kernel=3, padding=1)
        )
    def forward(self, x):
        return x + self.block(x)

# -------------------------
# Encoder: early fusion of two frames (6 channels)
# -------------------------
class EarlyFusionEncoder(nn.Module):
    def __init__(self, feat_dim=256):
        super().__init__()
        # Keep it compact but expressive
        self.stem = nn.Sequential(
            ConvGNAct(6, 32, kernel=7, stride=2, padding=3),  # (B,32,H/2,W/2)
            Residual(32),
            ConvGNAct(32, 64, stride=2)                       # (B,64,H/4,W/4)
        )
        self.layer1 = nn.Sequential(
            Residual(64),
            ConvGNAct(64, 128, stride=2)                      # (B,128,H/8,W/8)
        )
        self.layer2 = nn.Sequential(
            Residual(128),
            ConvGNAct(128, feat_dim, stride=2)                # (B,feat_dim,H/16,W/16)
        )
        # small projection before decoder attention
        self.proj = nn.Conv2d(feat_dim, feat_dim, kernel_size=1, bias=False)

    def forward(self, img_t, img_tm1):
        # expect inputs shape (B,3,H,W)
        x = torch.cat([img_t, img_tm1], dim=1)  # (B,6,H,W)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        fmap = self.proj(x)  # (B, feat_dim, Hf, Wf)
        # Add vertical positional encoding
        H, W = fmap.shape[2:]
        yCoords = torch.zeros(1, 1, H, 1, device=fmap.device)
        usefulStart = 0.35  # 35% from top is useless
        usefulEnd = 0.84    # 84% from top (100% - 16% bottom)
        for i in range(H):
            normPos = i / (H - 1)
            if normPos < usefulStart:
                y = -1.0
            elif normPos > usefulEnd:
                y = 1.0
            else:
                y = (normPos - usefulStart) / (usefulEnd - usefulStart) * 2 - 1
            yCoords[0, 0, i, 0] = y
        fmap = torch.cat([fmap, yCoords.expand(fmap.size(0), 1, H, W)], dim=1)
        return fmap

# -------------------------
# Attention used by decoder (computes spatial attention conditioned on decoder state)
# -------------------------
class SpatialAttentionDecoder(nn.Module):
    def __init__(self, feat_dim, hidden_dim):
        super().__init__()
        self.key_conv = nn.Conv2d(feat_dim, feat_dim, 1, bias=False)
        self.value_conv = nn.Conv2d(feat_dim, feat_dim, 1, bias=False)
        self.query_fc = nn.Linear(hidden_dim, feat_dim)
        # scale when computing dot-product
        self.scale = feat_dim ** -0.5

    def forward(self, feat_map, query):
        # feat_map: (B, C, H, W), query: (B, hidden_dim)
        B, C, H, W = feat_map.shape
        keys = self.key_conv(feat_map).view(B, C, -1).permute(0, 2, 1)   # (B, S, C)
        vals = self.value_conv(feat_map).view(B, C, -1).permute(0, 2, 1)   # (B, S, C)
        q = self.query_fc(query).unsqueeze(1)                             # (B,1,C)
        attn_logits = torch.bmm(q, keys.permute(0,2,1)) * self.scale      # (B,1,S)
        attn = torch.softmax(attn_logits, dim=-1)                         # (B,1,S)
        context = torch.bmm(attn, vals).squeeze(1)                        # (B,C)
        attn_map = attn.view(B, 1, H, W)
        return context, attn_map

# -------------------------
# Decoder: autoregressive GRUCell that attends to image features each step
# -------------------------
class AttentiveGRUDecoder(nn.Module):
    def __init__(self, feat_dim=256, hidden_dim=256, pred_steps=12):
        super().__init__()
        self.pred_steps = pred_steps
        self.hidden_dim = hidden_dim
        self.attn = SpatialAttentionDecoder(feat_dim, hidden_dim)
        # map pooled features -> initial h/c
        self.init_h = nn.Linear(feat_dim, hidden_dim)
        self.init_c = nn.Linear(feat_dim, hidden_dim)
        # GRUCell input: prev_xy (2) concatenated with context (feat_dim)
        self.grucell = nn.GRUCell(input_size=2 + feat_dim, hidden_size=hidden_dim)
        self.out_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 2)   # predict delta x,y for next step
        )

    def forward(self, feat_map, teacher_forcing=False, gt_traj=None, tf_ratio=0.9):
        # feat_map: (B, C, H, W)
        B = feat_map.size(0)
        device = feat_map.device
        pooled = F.adaptive_avg_pool2d(feat_map, (1,1)).view(B, -1)   # (B, feat_dim)
        h = self.init_h(pooled)                                      # (B, hidden_dim)
        c = self.init_c(pooled)                                      # (B, hidden_dim) - unused but kept for API parity
        prev_xy = torch.zeros(B, 2, device=device)                   # start at origin in vehicle frame
        preds = []
        attn_maps = []
        for t in range(self.pred_steps):
            context, attn_map = self.attn(feat_map, h)               # (B, feat_dim), (B,1,H,W)
            gru_in = torch.cat([prev_xy, context], dim=1)           # (B, 2 + feat_dim)
            h = self.grucell(gru_in, h)                             # (B, hidden_dim)
            delta = self.out_head(h)                                # (B, 2)
            next_xy = prev_xy + delta                               # absolute position (accumulated)
            preds.append(next_xy.unsqueeze(1))
            attn_maps.append(attn_map)
            # teacher forcing option (if gt_traj provided)
            if teacher_forcing and gt_traj is not None and torch.rand(1).item() < tf_ratio:
                prev_xy = gt_traj[:, t, :].detach()                 # feed GT
            else:
                prev_xy = next_xy.detach()
        preds = torch.cat(preds, dim=1)  # (B, T, 2)
        return preds, attn_maps

# -------------------------
# Full model wrapper
# -------------------------
class TrajectoryModel(nn.Module):
    def __init__(self, featDim=256, hiddenDim=256, predSteps=12, useAuxDyn=False, intervalSeconds=0.25):
        super().__init__()
        self.encoder = EarlyFusionEncoder(feat_dim=featDim)
        self.decoder = AttentiveGRUDecoder(feat_dim=featDim + 1, hidden_dim=hiddenDim, pred_steps=predSteps)
        self.use_aux_dyn = useAuxDyn
        if useAuxDyn:
            self.aux = nn.Sequential(
                nn.AdaptiveAvgPool2d((1,1)),
                nn.Flatten(),
                nn.Linear(featDim + 1, 128),
                nn.ReLU(),
                nn.Linear(128, 2)   # predict speed, accel (auxiliary only)
            )

        self.name = "TrajectoryModel_EarlyFusion_AttentiveGRU_V2"

        # Input and output specs
        self.inputSpec = {
            'image_size': (360, 640),  # (height, width)
            'temporal_delay_seconds': 0.1  # Time delay between previous and current images
        }
        self.outputSpec = {
            'num_vectors': predSteps,  # Number of predicted trajectory vectors
            'intervalSeconds': intervalSeconds  # Time interval between each predicted vector
        }

    def forward(self, img_t, img_tm1, gt_traj=None, teacher_forcing=False, tf_ratio=0.9):
        fmap = self.encoder(img_t, img_tm1)
        preds, attn_maps = self.decoder(fmap, teacher_forcing=teacher_forcing, gt_traj=gt_traj, tf_ratio=tf_ratio)
        aux = None
        if self.use_aux_dyn:
            aux = self.aux(fmap)
        return preds, aux, attn_maps

# -------------------------
# Quick smoke test & param count
# -------------------------
if __name__ == "__main__":
    batchSize = 2
    dummyImg1 = torch.randn(batchSize, 3, 360, 640)
    dummyImg2 = torch.randn(batchSize, 3, 360, 640)
    model = TrajectoryModel(featDim=512, hiddenDim=512, predSteps=12, useAuxDyn=True)
    model.eval()
    
    with torch.no_grad():
        preds, aux, attn = model(dummyImg1, dummyImg2)
        
    print("preds:", preds.shape)   # expect (B, 12, 2)
    if aux is not None:
        print("aux:", aux.shape)
        
    print("Input spec:", model.inputSpec)
    print("Output spec:", model.outputSpec)
        
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total trainable params:", total_params)