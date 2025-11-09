import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Utility blocks
# -------------------------
class ConvGNAct(nn.Module):
    def __init__(self, inCh, outCh, kernel=3, stride=1, padding=1, groups=32):
        super().__init__()
        self.conv = nn.Conv2d(inCh, outCh, kernel, stride, padding, bias=False)
        gnGroups = min(groups, outCh)
        self.gn = nn.GroupNorm(gnGroups, outCh)
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
# Encoder: deeper with more residuals for richer features
# -------------------------
class EarlyFusionEncoder(nn.Module):
    def __init__(self, featDim=256):
        super().__init__()
        self.stem = nn.Sequential(
            ConvGNAct(6, 32, kernel=7, stride=2, padding=3),  # H/2
            Residual(32),
            Residual(32),  # Added for depth
            ConvGNAct(32, 64, stride=2)                       # H/4
        )
        self.layer1 = nn.Sequential(
            Residual(64),
            Residual(64),  # Added
            ConvGNAct(64, 128, stride=2)                      # H/8
        )
        self.layer2 = nn.Sequential(
            Residual(128),
            Residual(128),  # Added
            ConvGNAct(128, featDim, stride=1)                 # keep H/8
        )
        self.skipProj = nn.Conv2d(64, featDim, kernel_size=1, bias=False)
        self.fusion = nn.Conv2d(featDim * 2, featDim, kernel_size=1, bias=False)

        # sinusoidal frequencies
        freqs = torch.pow(10000.0, -torch.arange(0, featDim // 4, dtype=torch.float32) / (featDim // 4))
        self.register_buffer('freqs', freqs)

    def _sinusoidal_pos_emb(self, H: int, W: int, device: torch.device):
        y = torch.arange(H, device=device, dtype=torch.float32) / H
        x = torch.arange(W, device=device, dtype=torch.float32) / W

        y_emb = y[:, None, None] * self.freqs
        x_emb = x[None, :, None] * self.freqs

        sin_y = torch.sin(y_emb).expand(-1, W, -1)
        cos_y = torch.cos(y_emb).expand(-1, W, -1)
        sin_x = torch.sin(x_emb).expand(H, -1, -1)
        cos_x = torch.cos(x_emb).expand(H, -1, -1)

        emb = torch.cat([sin_x, cos_x, sin_y, cos_y], dim=-1)   # (H,W,C)
        return emb.permute(2, 0, 1)                             # (C,H,W)

    def forward(self, imgT, imgTm1):
        x = torch.cat([imgT, imgTm1], dim=1)

        xStem = self.stem(x)
        xL1 = self.layer1(xStem)
        xL2 = self.layer2(xL1)

        # skip from stem
        skip = F.interpolate(self.skipProj(xStem), size=xL2.shape[2:], mode='bilinear', align_corners=False)
        xL2 = xL2 + skip

        # multi-scale
        l1Proj = F.interpolate(self.skipProj(xL1[:, :64, ...]), size=xL2.shape[2:], mode='bilinear', align_corners=False)
        fused = self.fusion(torch.cat([xL2, l1Proj], dim=1))

        # positional encoding
        B, C, H, W = fused.shape
        posEmb = self._sinusoidal_pos_emb(H, W, fused.device)
        posEmb = posEmb.unsqueeze(0).expand(B, -1, -1, -1)

        return fused + posEmb


# -------------------------
# Multi-head spatial attention
# -------------------------
class SpatialAttentionDecoder(nn.Module):
    def __init__(self, featDim, hiddenDim, numHeads=8):
        super().__init__()
        self.multihead = nn.MultiheadAttention(
            embed_dim=featDim, num_heads=numHeads, batch_first=True, dropout=0.1
        )
        self.queryProj = nn.Linear(hiddenDim, featDim)

    def forward(self, featMap, query):
        B, C, H, W = featMap.shape
        kv = featMap.flatten(2).permute(0, 2, 1)          # (B,S,C)
        q = self.queryProj(query).unsqueeze(1)            # (B,1,C)

        attnOut, attnW = self.multihead(q, kv, kv)
        context = attnOut.squeeze(1)

        attnMap = attnW.view(B, -1, H, W).mean(dim=1, keepdim=True)
        return context, attnMap


# -------------------------
# Transformer decoder: deeper, more heads, dropout for stability
# -------------------------
class TransformerDecoder(nn.Module):
    def __init__(self, featDim=256, hiddenDim=256, predSteps=12,
                 numLayers=2, numHeads=8):
        super().__init__()
        self.predSteps = predSteps
        self.hiddenDim = hiddenDim
        self.featDim = featDim

        decLayer = nn.TransformerDecoderLayer(
            d_model=hiddenDim, nhead=numHeads,
            dim_feedforward=hiddenDim * 2, batch_first=True, activation='gelu', dropout=0.1
        )
        self.transformer = nn.TransformerDecoder(decLayer, num_layers=numLayers)

        self.attn = SpatialAttentionDecoder(featDim, hiddenDim, numHeads)

        # project flattened feature map to hiddenDim
        self.memoryProj = nn.Linear(featDim, hiddenDim)

        self.posEmb = nn.Parameter(torch.randn(1, predSteps + 1, hiddenDim))
        self.initEmb = nn.Linear(2, hiddenDim)
        self.outHead = nn.Sequential(
            nn.Linear(hiddenDim, hiddenDim // 2),
            nn.ReLU(),
            nn.Linear(hiddenDim // 2, 2)
        )

    def forward(self, featMap, teacherForcing=False, gtTraj=None, tfRatio=0.9):
        B = featMap.size(0)
        device = featMap.device

        # flatten + project to hiddenDim
        memory = featMap.flatten(2).permute(0, 2, 1)  # (B,S,C)
        memory = self.memoryProj(memory)              # (B,S,hiddenDim)

        xy = torch.zeros(B, 2, device=device)
        seqEmb = self.initEmb(xy).unsqueeze(1)        # (B,1,hidden)

        preds, attnMaps = [], []

        for t in range(self.predSteps):
            query = seqEmb[:, -1, :]
            context, attnMap = self.attn(featMap, query)

            tgt = seqEmb + self.posEmb[:, :seqEmb.size(1), :]
            curLen = tgt.size(1)
            causalMask = torch.triu(
                torch.full((curLen, curLen), float('-inf'), device=device),
                diagonal=1
            )

            out = self.transformer(tgt, memory, tgt_mask=causalMask)
            nextHidden = out[:, -1, :]
            delta = self.outHead(nextHidden)
            nextXy = xy + delta

            preds.append(nextXy.unsqueeze(1))
            attnMaps.append(attnMap)

            newEmb = nextHidden.unsqueeze(1)
            seqEmb = torch.cat([seqEmb, newEmb], dim=1)

            if teacherForcing and gtTraj is not None and torch.rand(1).item() < tfRatio:
                xy = gtTraj[:, t, :].detach()
                seqEmb[:, -1, :] = self.initEmb(xy)
            else:
                xy = nextXy.detach()

        preds = torch.cat(preds, dim=1)
        return preds, attnMaps


# -------------------------
# Full model
# -------------------------
class TrajectoryModel(nn.Module):
    def __init__(self, featDim=256, hiddenDim=256, predSteps=12, useAuxDyn=False, intervalSeconds=0.25):
        super().__init__()
        self.encoder = EarlyFusionEncoder(featDim=featDim)
        self.decoder = TransformerDecoder(
            featDim=featDim, hiddenDim=hiddenDim, predSteps=predSteps
        )
        self.useAuxDyn = useAuxDyn
        if useAuxDyn:
            self.aux = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(featDim, 128),
                nn.ReLU(),
                nn.Linear(128, 4)
            )
        self.name = "TrajectoryFusionTransformer_TFTV2"

        # Input and output specs
        self.inputSpec = {
            'image_size': (360, 640),  # (height, width)
            'temporal_delay_seconds': 0.1  # Time delay between previous and current images
        }
        self.outputSpec = {
            'num_vectors': predSteps,  # Number of predicted trajectory vectors
            'intervalSeconds': intervalSeconds  # Time interval between each predicted vector
        }

    def forward(self, imgT, imgTm1, gtTraj=None, teacherForcing=False, tfRatio=0.9):
        fmap = self.encoder(imgT, imgTm1)
        preds, attnMaps = self.decoder(
            fmap, teacherForcing=teacherForcing, gtTraj=gtTraj, tfRatio=tfRatio
        )
        aux = self.aux(fmap) if self.useAuxDyn else None
        return preds, aux, attnMaps


# -------------------------
# Smoke test
# -------------------------
if __name__ == "__main__":
    B = 2
    img1 = torch.randn(B, 3, 360, 640)
    img2 = torch.randn(B, 3, 360, 640)

    model = TrajectoryModel(featDim=512, hiddenDim=768, predSteps=12, useAuxDyn=False)
    model.eval()

    with torch.no_grad():
        preds, aux, attn = model(img1, img2)

    print("preds:", preds.shape)
    if aux is not None:
        print("aux:", aux.shape)
        
    print("Input spec:", model.inputSpec)
    print("Output spec:", model.outputSpec)

    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Trainable params:", total)