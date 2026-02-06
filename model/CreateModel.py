import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# Small utility blocks
# -------------------------
class ConvGNAct(nn.Module):
    def __init__(self, inCh, outCh, kernel=3, stride=1, padding=1, groups=32):
        super().__init__()
        self.conv = nn.Conv2d(inCh, outCh, kernel, stride, padding, bias=False)
        gnGroups = min(groups, outCh)
        self.gn = nn.GroupNorm(gnGroups, outCh)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.gn(self.conv(x)))


class SEBlock(nn.Module):
    def __init__(self, ch, reduction=8):
        super().__init__()
        reducedCh = max(1, ch // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch, reducedCh),
            nn.SiLU(inplace=True),
            nn.Linear(reducedCh, ch),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class Residual(nn.Module):
    def __init__(self, ch, useSe=False):
        super().__init__()
        self.block = nn.Sequential(
            ConvGNAct(ch, ch, kernel=3, padding=1),
            ConvGNAct(ch, ch, kernel=3, padding=1)
        )
        self.se = SEBlock(ch) if useSe else nn.Identity()

    def forward(self, x):
        out = x + self.block(x)
        return self.se(out)


def buildNonUniformTimeOffsets(vectorCount:int, totalTime:float) -> list:
    """Builds a list of non-uniform time offsets for the predicted vectors.
    
    If vectorCount is 12 and totalTime is 3.0, uses a predefined non-uniform pattern.
    Otherwise, generates uniform offsets based on totalTime and vectorCount.
    Args:
        vectorCount (int): The number of vectors to predict.
        totalTime (float): The total time span for the predictions.
    Returns:
        list: A list of time offsets for each predicted vector.
    """
    
    if vectorCount == 12 and abs(totalTime - 3.0) < 1e-6:
        timeDeltas = [0.1] * 6 + [0.25] * 4 + [0.7] * 2
    else:
        step = totalTime / max(1, vectorCount)
        timeDeltas = [step] * vectorCount

    timeOffsets = []
    running = 0.0
    for delta in timeDeltas:
        running += float(delta)
        timeOffsets.append(running)

    if timeOffsets:
        scale = totalTime / max(1e-6, timeOffsets[-1])
        if abs(scale - 1.0) > 1e-6:
            timeOffsets = [t * scale for t in timeOffsets]

    return [round(t, 4) for t in timeOffsets]


# -------------------------
# Encoder: shared-weight motion encoder + FPN fusion
# -------------------------
class MotionBackbone(nn.Module):
    """
    A lightweight CNN backbone that extracts multi-scale features from input images.
    Consists of a stem block followed by three stages, each downsampling the spatial resolution and increasing the channel count. Each stage includes a residual block with optional SE attention. The backbone outputs two feature maps at 1/8 and 1/16 resolution for subsequent fusion in the MotionFpnEncoder.
    Args:
        baseChannels (int): The number of channels in the first stage, which is then scaled up in subsequent stages.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing the 1/8 and 1/16 resolution feature maps extracted from the input image.
    """
    def __init__(self, baseChannels=32):
        super().__init__()
        self.stem = nn.Sequential(
            ConvGNAct(3, baseChannels, kernel=5, stride=2, padding=2),
            Residual(baseChannels, useSe=True)
        )
        self.stage1 = nn.Sequential(
            ConvGNAct(baseChannels, baseChannels * 2, stride=2),
            Residual(baseChannels * 2, useSe=True)
        )
        self.stage2 = nn.Sequential(
            ConvGNAct(baseChannels * 2, baseChannels * 3, stride=2),
            Residual(baseChannels * 3, useSe=True)
        )
        self.stage3 = nn.Sequential(
            ConvGNAct(baseChannels * 3, baseChannels * 4, stride=2),
            Residual(baseChannels * 4, useSe=True)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        feat8 = self.stage2(x)
        feat16 = self.stage3(feat8)
        return feat8, feat16


class MotionFpnEncoder(nn.Module):
    """
    Feature pyramid encoder that fuses motion cues between consecutive frames.
    Builds a MotionBackbone to extract multi-scale features from the current and
    previous frames, concatenates per-scale features with their temporal
    differences, fuses them through 1x1 convolutions, and combines scales using a
    top-down FPN-style fusion. Adds normalized x/y coordinate channels to the
    final feature map to provide explicit spatial context.
    Args:
        featDim (int): Output feature dimensionality after fusion.
        baseChannels (int): Base channel count for the backbone.
    Returns:
        torch.Tensor: Fused feature map with appended coordinate channels of
            shape (B, featDim + 2, H, W).
    """
    def __init__(self, featDim=192, baseChannels=32):
        super().__init__()
        self.backbone = MotionBackbone(baseChannels=baseChannels)
        self.fuse8 = nn.Conv2d(baseChannels * 3 * 3, featDim, kernel_size=1, bias=False)
        self.fuse16 = nn.Conv2d(baseChannels * 4 * 3, featDim, kernel_size=1, bias=False)
        self.fpnFuse = ConvGNAct(featDim, featDim, kernel=3, padding=1)

    def forward(self, imgT, imgTm1):
        feat8T, feat16T = self.backbone(imgT)
        feat8Tm1, feat16Tm1 = self.backbone(imgTm1)

        motion8 = torch.cat([feat8T, feat8Tm1, feat8T - feat8Tm1], dim=1)
        motion16 = torch.cat([feat16T, feat16Tm1, feat16T - feat16Tm1], dim=1)

        fused8 = self.fuse8(motion8)
        fused16 = self.fuse16(motion16)
        fused16Up = F.interpolate(fused16, size=fused8.shape[-2:], mode="bilinear", align_corners=False)

        fmap = self.fpnFuse(fused8 + fused16Up)

        batchSize, _, height, width = fmap.shape
        yCoords = torch.linspace(-1.0, 1.0, height, device=fmap.device).view(1, 1, height, 1).expand(batchSize, 1, height, width)
        xCoords = torch.linspace(-1.0, 1.0, width, device=fmap.device).view(1, 1, 1, width).expand(batchSize, 1, height, width)
        fmap = torch.cat([fmap, xCoords, yCoords], dim=1)

        return fmap


# -------------------------
# Decoder: lightweight transformer over spatial memory
# -------------------------
class VectorTransformerDecoder(nn.Module):
    def __init__(self, featDim=194, hiddenDim=256, predSteps=12, numHeads=4, numLayers=2, dropout=0.1, timeSteps=None):
        super().__init__()
        self.predSteps = predSteps
        self.memoryProj = nn.Conv2d(featDim, hiddenDim, kernel_size=1, bias=False)
        self.memoryNorm = nn.LayerNorm(hiddenDim)

        decoderLayer = nn.TransformerDecoderLayer(
            d_model=hiddenDim,
            nhead=numHeads,
            dim_feedforward=hiddenDim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(decoderLayer, num_layers=numLayers)

        self.queryEmbed = nn.Parameter(torch.randn(predSteps, hiddenDim))
        self.timeMlp = nn.Sequential(
            nn.Linear(1, hiddenDim),
            nn.SiLU(inplace=True),
            nn.Linear(hiddenDim, hiddenDim)
        )

        if timeSteps is None:
            timeSteps = [float(i + 1) for i in range(predSteps)]
        self.register_buffer("timeSteps", torch.tensor(timeSteps, dtype=torch.float32), persistent=False)

        self.outHead = nn.Sequential(
            nn.Linear(hiddenDim, hiddenDim // 2),
            nn.SiLU(inplace=True),
            nn.Linear(hiddenDim // 2, 2)
        )

    def forward(self, featMap, timeSteps=None):
        batchSize = featMap.size(0)
        memory = self.memoryProj(featMap)
        memory = memory.flatten(2).permute(0, 2, 1)
        memory = self.memoryNorm(memory)

        query = self.queryEmbed.unsqueeze(0).expand(batchSize, -1, -1)
        if timeSteps is None:
            timeSteps = self.timeSteps
        else:
            timeSteps = torch.tensor(timeSteps, dtype=torch.float32, device=featMap.device)
        timeEmbed = self.timeMlp(timeSteps.view(1, -1, 1))
        query = query + timeEmbed

        decoded = self.decoder(query, memory)
        preds = self.outHead(decoded)
        attnMaps = [None for _ in range(self.predSteps)]
        return preds, attnMaps


# -------------------------
# Full model wrapper
# -------------------------
class TrajectoryModel(nn.Module):
    def __init__(
        self,
        featDim=192,
        hiddenDim=256,
        predSteps=12,
        useAuxDyn=False,
        intervalSeconds=0.25,
        totalSeconds=3.0,
        timeSteps=None,
        baseChannels=32,
    ):
        super().__init__()
        if timeSteps is None:
            totalSeconds = float(totalSeconds) if totalSeconds is not None else float(intervalSeconds * predSteps)
            timeSteps = buildNonUniformTimeOffsets(predSteps, totalSeconds)

        self.timeSteps = timeSteps
        self.encoder = MotionFpnEncoder(featDim=featDim, baseChannels=baseChannels)
        self.decoder = VectorTransformerDecoder(
            featDim=featDim + 2,
            hiddenDim=hiddenDim,
            predSteps=predSteps,
            numHeads=4,
            numLayers=2,
            dropout=0.1,
            timeSteps=timeSteps,
        )

        self.use_aux_dyn = useAuxDyn
        if useAuxDyn:
            self.aux = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(featDim + 2, 128),
                nn.SiLU(inplace=True),
                nn.Linear(128, 2)
            )

        self.name = "TrajectoryModel_MotionFpn_Transformer_V1"

        # Input and output specs
        self.inputSpec = {
            'image_size': (360, 640),
            'temporal_delay_seconds': 0.1
        }
        self.outputSpec = {
            'num_vectors': predSteps,
            'intervalSeconds': float(timeSteps[0]) if timeSteps else intervalSeconds,
            'totalSeconds': float(timeSteps[-1]) if timeSteps else float(predSteps * intervalSeconds),
            'timeSteps': [float(t) for t in timeSteps] if timeSteps else None,
        }

    def forward(self, imgT, imgTm1, gtTraj=None, teacherForcing=False, tfRatio=0.9, timeSteps=None):
        fmap = self.encoder(imgT, imgTm1)
        preds, attnMaps = self.decoder(fmap, timeSteps=timeSteps)
        aux = None
        if self.use_aux_dyn:
            aux = self.aux(fmap)
        return preds, aux, attnMaps


# -------------------------
# Quick smoke test & param count
# -------------------------
if __name__ == "__main__":
    batchSize = 2
    dummyImg1 = torch.randn(batchSize, 3, 360, 640)
    dummyImg2 = torch.randn(batchSize, 3, 360, 640)
    model = TrajectoryModel(featDim=256, hiddenDim=512, predSteps=12, useAuxDyn=True)
    model.eval()

    with torch.no_grad():
        preds, aux, attn = model(dummyImg1, dummyImg2)

    print("Model:", model.name)
    print("preds:", preds.shape)
    if aux is not None:
        print("aux:", aux.shape)

    print("Input spec:", model.inputSpec)
    print("Output spec:", model.outputSpec)

    totalParams = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total trainable params:", totalParams)