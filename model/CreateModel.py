import math
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
        gnGroups = math.gcd(outCh, min(groups, outCh))
        gnGroups = max(1, gnGroups)
        self.gn = nn.GroupNorm(gnGroups, outCh)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.gn(self.conv(x)))


class FiLM(nn.Module):
    def __init__(self, featDim: int, contextDim: int):
        super().__init__()
        self.scaleProj = nn.Linear(contextDim, featDim)
        self.shiftProj = nn.Linear(contextDim, featDim)

    def forward(self, fmap: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        if context is None:
            return fmap
        scale = self.scaleProj(context).unsqueeze(-1).unsqueeze(-1)
        shift = self.shiftProj(context).unsqueeze(-1).unsqueeze(-1)
        return fmap * (1.0 + scale) + shift


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
        self.egoFusion = FiLM(featDim + 2, 256)

    def forward(self, imgT, imgTm1, egoContext=None, returnMotionMap=False):
        feat8T, feat16T = self.backbone(imgT)
        feat8Tm1, feat16Tm1 = self.backbone(imgTm1)

        motionMap = None
        if returnMotionMap:
            motion8 = torch.mean(torch.abs(feat8T - feat8Tm1), dim=1, keepdim=True)
            motion16 = torch.mean(torch.abs(feat16T - feat16Tm1), dim=1, keepdim=True)
            motion16Up = F.interpolate(motion16, size=motion8.shape[-2:], mode="bilinear", align_corners=False)
            motionMap = 0.5 * motion8 + 0.5 * motion16Up

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
        fmap = self.egoFusion(fmap, egoContext)

        if returnMotionMap:
            return fmap, motionMap.squeeze(1)

        return fmap


# -------------------------
# Decoder: stepwise cross-attention over spatial memory
# -------------------------
class EgoStateEncoder(nn.Module):
    def __init__(self, inputDim: int = 3, hiddenDim: int = 256):
        super().__init__()
        self.hiddenDim = hiddenDim
        self.gru = nn.GRU(input_size=inputDim, hidden_size=hiddenDim, num_layers=1, batch_first=True)

    def forward(self, egoHistory: torch.Tensor) -> torch.Tensor:
        if egoHistory is None:
            return None
        _, hidden = self.gru(egoHistory)
        return hidden[-1]


class ScalarTimeEmbedding(nn.Module):
    def __init__(self, hiddenDim: int):
        super().__init__()
        midDim = max(64, hiddenDim // 2)
        self.net = nn.Sequential(
            nn.Linear(2, midDim),
            nn.SiLU(inplace=True),
            nn.Linear(midDim, hiddenDim),
        )

    def forward(self, timePairs: torch.Tensor) -> torch.Tensor:
        return self.net(timePairs)


class CrossAttentionDeltaKinematicDecoder(nn.Module):
    """
    Stepwise kinematic decoder that repeatedly attends back to the spatial
    feature map instead of planning from a single pooled visual summary.

    At each future step, the decoder forms a query from the recurrent state,
    ego-state context, previous predicted kinematics, and time embedding. That
    query cross-attends over the image memory, then updates a recurrent state
    used to predict the next delta in signed speed and yaw rate.
    """
    def __init__(self, featDim=386, hiddenDim=640, predSteps=12, numHeads=8, numLayers=3):
        super().__init__()
        self.predSteps = predSteps
        self.hiddenDim = hiddenDim
        self.numLayers = max(1, numLayers)
        self.memoryProj = nn.Conv2d(featDim, hiddenDim, 1, bias=False)
        self.memoryNorm = nn.LayerNorm(hiddenDim)
        self.egoProj = nn.Linear(256, hiddenDim)
        self.stateProj = nn.Linear(2, hiddenDim)
        self.timeEmbedding = ScalarTimeEmbedding(hiddenDim)
        self.queryProj = nn.Linear(hiddenDim * 4, hiddenDim)
        self.crossAttention = nn.MultiheadAttention(
            embed_dim=hiddenDim,
            num_heads=max(1, numHeads),
            dropout=0.1,
            batch_first=True,
        )
        self.attnContextProj = nn.Linear(hiddenDim * 2, hiddenDim)
        self.gru = nn.GRU(
            input_size=hiddenDim + hiddenDim + 256 + 2,
            hidden_size=hiddenDim,
            num_layers=self.numLayers,
            batch_first=True,
            dropout=0.1 if self.numLayers > 1 else 0.0,
        )
        self.initStateProj = nn.Linear(hiddenDim + 256, hiddenDim)
        self.deltaHead = nn.Sequential(
            nn.Linear(hiddenDim, hiddenDim // 2),
            nn.SiLU(inplace=True),
            nn.Linear(hiddenDim // 2, 2),
        )
        self.register_buffer(
            "vectorTimes",
            torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.85, 1.1, 1.35, 1.6, 2.3, 3.0], dtype=torch.float32),
            persistent=False,
        )

    def _resolveVectorTimes(self, batchSize, device, dtype, vectorTimes):
        defaultTimes = self.vectorTimes.to(device=device, dtype=dtype)
        if vectorTimes is None:
            # FIXED: dynamic fallback for non-12 predSteps using buildNonUniformTimeOffsets
            if defaultTimes.numel() != self.predSteps:
                dyn = torch.tensor(buildNonUniformTimeOffsets(self.predSteps, float(defaultTimes[-1])), device=device, dtype=dtype)
                return dyn.unsqueeze(0).expand(batchSize, -1)
            return defaultTimes.unsqueeze(0).expand(batchSize, -1)
        if vectorTimes.dim() == 1:
            vectorTimes = vectorTimes.unsqueeze(0).expand(batchSize, -1)
        if vectorTimes.size(1) != self.predSteps:
            # FIXED: dynamic fallback when provided vectorTimes length mismatches predSteps
            if defaultTimes.numel() != self.predSteps:
                dyn = torch.tensor(buildNonUniformTimeOffsets(self.predSteps, float(defaultTimes[-1])), device=device, dtype=dtype)
                return dyn.unsqueeze(0).expand(batchSize, -1)
            return defaultTimes.unsqueeze(0).expand(batchSize, -1)
        return vectorTimes.to(device=device, dtype=dtype)

    def _buildTimeFeatures(self, effectiveTimes: torch.Tensor) -> torch.Tensor:
        deltaTimes = effectiveTimes.clone()
        if effectiveTimes.size(1) > 1:
            deltaTimes[:, 1:] = effectiveTimes[:, 1:] - effectiveTimes[:, :-1]
        deltaTimes = torch.clamp(deltaTimes, min=1e-3)
        return torch.stack([effectiveTimes, deltaTimes], dim=-1)

    def forward(self, featMap, egoContext, vectorTimes=None, teacherForcing=True, tfRatio=0.9, gtTraj=None):
        batchSize, _, height, width = featMap.shape
        memory = self.memoryProj(featMap).flatten(2).permute(0, 2, 1)
        memory = self.memoryNorm(memory)
        pooledMemory = memory.mean(dim=1)

        if egoContext is None:
            egoContext = torch.zeros(batchSize, 256, device=featMap.device, dtype=featMap.dtype)

        effectiveTimes = self._resolveVectorTimes(batchSize, featMap.device, featMap.dtype, vectorTimes)
        timeFeatures = self._buildTimeFeatures(effectiveTimes)

        initState = torch.tanh(self.initStateProj(torch.cat([pooledMemory, egoContext], dim=1)))
        hiddenState = initState.unsqueeze(0).repeat(self.numLayers, 1, 1).contiguous()
        currentState = torch.zeros(batchSize, 2, device=featMap.device, dtype=featMap.dtype)
        predictions = []
        attnMaps = []

        for stepIndex in range(self.predSteps):
            useTeacherForcing = False
            if teacherForcing and gtTraj is not None:
                useTeacherForcing = bool(torch.rand(1, device=featMap.device).item() < tfRatio)

            # FIXED: teacher-forcing leakage (critical)
            # Use one-step-lag teacher forcing for delta-recurrent model to avoid
            # leaking the current-step target into the prediction for the same step.
            if stepIndex == 0:
                previousState = torch.zeros_like(currentState)
            else:
                previousState = gtTraj[:, stepIndex - 1, :] if useTeacherForcing else currentState
            timeEmbed = self.timeEmbedding(timeFeatures[:, stepIndex, :])
            queryState = hiddenState[-1]
            queryToken = self.queryProj(
                torch.cat(
                    [
                        queryState,
                        self.egoProj(egoContext),
                        self.stateProj(previousState),
                        timeEmbed,
                    ],
                    dim=1,
                )
            ).unsqueeze(1)

            attnContext, attnWeights = self.crossAttention(queryToken, memory, memory, need_weights=True)
            attnContext = self.attnContextProj(torch.cat([attnContext.squeeze(1), queryState], dim=1))
            stepInput = torch.cat([attnContext, timeEmbed, egoContext, previousState], dim=1).unsqueeze(1)
            gruOut, hiddenState = self.gru(stepInput, hiddenState)
            deltaState = self.deltaHead(gruOut.squeeze(1))
            currentState = currentState + deltaState
            predictions.append(currentState)
            attnMaps.append(attnWeights.view(batchSize, 1, height, width))

        preds = torch.stack(predictions, dim=1)
        return preds, attnMaps


# -------------------------
# Full model wrapper
# -------------------------
class SmoothKinematicTrajectoryModel(nn.Module):
    """
    Motion-aware CNN + cross-attention recurrent planner for smooth future ego motion.

    Predicts signed longitudinal speed (m/s) and yaw rate (rad/s)
    over non-uniform future time steps, while re-attending to the spatial image
    memory at every rollout step instead of using one pooled visual summary.

    Inputs (forward):
        imgT      : (B, 3, 360, 640)   current frame (normalized)
        imgTm1    : (B, 3, 360, 640)   previous frame (0.1 s earlier)
        egoHistory: (B, 8, 3)           last 8 states [speed, accel, yaw_rate] (optional)
        gtTraj    : (B, 12, 2)          ground-truth for teacher forcing (training only)

    Outputs:
        preds : (B, 12, 2)   [signed_speed_mps, yaw_rate_radps]
        aux   : None
        attn  : list of per-step cross-attention maps over the fused image memory

    Attributes:
        vectorTimes : tensor of the configured future time steps
        name        : "TrajectoryModel_MotionFpn_CrossAttentionKinematic_V2"
    """
    def __init__(
        self,
        featDim=384,
        hiddenDim=640,
        predSteps=12,
        useAuxDyn=False,
        intervalSeconds=0.25,
        totalSeconds=3.0,
        vectorTimes=None,
        baseChannels=48,
        numHeads=8,
        numLayers=3,
        egoDropoutProb=0.15,
    ):
        super().__init__()
        if vectorTimes is not None and len(vectorTimes) == predSteps:
            self.vectorTimes = [float(t) for t in vectorTimes]
        else:
            self.vectorTimes = buildNonUniformTimeOffsets(predSteps, totalSeconds)
        self.encoder = MotionFpnEncoder(featDim=featDim, baseChannels=baseChannels)
        self.egoEncoder = EgoStateEncoder()
        self.decoder = CrossAttentionDeltaKinematicDecoder(
            featDim=featDim + 2,
            hiddenDim=hiddenDim,
            predSteps=predSteps,
            numHeads=numHeads,
            numLayers=numLayers,
        )
        self.name = "TrajectoryModel_MotionFpn_CrossAttentionKinematic_V2"
        self.inputSpec = {
            'image_size': (360, 640),
            'temporal_delay_seconds': 0.1
        }
        self.outputSpec = {
            'num_vectors': predSteps,
            'intervalSeconds': float(self.vectorTimes[0]),
            'totalSeconds': float(self.vectorTimes[-1]),
        }
        self.use_aux_dyn = False
        self.egoDropoutProb = float(egoDropoutProb)

    def forward(self, imgT, imgTm1, gtTraj=None, teacherForcing=True, tfRatio=0.9, egoHistory=None, vectorTimes=None, returnMotionMap=False):
        if egoHistory is None:
            batchSize = imgT.shape[0]
            egoHistory = torch.zeros(batchSize, 8, 3, device=imgT.device, dtype=imgT.dtype)
        egoContext = self.egoEncoder(egoHistory)
        if self.training and egoContext is not None and self.egoDropoutProb > 0.0:
            keepMask = (torch.rand(egoContext.size(0), 1, device=egoContext.device) >= self.egoDropoutProb).to(egoContext.dtype)
            egoContext = egoContext * keepMask
        if returnMotionMap:
            fmap, motionMap = self.encoder(imgT, imgTm1, egoContext=egoContext, returnMotionMap=True)
        else:
            fmap = self.encoder(imgT, imgTm1, egoContext=egoContext)
            motionMap = None
        preds, attnMaps = self.decoder(
            fmap,
            egoContext,
            vectorTimes=vectorTimes,
            teacherForcing=teacherForcing,
            tfRatio=tfRatio,
            gtTraj=gtTraj,
        )
        aux = None
        if returnMotionMap:
            return preds, aux, attnMaps, motionMap
        return preds, aux, attnMaps


class TrajectoryModel(SmoothKinematicTrajectoryModel):
    pass


# -------------------------
# Quick smoke test & param count
# -------------------------
if __name__ == "__main__":
    batchSize = 2
    dummyImg1 = torch.randn(batchSize, 3, 360, 640)
    dummyImg2 = torch.randn(batchSize, 3, 360, 640)
    dummyEgoHistory = torch.randn(batchSize, 8, 3)
    model = SmoothKinematicTrajectoryModel(featDim=384, hiddenDim=640, predSteps=12, baseChannels=48, numHeads=8, numLayers=3)
    model.eval()

    with torch.no_grad():
        preds, aux, attn = model(dummyImg1, dummyImg2, egoHistory=dummyEgoHistory)

    print("Model:", model.name)
    print("preds:", preds.shape)
    if aux is not None:
        print("aux:", aux.shape)

    print("Input spec:", model.inputSpec)
    print("Output spec:", model.outputSpec)

    totalParams = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total trainable params:", totalParams)