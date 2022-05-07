import torch.nn as nn

LearnedBatchNorm = nn.BatchNorm2d
NonAffineGroupNorm = lambda planes: nn.GroupNorm(32, planes)


class NonAffineBatchNorm(nn.BatchNorm2d):
    def __init__(self, dim):
        super(NonAffineBatchNorm, self).__init__(dim, affine=False)
