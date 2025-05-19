import cupy
import math
import torch

from timm.layers import DropPath

from path.generator import *


cuda_kernel_forward = cupy.RawKernel(r'''
extern "C" __global__
void MultiPathForward(
    const float* input,
    const int* paths,  // Shape: [nPath, H*W, 2]
    float* output,
    const int bs,
    const int iDim,
    const int h,
    const int w,
    const int nPath
) {
    // global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Total number of elements to process
    int total_elements = bs * nPath * h * w;

    if (idx < total_elements) {
        // Calculate position in output tensor
        int wOut = idx % w;
        int hOut = (idx / w) % h;
        int pIdx = (idx / (w * h)) % nPath;
        int b = idx / (w * h * nPath);

        // Get the traversal indices for this position
        int mapIdx = pIdx * (h * w) + hOut * w + wOut;
        int hIn = paths[mapIdx * 2];
        int wIn = paths[mapIdx * 2 + 1];

        // For each input channel
        for(int c = 0; c < iDim; c++) {
            // Calculate input index
            int iIdx = b * (iDim * h * w) +
                       c * (h * w) +
                       hIn * w +
                       wIn;

            // Calculate output index
            int oIdx = b * (nPath * iDim * h * w) +
                       (pIdx * iDim + c) * (h * w) +
                       hOut * w +
                       wOut;

            output[oIdx] = input[iIdx];
        }
    }
}
''', 'MultiPathForward')


class PathTraversalCPU(torch.nn.Module):
    '''
    CPU ver.
    '''
    def __init__(self, paths):
        """
        paths: Tensor of paths [nPath, H*W, 2] containing (h,w) indices
        """
        super().__init__()
        self.paths = paths

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        resImgs = []
        for path in self.paths:
            hs, ws = path[:, 0], path[:, 1]
            resImgs.append(img[:, :, hs, ws])
        return torch.cat(resImgs, dim=1)


class PathTraversalCUDAOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, img, paths):
        assert img.is_cuda and paths.is_cuda
        bs, iDim, h, w = img.shape
        nPath = paths.shape[0]

        output = torch.empty(
            bs, nPath * iDim, h * w,
            device=img.device,
            dtype=img.dtype)  # o: [B, nPath*C, H, W]

        threads_per_block = 256
        total_elements = bs * nPath * h * w
        blocks_per_grid = (total_elements + threads_per_block - 1) // threads_per_block

        cuda_kernel_forward(
            grid=(blocks_per_grid,),
            block=(threads_per_block,),
            args=(img.data_ptr(),
                  paths.data_ptr(),
                  output.data_ptr(),
                  bs, iDim, h, w, nPath))
        # for backward
        ctx.save_for_backward(img, paths, torch.tensor([bs, iDim, h, w, nPath]))
        return output


class PathTraversalCUDA(torch.nn.Module):
    def __init__(self, paths):
        """
        paths: Tensor of paths [nPath, H*W, 2] containing (h,w) indices
        """
        super().__init__()
        self.register_buffer('paths', paths)

    def forward(self, x):
        return PathTraversalCUDAOp.apply(x, self.paths)


def Benchmark(bs, s):
    img = torch.rand((bs, 3, s, s))
    # paths = GetPaths(s, s, config=[['g', 0,  [0, 0],   0],
    #                                ['g', 5,  [0, 4],  90],
    #                                ['g', 32, [4, 5], 180],
    #                                ['g', 5,  [4, 4], 270],
    #                                ['g', 5,  [2, 2],   0]])
    paths = GetPaths(s, s, config=[['g', 0,  [0, 0],   0],
                                   ['g', 5,  [4, 4], 270],
                                   ['g', 5,  [2, 2],   0]])
    gimg = img.cuda()
    gpaths = paths.cuda()
    sample1 = PathTraversalCUDA(paths=gpaths).cuda()
    sample2 = PathTraversalCPU(paths=paths)

    def OneCudaPass():
        out = sample1(gimg)
        torch.cuda.synchronize()
        return out

    def OneNormalPass():
        return sample2(img)

    out1 = OneCudaPass()
    out2 = OneNormalPass()
    print(out1.size(), out2.size(), torch.allclose(out1.cpu(), out2, atol=1e-5))
    import time
    start = time.perf_counter()
    for i in range(1000):
        OneNormalPass()
    end = time.perf_counter()
    t1 = end-start
    print(f"{(t1):.3f}ms.")
    torch.cuda.synchronize()
    start = time.perf_counter()
    for i in range(1000):
        OneCudaPass()
    end = time.perf_counter()
    t2 = end-start
    print(f"{(t2):.3f}ms.")
    print(f"{(t1/t2):.3f}x acceleration.")


class LayerNorm(torch.nn.Module):
    '''
    channel first...
    '''
    def __init__(self,
        normShape: int,
        eps: float=1e-6
    ) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(normShape))
        self.bias = torch.nn.Parameter(torch.zeros(normShape))
        self.eps = eps

    def forward(self, x) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None] * x + self.bias[:, None]
        return x


class EnhancedPathAwareECA(torch.nn.Module):
    def __init__(self, nDim, nPaths, gamma=2, beta=1):
        super().__init__()
        self.nPaths = nPaths
        self.dimPerPath = nDim // nPaths
        # path-specific
        t = int(abs(math.log2(self.dimPerPath) / gamma + beta / gamma))
        k1 = t if t % 2 else t + 1
        k2 = k1 * 2 - 1
        # dual-scale convolutions for each path
        self.path_convs = torch.nn.ModuleList([
            torch.nn.ModuleList([
                torch.nn.Conv1d(1, 1, k1, padding=(k1-1)//2, bias=False),
                torch.nn.Conv1d(1, 1, k2, padding=(k2-1)//2, bias=False)
            ]) for _ in range(nPaths)
        ])
        # cross-path interaction
        self.cross_path_mlp = torch.nn.Sequential(
            torch.nn.LayerNorm(nPaths),
            torch.nn.Linear(nPaths, nPaths * 2),
            torch.nn.GELU(),
            torch.nn.Linear(nPaths * 2, nPaths),
            torch.nn.Sigmoid()
        )
        self.combine = torch.nn.Conv1d(2, 1, 1, bias=True)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        b, c, l = x.size()
        # group channels by paths
        path_features = x.view(b, self.nPaths, self.dimPerPath, l)
        path_attns = []
        for i, convs in enumerate(self.path_convs):
            path_feat = path_features[:, i]
            y = path_feat.mean(-1).unsqueeze(1)
            # dual-scale processing
            y1 = convs[0](y)
            y2 = convs[1](y)
            y_combined = self.combine(torch.cat([y1, y2], dim=1))
            path_attns.append(self.sigmoid(y_combined))
        # combine path-specific attentions
        path_attns = torch.stack(path_attns, dim=1)
        cross_path_feats = path_attns.mean(-1).squeeze(2)
        cross_path_weights = self.cross_path_mlp(cross_path_feats)
        cross_path_weights = cross_path_weights.unsqueeze(-1).unsqueeze(-1)
        # both attentions
        path_attns = path_attns * cross_path_weights
        path_attns = path_attns.unsqueeze(3).expand(-1, -1, -1, l, -1)
        path_attns = path_attns.transpose(-1, -2)
        path_features = path_features.unsqueeze(2) * path_attns
        return path_features.squeeze(2).reshape(b, c, l)


class PathConvBlock(torch.nn.Module):
    '''
    say sth.
    '''
    def __init__(self,
        dim: int,
        nPath: int,
        exp: int=4,
        probShortcut: float=0.1,
        initScalingFactor: float=1e-6
    ) -> None:
        super().__init__()
        self.attn = EnhancedPathAwareECA(dim, nPath)
        self.dwconv = torch.nn.Sequential(
            torch.nn.Conv1d(dim, dim, 9, padding=4, groups=dim),
            LayerNorm(dim))
        self.invBot = torch.nn.Sequential(
            torch.nn.Conv1d(dim, exp*dim, 1),
            LayerNorm(exp*dim),
            torch.nn.GELU(),
            torch.nn.Conv1d(exp*dim, dim, 1),
        )
        self.scaling = torch.nn.Parameter(initScalingFactor * torch.ones((dim)), requires_grad=True)
        self.shortcut = DropPath(probShortcut) if probShortcut > 0. else torch.nn.Identity()

    def forward(self, x) -> torch.Tensor:
        inp = x
        x = x + self.attn(x)
        out = self.dwconv(x)
        out = self.invBot(out).permute(0, 2, 1)
        out = out * self.scaling
        out = out.permute(0, 2, 1)
        out = self.shortcut(out) + inp
        return out


class PathConvStem(torch.nn.Module):
    def __init__(self, iDim, oDim, ks, nPath):
        super().__init__()
        self.nPath = nPath
        self.conv1 = torch.nn.Conv1d(iDim, oDim, 1)
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv1d(oDim, oDim, ks, stride=1, padding=(ks-1)//2, groups=oDim),
            LayerNorm(oDim))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class PathConvLR(torch.nn.Module):
    '''
    1D Path Convolution low res.
    32^2 / 64^2
    '''
    def __init__(self,
        path,
        iDim: int=3,
        nClass: int=1000,
        depth: list=[3, 3, 3, 3],
        nDim: list=[60, 120, 240, 480],
        stochasticDepRate: float=0.1,
        imgLen: int=1024
    ) -> None:
        super().__init__()

        nPath = path.size(0)
        iDim *= nPath
        probShortcut = [x.item() for x in torch.linspace(0, stochasticDepRate, sum(depth))]

        self.pathSampling = PathTraversalCUDA(path)
        self.posEmbedding = torch.nn.Parameter(torch.randn(1, nDim[0], imgLen))
        self.stem = PathConvStem(iDim, nDim[0], 11, nPath)
        self.downsample = torch.nn.ModuleList()
        for i in range(3):
            self.downsample.append(torch.nn.Sequential(
                torch.nn.Conv1d(nDim[i], nDim[i+1], 9, stride=2, padding=4, groups=nDim[i]),
                LayerNorm(nDim[i+1])))

        curr = 0
        self.stage = torch.nn.ModuleList()
        for i in range(4):
            self.stage.append(torch.nn.Sequential(
                *[PathConvBlock(
                    nDim[i], nPath,
                    probShortcut=probShortcut[curr+j]) for j in range(depth[i])]))
            curr += depth[i]

        self.norm = torch.nn.LayerNorm(nDim[-1], eps=1e-6)  # final LN
        self.head = torch.nn.Linear(nDim[-1], nClass)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, torch.nn.Conv1d):
            torch.nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_normal_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, x) -> torch.Tensor:
        x = self.pathSampling(x)
        torch.cuda.synchronize()
        x = self.stem(x)
        x = x + self.posEmbedding
        for i in range(4):
            if i > 0:
                x = self.downsample[i-1](x)
            x = self.stage[i](x)
        x = self.norm(x.mean([-1]))
        x = self.head(x)
        return x


def PathConvS(**kwargs):
    '''
    similar size to ResNet18
    '''
    arch = PathConvLR
    return arch(
        depth=[2, 2, 2, 2],
        nDim=[48, 96, 192, 384],
        **kwargs)


def PathConvB(**kwargs):
    '''
    similar size to ResNet50
    '''
    arch = PathConvLR
    return arch(
        depth=[2, 2, 3, 3],
        nDim=[60, 120, 240, 480],
        **kwargs)


if __name__ == "__main__":
    # Benchmark( 128,  32)
    # Benchmark( 128,  64)
    # Benchmark( 128, 224)
    # Benchmark( 512,  32)
    # Benchmark( 512,  64)
    # Benchmark( 512, 224)
    # Benchmark(1024,  32)
    # Benchmark(1024,  64)
    # Benchmark(1024, 224)
    pass
