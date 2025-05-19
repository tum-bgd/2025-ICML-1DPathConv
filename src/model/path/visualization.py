import numpy
import os

from matplotlib import pyplot as plt

from generator import *


def PlotXYChain(xs, ys, nbDistChain=None, nbDistRScan=None, noCurve=False, pSize=12):
    plt.gca().set_aspect('equal')
    plt.tight_layout()
    plt.axis('off')
    defaultColor = '#54494B'
    posColor = '#fdb863'
    negColor = '#5e3c99'
    for i in range(1, len(xs)):
        # edges
        if (abs(xs[i-1]-xs[i]) <= 1 and abs(ys[i-1]-ys[i]) <= 1) or noCurve:
            plt.plot([xs[i-1], xs[i]], [ys[i-1], ys[i]], '-', color=defaultColor)
        else:
            def DrawCurve(p1, p2):
                if p1[0] == p2[0]:
                    # x the same
                    dist = abs(p1[1] - p2[1])
                    x = numpy.linspace(p1[1], p2[1], 100)
                    y = 0.5 * (1/dist) * (x-p1[1])*(x-p2[1])
                    plt.plot((int(p1[0]==0)*2-1)*y+p1[0], x, '-', color=defaultColor)
                elif p1[1] == p2[1]:
                    # y the same
                    dist = abs(p1[0] - p2[0])
                    x = numpy.linspace(p1[0], p2[0], 100)
                    y = 0.5 * (1/dist) * (x-p1[0])*(x-p2[0])
                    plt.plot(x, (int(p1[1]==0)*2-1)*y+p1[1], '-', color=defaultColor)
                else:
                    a = (p2[1] - p1[1])/(numpy.cosh(p2[0]) - numpy.cosh(p1[0]))
                    b = p1[1] - a*numpy.cosh(p1[0])
                    x = numpy.linspace(p1[0], p2[0], 100)
                    y = a*numpy.cosh(x) + b
                    plt.plot(x, y, '-', color=defaultColor)
            DrawCurve((xs[i-1], ys[i-1]), (xs[i], ys[i]))
    for x, y in zip(xs, ys):
        # vertices, xy, wh
        if nbDistChain is None:
            plt.plot(x, y, 'o', color=defaultColor, markersize=pSize)
        elif nbDistChain[y, x] <= nbDistRScan[y, x]:
            plt.plot(x, y, 'o', color=posColor, markersize=pSize)
        else:
            plt.plot(x, y, 'o', color=negColor, markersize=pSize)


def DrawCurve(h, w, sfc='g', rotate=0, pSize=10, drawCost=False):
    figSize = 5 * max(h, w) / 8
    plt.figure(figsize=(figSize, figSize))
    if sfc == 'g':
        origChain = GetChain(h, w, root=(0, 0), algo=GilbertSFC)
        name = 'hilbert'
        noCurve = False
    elif sfc == 'r':
        origChain = GetChain(h, w, root=(0, 0), algo=RasterScan)
        name = 'raster'
        noCurve = True
    elif sfc == 'z':
        origChain = GetChain(h, w, algo=ZCurve)
        name = 'zcurve'
        noCurve = True
    chain = SampleChain(origChain, h, w, bl=(0, 0), rotate=rotate)
    xs, ys = Chain2XY(chain)
    if drawCost:
        rScanChain = GetChain(h, w, root=(0, 0), algo=RasterScan)
        rScanNBDist = GetNeighborDistance(rScanChain, h, w)
        chainNBDist = GetNeighborDistance(chain, h, w)
        PlotXYChain(xs, ys, nbDistChain=chainNBDist, nbDistRScan=rScanNBDist, noCurve=noCurve, pSize=pSize)
        plt.savefig(os.path.join(TAR_DIR, f'{name}_{h}_{w}_{rotate}_dist.svg'), transparent=True, format='svg')
    else:
        PlotXYChain(xs, ys, noCurve=noCurve, pSize=pSize)
        plt.savefig(os.path.join(TAR_DIR, f'{name}_{h}_{w}_{rotate}.svg'), transparent=True, format='svg')
    plt.close()


def DrawShiftedCurve(h, w, nPad, bl, sfc='g', rotate=0, pSize=10, drawCost=False):
    name = 'hilbert'
    algo = GilbertSFC
    newH = h+nPad
    newW = w+nPad
    noCurve = False
    if sfc == 'z':
        name = 'zcurve'
        algo = ZCurve
        newH, newW = max(newH, newW), max(newH, newW)
        assert bin(newH).count('1') == 1
        noCurve = True
    bigChain = GetChain(newH, newW, algo=algo)
    chain = SampleChain(bigChain, h, w, bl=bl, rotate=rotate)

    figSize = 5 * (w // 8)
    plt.figure(figsize=(figSize, figSize))
    xs, ys = Chain2XY(chain)
    if drawCost:
        rScanChain = GetChain(h, w, root=(0, 0), algo=RasterScan)
        rScanNBDist = GetNeighborDistance(rScanChain, h, w)
        chainNBDist = GetNeighborDistance(chain, h, w)
        PlotXYChain(xs, ys, nbDistChain=chainNBDist, nbDistRScan=rScanNBDist, noCurve=noCurve, pSize=pSize)
        plt.savefig(
            os.path.join(TAR_DIR, f'{name}_{h}_{w}_shifted_{nPad}_{bl}_{rotate}_dist.svg'),
            transparent=True, format='svg')
    else:
        PlotXYChain(xs, ys, noCurve=noCurve, pSize=pSize)
        plt.savefig(
            os.path.join(TAR_DIR, f'{name}_{h}_{w}_shifted_{nPad}_{bl}_{rotate}.svg'),
            transparent=True, format='svg')
    plt.close()


TAR_DIR = './pic'
if __name__ == "__main__":
    if not os.path.exists(TAR_DIR):
        os.makedirs(TAR_DIR)

    # Example usage:
    # DrawCurve( 8,  8, sfc='g', pSize=12, drawCost=True)
    # DrawCurve( 8,  8, sfc='z', pSize=12, drawCost=True)
    # DrawCurve(16, 16, sfc='g', pSize=13.5, drawCost=True)
    # DrawCurve(22, 22, sfc='g', pSize=12)
    # DrawShiftedCurve(16, 16, 6, (1, 5), sfc='g', pSize=12, drawCost=True)
    # DrawCurve(16, 16, sfc='g', pSize=12)
    # DrawShiftedCurve(16, 16, 6, (0, 5), sfc='g', pSize=10)
    # DrawCurve( 16,  16, sfc='g', pSize=12, drawCost=True)
    # DrawCurve( 32,  32, sfc='g', pSize=12, drawCost=True)
    # DrawCurve( 64,  64, sfc='g', pSize=12, drawCost=True)
    # DrawCurve(128, 128, sfc='g', pSize=12, drawCost=True)
    # DrawCurve( 16,  16, sfc='z', pSize=12, drawCost=True)
    # DrawCurve( 32,  32, sfc='z', pSize=12, drawCost=True)
    # DrawCurve( 64,  64, sfc='z', pSize=12, drawCost=True)
    # DrawCurve(128, 128, sfc='z', pSize=12, drawCost=True)
    # DrawCurve(18, 21, sfc='g', pSize=12)
    # DrawShiftedCurve(18, 21, 11, (0, 5), sfc='z', pSize=8)