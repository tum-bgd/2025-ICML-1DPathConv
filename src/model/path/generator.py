import numpy
import networkx as nx
import torch
import warnings

from operator import sub
from tqdm import tqdm


def EmptyPatchGraph(nRow, nCol):
    g = nx.Graph()
    g.add_nodes_from((i, j) for i in range(nRow) for j in range(nCol))
    return g


def GilbertD2XY(idx, w, h):
    '''
    Adapted from [jakubcerveny/gilbert](https://github.com/jakubcerveny/gilbert) (BSD 2-Clause License)
    Original copyright (c) 2018, Jakub Červený
    Generalized Hilbert sfc for arbitrary-sized 2D rectangular grids.
    Takes a position along the gilbert curve and returns
    its 2D (x,y) coordinate.
    '''
    if w >= h:
        return GilbertD2XYRec(idx, 0, 0, 0, w, 0, 0, h)
    return GilbertD2XYRec(idx, 0, 0, 0, 0, h, w, 0)


def SGN(x):
    '''
    signature func.
    '''
    return -1 if x < 0 else (1 if x > 0 else 0)


def GilbertD2XYRec(dstIdx, curIdx, x, y, ax, ay, bx, by):
    '''
    Adapted from [jakubcerveny/gilbert](https://github.com/jakubcerveny/gilbert) (BSD 2-Clause License)
    Original copyright (c) 2018, Jakub Červený
    '''

    w = abs(ax + ay)
    h = abs(bx + by)

    (dax, day) = (SGN(ax), SGN(ay))  # unit major direction
    (dbx, dby) = (SGN(bx), SGN(by))  # unit orthogonal direction

    di = dstIdx - curIdx

    if h == 1: return (x + dax*di, y + day*di)
    if w == 1: return (x + dbx*di, y + dby*di)

    (ax2, ay2) = (ax//2, ay//2)
    (bx2, by2) = (bx//2, by//2)

    w2 = abs(ax2 + ay2)
    h2 = abs(bx2 + by2)

    if 2*w > 3*h:
        if (w2 % 2) and (w > 2):
            # prefer even steps
            (ax2, ay2) = (ax2 + dax, ay2 + day)

        # long case: split in two parts only
        nxtIdx = curIdx + abs((ax2 + ay2)*(bx + by))
        if (curIdx <= dstIdx) and (dstIdx < nxtIdx):
            return GilbertD2XYRec(dstIdx, curIdx,  x, y, ax2, ay2, bx, by)
        curIdx = nxtIdx

        return GilbertD2XYRec(dstIdx, curIdx, x+ax2, y+ay2, ax-ax2, ay-ay2, bx, by)

    if (h2 % 2) and (h > 2):
        # prefer even steps
        (bx2, by2) = (bx2 + dbx, by2 + dby)

    # standard case: one step up, one long horizontal, one step down
    nxtIdx = curIdx + abs((bx2 + by2)*(ax2 + ay2))
    if (curIdx <= dstIdx) and (dstIdx < nxtIdx):
        return GilbertD2XYRec(dstIdx, curIdx, x,y, bx2,by2, ax2,ay2)
    curIdx = nxtIdx

    nxtIdx = curIdx + abs((ax + ay)*((bx - bx2) + (by - by2)))
    if (curIdx <= dstIdx) and (dstIdx < nxtIdx):
        return GilbertD2XYRec(dstIdx, curIdx,
                              x+bx2, y+by2,
                              ax, ay,
                              bx-bx2, by-by2)
    curIdx = nxtIdx

    return GilbertD2XYRec(dstIdx, curIdx,
                          x+(ax-dax)+(bx2-dbx),
                          y+(ay-day)+(by2-dby),
                          -bx2, -by2,
                          -(ax-ax2), -(ay-ay2))


def GilbertSFC(h, w):
    patchGraph = EmptyPatchGraph(h, w)
    prevNode = (0, 0)
    edges = []
    for idx in range(w*h):
        currNode = GilbertD2XY(idx, h, w)
        if prevNode == currNode:
            continue
        edges.append((prevNode, currNode))
        prevNode = currNode
    patchGraph.add_edges_from(edges)
    return patchGraph


def RasterScan(h, w):
    patchGraph = EmptyPatchGraph(h, w)
    edges = []
    for i in range(h):
        for j in range(w-1):
            edges.append(((h-i-1, j), (h-i-1, j+1)))
        if i > 0:
            edges.append(((h-i, w-1), (h-i-1, 0)))
    patchGraph.add_edges_from(edges)
    return patchGraph


def ZCurve(h, w):
    '''
    Generate Z-curve coordinates for a given 2D size.
    Returns a list of [h, w] coordinates following the Z-curve pattern.
    '''
    def interleave_bits(x, y):
        """Helper function to interleave bits of x and y to create Z-order value"""
        z = 0
        for i in range(max(x.bit_length(), y.bit_length())):
            z |= ((x & (1 << i)) << i) | ((y & (1 << i)) << (i + 1))
        return z

    def deinterleave_bits(z):
        # Helper func. to deinterleave bits to get (x, y) coordinates
        x = y = 0
        for i in range(z.bit_length()):
            if i % 2 == 0:
                x |= (z & (1 << i)) >> (i // 2)
            else:
                y |= (z & (1 << i)) >> ((i + 1) // 2)
        return x, y

    n = w * h
    coords = []
    for i in range(n):
        x, y = deinterleave_bits(i)
        if x < w and y < h:  # Only include points within bounds
            coords.append([y, w-1-x])  # [h, w]
    return coords


def GetChain(h, w, root=(0, 0), algo=GilbertSFC):
    patchGraph = algo(h, w)
    if algo == ZCurve:
        return patchGraph
    for n in patchGraph.nodes():
        if patchGraph.degree(n) == 1:
            root = n
            break
    chain = list(nx.dfs_preorder_nodes(patchGraph, source=root))
    res = []
    for n in chain:
        res.append([int(n[0]), int(n[1])])
    return res


def RotateChain(chain, rotate=0):
    '''
    rotate chain clockwise
    '''
    assert rotate in [0, 90, 180, 270]
    if rotate == 0:
        return chain
    xs, ys = Chain2XY(chain)
    w, h = max(xs), max(ys)
    if not w == h:
        # not squared, 180 only
        assert rotate == 180
    import copy
    if rotate == 90:
        # |-
        for p in chain:
            tmp = copy.deepcopy(p)
            p[0] = h - tmp[1]
            p[1] = tmp[0]
    elif rotate == 180:
        # T
        for p in chain:
            tmp = copy.deepcopy(p)
            p[0] = h - tmp[0]
            p[1] = w - tmp[1]
    elif rotate == 270:
        # -|
        for p in chain:
            tmp = copy.deepcopy(p)
            p[0] = tmp[1]
            p[1] = w - tmp[0]
    return chain


def SampleChain(rawChain, h, w, bl=(0, 0), rotate=0):
    if len(rawChain) == h*w:
        # not shifted
        return RotateChain(rawChain, rotate=rotate)
    else:
        # shifted
        chain = []
        for p in rawChain:
            if p[0] >= bl[0]         and p[1] >= bl[1] and \
               p[0] <  bl[0]+h and p[1] <  bl[1]+w:
                chain.append(list(map(sub, p, bl)))
    return RotateChain(chain, rotate=rotate)


def CheckCompleteness(h, w, chain):
    """
    Verify if coordinates traverse the entire 2D space exactly once.
    Args:
        h, w: size of 2D space
        chain: List of [h, w] coordinates
    Returns:
        bool: True if traversal is valid, False otherwise
    """
    # 1. Length matches total space?
    if not len(chain) == h * w:
        return False
    # 2. All coordinates are within bounds?
    visited = set()
    for thisH, thisW in chain:
        # Check bounds
        if not (0 <= thisH < h and 0 <= thisW < w):
            return False
        # Check duplicates
        coord = (thisH, thisW)
        if coord in visited:
            return False
        visited.add(coord)
    # 3. All points are visited?
    if len(visited) != h * w:
        return False
    return True


def GetPaths(h, w, config=None):
    '''
    Get multiple path to traverse images of size [h, w].
    config: a list of [sfc, nPad, bl, rotate]
        sfc: 'r', 'g', 'z'
    output: [nPath, h*w, 2], h-w (row-col, y-x) order
    '''
    paths = []
    for sfc, nPad, bl, rotate in tqdm(config):
        # print(sfc, nPad, bl, rotate)
        if nPad == 0:
            # no pad, no shift
            assert(bl == [0, 0])
        if sfc == 'r':
            chain = GetChain(h, w, root=(0, 0), algo=RasterScan)
        elif sfc == 'g':
            bigChain = GetChain(h+nPad, w+nPad, root=(0, 0), algo=GilbertSFC)
            chain = SampleChain(bigChain, h, w, bl=bl, rotate=rotate)
        elif sfc == 'z':
            newSize = max(h+nPad, w+nPad)
            assert bin(newSize).count('1') == 1
            bigChain = GetChain(newSize, newSize, algo=ZCurve)
            chain = SampleChain(bigChain, h, w, bl=bl, rotate=rotate)
        else:
            raise Exception(f'sfc {sfc} not defined')
        assert CheckCompleteness(h, w, chain)
        paths.append(torch.tensor(chain))
    return torch.stack([path.to(torch.int32) for path in paths])


def Chain2XY(chain):
    '''
    chain is row(h/y),col(w/x)
    |__ Cartesian coordinate system
    '''
    y, x = zip(*chain)
    return list(x), list(y)


def FindNeighbors(n, h, w):
    res = [[n[0]-1, n[1]-1], [n[0]-1, n[1]], [n[0]-1, n[1]+1],\
           [n[0]  , n[1]-1]                , [n[0]  , n[1]+1],\
           [n[0]+1, n[1]-1], [n[0]+1, n[1]], [n[0]+1, n[1]+1]]
    return [p for p in res if 0<=p[0] and p[0]<=h-1 and 0<=p[1] and p[1]<=w-1]


def GetDistance(chain, n0, n1):
    return abs(chain.index(n0) - chain.index(n1))


def GetNeighborDistance(chain, h, w):
    res = numpy.zeros((h, w))
    for i in range(h):
        for j in range(w):
            n = [i, j]
            dist = [GetDistance(chain, n, p) for p in FindNeighbors(n, h, w)]
            res[i, j] = sum(dist)
    return res


def LocalityMeasure(s):
    rPath = GetChain(s, s, root=(0, 0), algo=RasterScan)
    hPath = GetChain(s, s, root=(0, 0), algo=GilbertSFC)
    zPath = GetChain(s, s, root=(0, 0), algo=ZCurve)
    rPathNBDist = GetNeighborDistance(rPath, s, s)
    hPathNBDist = GetNeighborDistance(hPath, s, s)
    zPathNBDist = GetNeighborDistance(zPath, s, s)
    print(f"{s} x {s}:")
    print(f"  raster scan: total = {rPathNBDist.sum()}")
    print(f"      Hilbert: total = {hPathNBDist.sum()}, better ratio = {(hPathNBDist < rPathNBDist).sum()/(s*s)*100:.2f}%")
    print(f"      Z-order: total = {zPathNBDist.sum()}, better ratio = {(zPathNBDist < rPathNBDist).sum()/(s*s)*100:.2f}%")


if __name__ == "__main__":
    # paths = GetPaths(64, 64, config=[['g', 5, [0, 0],   0],
    #                                  ['z', 0, [0, 0],  90],
    #                                  ['g', 0, [0, 0], 180]])
    # print(paths.size())
    # LocalityMeasure(32)
    # LocalityMeasure(64)
    # LocalityMeasure(128)
    # LocalityMeasure(256)
    pass
