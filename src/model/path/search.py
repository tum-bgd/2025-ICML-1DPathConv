import multiprocessing
multiprocessing.set_start_method('spawn', force=True)
import pickle

from functools import partial
from multiprocessing import Pool
from pulp import *

from generator import *
from visualization import *


def NextPo2(n):
    orig = n
    if n <= 0:
        return 1
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    if orig == n + 1:
        return orig * 2
    return n + 1


def GetCandidateHilbertConfig(h, w):
    # all possible paths [sfc, nPad, bl, rotate]
    config = []
    pad = h
    if h == 224:
        pad = 16
    rotation = [0, 90, 180, 270]
    # normal hilbert
    config.extend([['g', 0, [0, 0], r] for r in rotation])
    # padded hilbert
    config.extend([['g', p, [i, j], r]
        for p in range(1, pad)
            for i in range(0, p+1)
                for j in range(0, p+1)
                    for r in rotation
                        if abs(i-j) > 1])
    return config


def GetCandidateZCurveConfig(h, w):
    # padded z-curve
    config = []
    rotation = [0, 90, 180, 270]
    # normal z-curve
    config.append(['z', 0, [0, 0], 0])
    # padded z-curve
    pad = NextPo2(h) - h
    config.extend([['z', pad, [i, j], r]
        for i in range(0, pad+1)
            for j in range(0, pad+1)
                for r in rotation
                    if abs(i-j) > 1])
    return config


def ProcessOneZCurveConfig(config, h, w, bigChain, rScanNBDist, verbose=False):
    '''
    Process a single configuration
    '''
    sfc, nPad, bl, rotate = config
    if verbose:
        # DEBUG: which CPU core this process is running on
        import os
        print(f"Processing on CPU {os.getpid()} - Config: {config}")
    chain = SampleChain(bigChain, h, w, bl=bl, rotate=rotate)
    assert CheckCompleteness(h, w, chain)
    chainNBDist = GetNeighborDistance(chain, h, w)
    isShorter = chainNBDist < rScanNBDist
    totalCost = chainNBDist.sum()
    return (sfc, nPad, tuple(bl), rotate), isShorter, totalCost


def GetZCurveCost(h, w, configs, overwrite=False):
    '''
    configs: a list of [sfc, nPad, bl, rotate]
        sfc: 'r', 'g', 'z'
    '''
    if os.path.exists(ZCurveCostDir) and not overwrite:
        print(ZCurveCostDir, "exists, overwrite is set to False.")
        return
    shorter, cost = {}, {}
    nPad = h
    if h == 224:
        nPad = 32
    newSize = max(h+nPad, w+nPad)
    assert bin(newSize).count('1') == 1

    bigChain = GetChain(newSize, newSize, algo=ZCurve)
    rScanChain = GetChain(h, w, root=(0, 0), algo=RasterScan)
    rScanNBDist = GetNeighborDistance(rScanChain, h, w)
    # Create partial function with all shared data
    processConfig = partial(
        ProcessOneZCurveConfig,
        h=h,
        w=w,
        bigChain=bigChain,
        rScanNBDist=rScanNBDist
    )
    with Pool(processes=multiprocessing.cpu_count()) as pool:
        results = list(tqdm(pool.imap(processConfig, configs), total=len(configs)))
    # Store results
    for config, isShorter, totalCost in results:
        shorter[config] = isShorter
        cost[config] = totalCost
    with open(ZCurveCostDir, 'wb') as f:
        pickle.dump([shorter, cost], f)


def ProcessOneHilbertConfig(config, h, w, rScanNBDist, verbose=False):
    '''
    Process a single configuration
    '''
    sfc, nPad, bl, rotate = config
    if verbose:
        # DEBUG: which CPU core this process is running on
        import os
        print(f"Processing on CPU {os.getpid()} - Config: {config}")
    bigChain = GetChain(h+nPad, w+nPad, algo=GilbertSFC)
    chain = SampleChain(bigChain, h, w, bl=bl, rotate=rotate)
    assert CheckCompleteness(h, w, chain)
    chainNBDist = GetNeighborDistance(chain, h, w)
    isShorter = chainNBDist < rScanNBDist
    totalCost = chainNBDist.sum()
    return (sfc, nPad, tuple(bl), rotate), isShorter, totalCost


def GetHilbertCost(h, w, configs, overwrite=False):
    '''
    configs: a list of [sfc, nPad, bl, rotate]
        sfc: 'r', 'g', 'z'
    '''
    if os.path.exists(HilbertCostDir) and not overwrite:
        print(HilbertCostDir, "exists, overwrite is set to False.")
        return
    shorter, cost = {}, {}

    rScanChain = GetChain(h, w, root=(0, 0), algo=RasterScan)
    rScanNBDist = GetNeighborDistance(rScanChain, h, w)
    # Create partial function with all shared data
    processConfig = partial(
        ProcessOneHilbertConfig,
        h=h,
        w=w,
        rScanNBDist=rScanNBDist
    )
    with Pool(processes=multiprocessing.cpu_count()) as pool:
        results = list(tqdm(pool.imap(processConfig, configs), total=len(configs)))
    # Store results
    for config, isShorter, totalCost in results:
        shorter[config] = isShorter
        cost[config] = totalCost
    with open(HilbertCostDir, 'wb') as f:
        pickle.dump([shorter, cost], f)


def LoadCost(dir):
    with open(dir, 'rb') as f:
        return pickle.load(f)


def PixelCoverSolver(shorter, cost, solver='greedy'):
    # ilp is too slow. Debug using small inputs
    assert solver in ['greedy', 'ilp', 'rr']
    # 2D arrays => sets of positions
    posSet = {
        key: set((i,j)
                for i in range(arr.shape[0])
                    for j in range(arr.shape[1])
                        if arr[i,j])
        for key, arr in shorter.items()}

    allPos = set.union(*posSet.values())  # all pos to be covered

    def GreedySolver():
        uncovered = allPos.copy()
        selected = []
        while uncovered:
            # Find configuration with best cost-effectiveness
            bestRatio = float('inf')
            bestConfig = None
            for config in posSet:
                if config in selected:
                    continue
                newCovered = len(uncovered & posSet[config])
                if newCovered == 0:
                    continue
                ratio = cost[config] / newCovered
                if ratio < bestRatio:
                    bestRatio = ratio
                    bestConfig = config
            if bestConfig is None:
                break
            selected.append(bestConfig)
            uncovered -= posSet[bestConfig]
        return selected

    def ILPSolver():
        # ILP. Surely, much much slower
        prob = LpProblem("PixelCoverage", LpMinimize)
        # decision variables
        config_vars = LpVariable.dicts("config",
                                       shorter.keys(),
                                       lowBound=0,
                                       upBound=1,
                                       cat='Binary')
        # objective function
        prob += lpSum([cost[config] * config_vars[config]
                      for config in shorter.keys()])
        # coverage constraints
        for pos in allPos:
            prob += lpSum([config_vars[config]
                         for config in shorter.keys()
                         if pos in posSet[config]]) >= 1
        # solve
        prob.solve(PULP_CBC_CMD(msg=False))
        return [config for config in shorter.keys()
                if config_vars[config].value() > 0.5]

    def RRSolver(nRound=30, scalingFactor=1.2):
        import numpy as np
        from scipy.optimize import linprog
        configs = list(shorter.keys())
        n_configs = len(configs)
        # construct LP matrix
        A = []
        for pos in allPos:
            row = [1 if pos in posSet[config] else 0 for config in configs]
            A.append(row)
        c = [cost[config] for config in configs]
        b = [1] * len(allPos)
        bounds = [(0, 1) for _ in range(n_configs)]
        result = linprog(c, A_ub=-np.array(A), b_ub=-np.array(b),
                        bounds=bounds, method='highs')
        if not result.success:
            return None
        best_solution = None
        best_cost = float('inf')
        # scale up probabilities to ensure better coverage
        probs = np.minimum(result.x * scalingFactor, 1.0)
        for _ in range(nRound):
            selected = []
            # 1st round: Use scaled probabilities
            for i, prob in enumerate(probs):
                if np.random.random() < prob:
                    selected.append(configs[i])
            # check coverage, add missing coverage greedily
            covered = set().union(*[posSet[config] for config in selected])
            if covered != allPos:
                uncovered = allPos - covered
                # Add configurations greedily until coverage is complete
                while uncovered:
                    bestRatio = float('inf')
                    bestConfig = None
                    for config in configs:
                        if config in selected:
                            continue
                        newCovered = len(uncovered & posSet[config])
                        if newCovered == 0:
                            continue
                        ratio = cost[config] / newCovered
                        if ratio < bestRatio:
                            bestRatio = ratio
                            bestConfig = config
                    if bestConfig is None:
                        break
                    selected.append(bestConfig)
                    uncovered -= posSet[bestConfig]
            # Update best solution
            total_cost = sum(cost[config] for config in selected)
            if total_cost < best_cost:
                best_cost = total_cost
                best_solution = selected.copy()
        return best_solution

    if solver == 'greedy':
        return GreedySolver()
    elif solver == 'ilp':
        print("ILP is extremely slow when #inputs is large.")
        return ILPSolver()
    elif solver == 'rr':
        return RRSolver()


def GetMinimalPathSet(dir, solver='greedy'):
    assert solver in ['greedy', 'rr']
    shorter, cost = LoadCost(dir)
    path = PixelCoverSolver(shorter, cost, solver=solver)
    print(path, sum([cost[p] for p in path]))


h, w = 32, 32
# h, w = 64, 64
# h, w = 224, 224
ZCurveCostDir = f'./src/model/path/zcurve_cost_{h}_{w}.pkl'
HilbertCostDir = f'./src/model/path/hilbert_cost_{h}_{w}.pkl'
if __name__ == "__main__":
    # multiprocessing.cpu_count(): 128
    GetZCurveCost(h, w, GetCandidateZCurveConfig(h, w))
    GetMinimalPathSet(ZCurveCostDir, solver='greedy')
    # GetMinimalPathSet(ZCurveCostDir, solver='rr')  # potential better results, but slow
    GetHilbertCost(h, w, GetCandidateHilbertConfig(h, w))
    GetMinimalPathSet(HilbertCostDir, solver='greedy')
    # GetMinimalPathSet(HilbertCostDir, solver='rr') # potential better results, but slow
    '''
    [32 x 32]
    Z-curve (~10s 3969):
        greedy (<1s):
            [('z', 0, (0, 0), 0),
             ('z', 32, (10, 18), 0),
             ('z', 32, (12, 20), 0)], cost = 582984.0
        rr (~2s):
            [('z', 32, (16, 26), 180),
             ('z', 32, (14, 16), 0),
             ('z', 32, (12, 20), 0)], cost = 576988.0
    Hilbert (~2min 39684):
        greedy (~8s):
            [('g', 0, (0, 0), 0),
             ('g', 24, (14, 1), 180),
             ('g', 18, (7, 0), 270)], cost = 686166.0
        rr (~1min):
            [('g', 5, (5, 0), 270),
             ('g', 0, (0, 0), 90),
             ('g', 19, (12, 0), 90)], cost = 675730.0

    [64 x 64]
    Z-curve (~10min 16129):
        greedy (~1min):
            [('z', 64, (32, 16), 0),
             ('z', 64, (28, 42), 90),
             ('z', 64, (24, 40), 0)], cost = 4856128.0
        rr (~5min):
            [('z', 64, (36, 32), 270),
             ('z', 64, (36, 34), 90),
             ('z', 64, (28, 32), 0)], cost = 4771756.0
    Hilbert (~4h 333316):
        greedy (~10min):
            [('g', 0, (0, 0), 0),
             ('g', 7, (7, 0), 180),
             ('g', 40, (13, 1), 0)], cost = 5626494.0
        rr (~2h):
            [('g', 0, (0, 0), 0),
             ('g', 7, (7, 0), 180),
             ('g', 40, (13, 1), 0)], cost = 5626494.0

    [224 x 224]
    Z-curve:
        greedy:
            [('z', 0, (0, 0), 0),
             ('z', 32, (11, 14), 180),
             ('z', 32, (16, 14), 0)], cost = 204649852.0
    Hilbert:
        greedy:
            [('g', 0, (0, 0), 0),
             ('g', 9, (9, 0), 270),
             ('g', 11, (11, 0), 90),
             ('g', 0, (0, 0), 180)], cost = 324363926.0
        rr:
            [('g', 13, (10, 5), 270),
             ('g', 10, (5, 0), 270),
             ('g', 7, (7, 0), 90)], cost = 245831552.0
    '''
