import numpy as np
from scipy.optimize import least_squares
import itertools as it
from typing import Optional, Sequence, Tuple

# quantum gates for testing
Toffoli = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 1, 0]])
Fredkin = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1]])

def decomposition(
    X_target: np.ndarray,
    n: int = 6,
    tries: int = 200,
    init_scale: float = 1.0,
    method: str = 'trf',
    **kwargs,
) -> np.ndarray:
    """
    Recover an n x n matrix A such that, for the specified 3-index triples,
    the matrix X computed via 3x3 permanents matches (or approximates) X_target.

    Each equation uses a 3x3 submatrix of A with rows (r1,r2,r3) and columns (c1,c2,c3).
    The 3x3 permanent is the sum over all 3! products without alternating signs.
    """

    # Backward-compat alias: support legacy 'chute_scale'
    if 'chute_scale' in kwargs and isinstance(kwargs['chute_scale'], (int, float)):
        init_scale = float(kwargs['chute_scale'])

    def permanent_3x3(M: np.ndarray) -> float:
        s = 0.0
        for p in it.permutations((0, 1, 2)):
            s += M[0, p[0]] * M[1, p[1]] * M[2, p[2]]
        return float(s)

    triples = [
        (0, 2, 4),
        (0, 2, 5),
        (0, 3, 4),
        (0, 3, 5),
        (1, 2, 4),
        (1, 2, 5),
        (1, 3, 4),
        (1, 3, 5),
    ]

    m = len(triples)
    assert X_target.shape == (m, m), f"X_target must be {m}x{m}, got {X_target.shape}"

    def residuals(vars_flat: np.ndarray) -> np.ndarray:
        A = vars_flat.reshape(n, n)
        diffs = []
        for i in range(m):
            r1, r2, r3 = triples[i]
            for j in range(m):
                c1, c2, c3 = triples[j]
                subM = np.array([
                    [A[r1, c1], A[r1, c2], A[r1, c3]],
                    [A[r2, c1], A[r2, c2], A[r2, c3]],
                    [A[r3, c1], A[r3, c2], A[r3, c3]],
                ])
                val_calc = permanent_3x3(subM)
                diffs.append(val_calc - X_target[i, j])
        return np.array(diffs, dtype=float)
        
    best_cost = float('inf')
    best_solution: Optional[np.ndarray] = None

    # Bounds configuration
    lb_val = -np.inf
    ub_val = np.inf
    lb = np.full(n*n, lb_val, dtype=float)
    ub = np.full(n*n, ub_val, dtype=float)

    for t in range(tries):
        x0 = np.random.uniform(-init_scale, init_scale, n*n)
        result = least_squares(
            residuals,
            x0,
            method=method,
            bounds=(lb, ub) if method in ('trf', 'dogbox') else (-np.inf, np.inf),
        )
        cost = float(result.cost)
        if cost < 1e-12:
            # Return the first exact solution found
            return result.x.reshape(n, n)
        if cost < best_cost:
            best_cost = cost
            best_solution = result.x.reshape(n, n)

    if best_solution is not None:
        raise ValueError(
            f"Exact solution not found. Best cost={best_cost:.3e}. Increase 'tries' or adjust 'init_scale'."
        )
    raise ValueError("Optimization failed to produce a solution.")
