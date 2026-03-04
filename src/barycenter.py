import numpy as np
import ot
from .utils import *

def ot_barycenter(x1, 
                  x2, 
                  C1, 
                  C2, 
                  *, 
                  nItermax=10, 
                  alpha=0.5, 
                  return_loss=True, 
                  return_diff=False,
                  thr=1e-4,
                  verbose=True):
    """
    Implementation of Marco Cuturi and Arnaud Doucet. “Fast computation of
                      Wasserstein barycenters”. In: International conference
                      on machine learning.  PMLR. 2014
    FOR 2 DISTRIBUTIONS ONLY.

    Returns the barycenter between weights x1 and x2 using Cuturi-Doucet's algorithm.

    Args:
        x1 (np.ndarray, (M1*N1))       : weights
        x2 (np.ndarray, (M2*N2))       : weights
        C1 (np.ndarray, (M1*N1, M1*N2)): cost matrix between supports of x1 and barycenter
        C2 (np.ndarray, (M2*N2, M1*N2)): cost matrix between supports of x2 and barycenter
        nItermax (int)                 : max number of iterations
        alpha (double, [0, 1])         : interpolation parameter
        return_loss                    : return loss (1 - alpha)OT(x1, x) + alpha OT(x2, x)
        return_diff                    : return iterate difference: ||x_k+1 - x_k||**2 / ||x_0||**2
        thr (double)                   : convergence criteria (for loss)
        verbose (bool)                 : prints info

    Returns:
        x (np.ndarray, (M1*N2))        : barycenter.
        loss (np.ndarray, opt)         : loss.      
        diff (np.ndarray, opt)         : diff between iterates.      
    """

    x0 = np.ones(C1.shape[1])   # initialize barycenter as uniform weights
    x0 /= x0.sum()   
    x0_norm = (x0**2).sum()

    loss = []
    if return_diff:
        diff = []

    x = x0.copy()
    ahat   = np.ones(x0.size) / x0.size
    atilde = np.ones(x0.size) / x0.size
    t = 1
    t0 = 1

    for k in range(nItermax):
        old_x = x.copy()
        b = (t + 1) / 2
        x = (1 - 1 / b) * ahat + 1 / b * atilde

        loss_1, log_1 = ot.lp.emd2(x1, x, C1, log=True) # log true to retrieve the dual solutions
        loss_2, log_2 = ot.lp.emd2(x2, x, C2, log=True) # log true to retrieve the dual solutions

        a_1 = log_1["v"]
        a_1 -= a_1.mean()
        a_2 = log_2["v"]
        a_2 -= a_2.mean()

        a = (1 - alpha) * a_1 + alpha * a_2

        atilde *= np.exp(-t0 * b * a)
        atilde /= atilde.sum()
        ahat = (1 - 1 / b) * ahat + 1 / b * atilde
        t = t + 1

        loss.append((1 - alpha) * loss_1 + alpha * loss_2)

        if return_diff and k >= 1:
            diff.append(((x - old_x)**2).sum() / x0_norm)

        if k > 1 and np.abs(loss[k] - loss[k - 1]) / loss[0] < thr:
            break

    if verbose:
        print(f"Convergence attained after {k} iterations.")

    if not return_loss and not return_diff:
        return x
    
    res = [x]

    if return_loss:
        res = *res, loss
    if return_diff:
        res = *res, diff

    return res


def uot_barycenter(x1, 
                   x2, 
                   c1, 
                   c2, 
                   rows1,
                   cols1,
                   rows2,
                   cols2,
                   eta, 
                   bary_size,
                   *, 
                   nItermax=10, 
                   alpha=.5,
                   return_loss=False,
                   return_diff=False,
                   thr=1e-4,
                   verbose=True,
                   eps=1e-16):
    """
    Computes UOT barycenter between two distributions. For more details, check IV. Algorithm.
    Uses only finite entries of cost matrix. As such, the matrices appear as vectors along with their row and column indices.
    For more details see functions cost_matrix_horizontal_overlap or cost_matrix_vertical_overlap.

    Args:
        x1 (np.ndarray, (M1*N1))       : weights
        x2 (np.ndarray, (M2*N2))       : weights
        c1 (np.ndarray)                : cost matrix as vector (see cost_matrix_horizontal_overlap) between supports of x1 and barycenter
        c2 (np.ndarray)                : cost matrix as vector (see cost_matrix_vertical_overlap) between supports of x2 and barycenter
        rows1 (np.ndarray)             : row indices of c1.
        cols1 (np.ndarray)             : column indices of c1.
        rows2 (np.ndarray)             : row indices of c2.
        cols2 (np.ndarray)             : column indices of c2.
        eta (double)                   : UOT barycenter (we take eta_1=eta_2=eta_3=eta_4).
        bary_size (size of the output) : Size of the output vector (cf. K in Algorithm).
        nItermax (int)                 : max number of iterations
        alpha (double, [0, 1])         : interpolation parameter
        return_loss                    : return loss (1 - alpha)OT(x1, x) + alpha OT(x2, x)
        return_diff                    : return iterate difference: ||x_k+1 - x_k||**2 / ||x_0||**2
        thr (double)                   : convergence criteria (for loss)
        verbose (bool)                 : prints info
        eps (double)                   : to avoid division by zero.

    Returns:
        x (np.ndarray, (bary_size))    : barycenter.
        loss (np.ndarray)         : loss.      
        diff (np.ndarray)         : diff between iterates.      
    """

    x0 = np.ones(bary_size)
    x0 /= x0.sum()
    x0_norm = (x0**2).sum()

    plan_10 = np.ones_like(c1)
    # plan_10 /= plan_10.sum()
    plan_20 = np.ones_like(c2)
    # plan_20 /= plan_20.sum()

    D1 = np.exp(-c1/eta/2)
    D2 = np.exp(-c2/eta/2)

    plan_1k1 = plan_10.copy()
    plan_2k1 = plan_20.copy()
    x        = x0.copy()

    loss = []

    if return_diff:
        diff = []

    for k in range(nItermax):
        old_x = x.copy()

        sum_row_1 = np.zeros_like(x1)
        sum_col_1 = np.zeros_like(x)

        np.add.at(sum_row_1, rows1, plan_1k1)
        np.add.at(sum_col_1, cols1, plan_1k1)

        sum_row_2 = np.zeros_like(x2)
        sum_col_2 = np.zeros_like(x)
        np.add.at(sum_row_2, rows2, plan_2k1)
        np.add.at(sum_col_2, cols2, plan_2k1)

        u1 = np.sqrt(x1 / (sum_row_1 + eps))
        v1 = np.sqrt(x  / (sum_col_1 + eps))

        u2 = np.sqrt(x2 / (sum_row_2 + eps))
        v2 = np.sqrt(x  / (sum_col_2 + eps))

        plan_1k1 = u1[rows1] * plan_1k1 * D1 * v1[cols1]
        plan_2k1 = u2[rows2] * plan_2k1 * D2 * v2[cols2]

        l1 = uot_loss(c1, plan_1k1, eta, x1, x, rows=rows1, cols=cols1)
        l2 = uot_loss(c2, plan_2k1, eta, x2, x, rows=rows2, cols=cols2)
        loss.append((1 - alpha) * l1 + alpha * l2)

        sum_col_1 = np.zeros_like(x)
        np.add.at(sum_col_1, cols1, plan_1k1)
        sum_col_2 = np.zeros_like(x)
        np.add.at(sum_col_2, cols2, plan_2k1)

        x = (1 - alpha) * sum_col_1 + alpha * sum_col_2

        if k > 0 and np.abs(loss[k] - loss[k-1]) / loss[0]  < thr:
            break
        if return_diff and k >= 1:
            diff.append(((x - old_x)**2).sum() / x0_norm)
    
    if verbose:
        print(f"Convergence attained after {k} iterations.")
    
    if not return_loss and not return_diff:
        return x
    
    res = [x]

    if return_loss:
        res = *res, loss
    if return_diff:
        res = *res, diff

    return res

def uot_loss(c, 
             plan, 
             eta, 
             a, 
             b, 
             *, 
             rows=None, 
             cols=None):
    """
        UOT Barycenter objective function (see eq. 34).
        It uses the vectorized forms of the cost matrices and plans. 
        To compute the sums of the rows and columns of the plans, i.e. T1_I and T^\top_J
        we require the indices of the plans' rows and cols
        i.e. plan[k] = T[i, j] with T the transport plan, i = rows[k], j = cols[k].

        Args:
            c (np.ndarray)    : vectorized cost matrix.
            plan (np.ndarray) : vectorized transport plan.
            eta (double, > 0) : UOT parameter.
            a (np.ndarray)    : weights a.
            b (np.ndarray)    : weights b.
            rows (np.ndarray) : plan rows' indices.
            cols (np.ndarray) : plan cols' indices.

        Returns:
            Objective evaluated at the input parameters.

    """
    # compute T1_I
    sum_row_plan = np.zeros_like(a)
    np.add.at(sum_row_plan, rows, plan)

    # compute T^\top 1_J
    sum_col_plan = np.zeros_like(b)
    np.add.at(sum_col_plan, cols, plan)

    P = (c * plan).sum()
    F1 = eta * kullback_leibler(sum_row_plan, a)
    F2 = eta * kullback_leibler(sum_col_plan, b)

    return P + F1 + F2