import numpy as np

def uot_sparse_barycenter(x1, 
                          x2, 
                          c1, 
                          c2, 
                          eta, 
                          rows1,
                          cols1,
                          rows2,
                          cols2,
                          bary_size,
                          *, 
                          nItermax=10, 
                          verbose=True,
                          zf=1e-16):
    
    x0 = np.ones(bary_size)
    x0 /= x0.sum()

    x0_norm = (x0**2).sum()

    plan_10 = np.ones_like(c1)
    plan_10 /= plan_10.sum()
    plan_20 = np.ones_like(c2)
    plan_20 /= plan_20.sum()

    K1 = np.exp(-c1/eta/2)
    K2 = np.exp(-c2/eta/2)

    plan_1k1 = plan_10.copy()
    plan_2k1 = plan_20.copy()
    x        = x0.copy()

    for k in range(nItermax):
        if verbose:
            print(f"At iteration {k}")
        old_x = x.copy()

        sum_row_1 = np.zeros_like(x1)
        sum_col_1 = np.zeros_like(x)

        np.add.at(sum_row_1, rows1, plan_1k1)
        np.add.at(sum_col_1, cols1, plan_1k1)

        sum_row_2 = np.zeros_like(x2)
        sum_col_2 = np.zeros_like(x)
        np.add.at(sum_row_2, rows2, plan_2k1)
        np.add.at(sum_col_2, cols2, plan_2k1)

        u1 = np.sqrt(x1 / (sum_row_1 + zf))
        v1 = np.sqrt(x  / (sum_col_1 + zf))

        u2 = np.sqrt(x2 / (sum_row_2 + zf))
        v2 = np.sqrt(x  / (sum_col_2 + zf))

        plan_1k1 = u1[rows1] * plan_1k1 * K1 * v1[cols1]
        plan_2k1 = u2[rows2] * plan_2k1 * K2 * v2[cols2]

        sum_col_1 = np.zeros_like(x)
        np.add.at(sum_col_1, cols1, plan_1k1)
        sum_col_2 = np.zeros_like(x)
        np.add.at(sum_col_2, cols2, plan_2k1)

        x = 0.5 * sum_col_1 + 0.5 * sum_col_2
        
    if verbose:
        print(f"Convergence attained after {k} iterations.")
    
    return x