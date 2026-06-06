import pandas as pd
import numpy as np
from scipy.stats import chi2

def g_square_test(data, x, y, z=[], alpha=0.05):
    """
    G-square test for conditional independence.
    Null Hypothesis: x and y are independent given z.
    Returns: (p_value, is_independent)
    """
    all_vars = list(set([x, y] + z))
    counts = data.groupby(all_vars).size().reset_index(name='N_xyz')

    if not z:
        total_n = len(data)
        nx = data[x].value_counts()
        ny = data[y].value_counts()

        g2 = 0
        for _, row in counts.iterrows():
            obs = row['N_xyz']
            exp = (nx[row[x]] * ny[row[y]]) / total_n
            g2 += obs * np.log(obs / exp) if exp > 0 else 0
        g2 *= 2
        
        df = (data[x].nunique() - 1) * (data[y].nunique() - 1)
    else:
        nz = data.groupby(z).size().reset_index(name='N_z')
        nxz = data.groupby([x] + z).size().reset_index(name='N_xz')
        nyz = data.groupby([y] + z).size().reset_index(name='N_yz')
        merged = counts.merge(nxz, on=[x]+z).merge(nyz, on=[y]+z).merge(nz, on=z)

        g2 = 0
        for _, row in merged.iterrows():
            obs = row['N_xyz']
            exp = (row['N_xz'] * row['N_yz']) / row['N_z']
            if obs > 0 and exp > 0:
                g2 += obs * np.log(obs / exp)
        g2 *= 2

        df = (data[x].nunique() - 1) * (data[y].nunique() - 1)
        for col in z:
            df *= data[col].nunique()
    
    if df <= 0:
        return 1.0, True

    p_value = chi2.sf(g2, df)
    return p_value, p_value > alpha
