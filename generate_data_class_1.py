import numpy as np
import pandas as pd

def gaussnk(n):
    """
    Parameters
    ----------
    n : int
        Dimension or signal whose length defines the dimension.

    Returns
    -------
    g : ndarray, shape (n,)
        Normalized discrete Gaussian.
    """
    t = np.arange(n)[:, None]        # column vector (n, 1)
    g = np.zeros((n, 1))
    s = np.sqrt(n)

    for j in range(-3, 4):
        g += np.exp(-np.pi * (t / s + j * s) ** 2)

    g = g.flatten()
    g = g / np.linalg.norm(g)

    return g


def rot(x, shift):
    """Circular shift (rotation)."""
    return np.roll(x, shift)


def modul(x, modo):
    """Modulation by a complex exponential."""
    N = x.shape[-1]
    n = np.arange(N)
    return np.exp(2j * np.pi * modo * n / N) * x


def rotmod(x, shift, modo):
    """
    Rotation followed by modulation (order matters!).
    # pass the vectors one by way, the function doesn't work with matrices

    Parameters
    ----------
    x : ndarray
        Input vector or 2D array (row-wise operation).
    shift : int
        Circular shift.
    modo : float, optional
        Modulation parameter (default = 1).

    Returns
    -------
    y : ndarray
        Rotated and modulated signal.
    """
    x = np.asarray(x)

    y = rot(x, shift)
    y = modul(y, modo)
    return y

n = 12
d = 2**n

# --- Generate base Gaussian kernel ---
g = gaussnk(d)

# Rotation offsets
main_cluster_coordinates = np.array([[1, 1],
                                     [1, 2],
                                     [2, 2],
                                     [2, 1]])

cluster_coordinates = np.vstack([
    main_cluster_coordinates,
    main_cluster_coordinates + np.array([int(d/2), int(d/8)]),
    main_cluster_coordinates + np.array([int(d - d/8), int(d/2)])
])

# --- Generate dataset ---
N = 2*d  # number of samples

F  = np.zeros((N, d), dtype=complex)

rng = np.random.default_rng(seed=42)  # with a predefined seed my clean data will always be the same

for i in range(N):
    c = rng.standard_normal(12)
    c = c / np.linalg.norm(c)

    f = np.zeros(d, dtype=complex)
    for k in range(12):
        f = f + c[k] * rotmod(g, cluster_coordinates[k, 0], cluster_coordinates[k, 1])  # pass the vectors one by way, the function doesn't work with matrices

    F[i, :] = f


def complex_to_str(z):
    return f"{z.real:.15f}{z.imag:+.15f}i"

df = pd.DataFrame(F)
df = df.map(complex_to_str)

df.to_csv("./data/data_clean_class_1_n_{}.csv".format(n),sep=",",header=False,index=False)