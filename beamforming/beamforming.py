import scipy.fft
import numpy as np
import numba as nb
from spectrum import dpss


def construct_slowness_grid(theta, vmin, vmax, Nslow):
    v_grid = np.linspace(vmin, vmax, Nslow)
    slow_grid = 1 / v_grid
    theta, slowness = np.meshgrid(theta, slow_grid)
    Sx, Sy = -slowness * np.sin(theta), -slowness * np.cos(theta)
    Sx, Sy = Sx.T, Sy.T
    return Sx, Sy, v_grid


def construct_times_beamforming(x, y, Sx, Sy):
    x0 = x.mean()
    y0 = y.mean()
    dx = x - x0
    dy = y - y0
    dt = Sx.T * dx + Sy.T * dy
    return dt


def precompute_A(dt, freqs):
    fdt = np.einsum("f,nk->fnk", freqs, dt, optimize=False)
    A = np.exp(2j * np.pi * fdt)  # (Nfreq, Nslow, Ns)
    return A


@nb.njit(nogil=True, parallel=True)
def compute_Cxy_jit(Xf, weights, scale):
    tapers, Nch, Nt = Xf.shape
    Cxy = np.zeros((Nch, Nch, Nt), dtype=nb.complex64)
    Xfc = Xf.conj()
    for i in nb.prange(Nch):
        for j in nb.prange(i + 1):
            for k in range(tapers):  # Do not prange this one!
                Cxy[i, j] += weights[k] * Xf[k, i] * Xfc[k, j]
            Cxy[i, j] = Cxy[i, j] * (scale[i] * scale[j])
    return Cxy


def CMTM(X, Nw, freq_band=None, fsamp=None):
    # Number of tapers
    K = int(2 * Nw)
    # Number of stations (m), time sampling points (Nx)
    m, Nf = X.shape
    # Next power of 2 (for FFT)
    NFFT = 2 ** int(np.log2(Nf) + 1) + 1

    # Subtract mean (over time axis) for each station
    X_mean = np.mean(X, axis=1)
    X_mean = np.tile(X_mean, [Nf, 1]).T
    X = X - X_mean

    # Compute taper weight coefficients
    tapers, eigenvalues = dpss(N=Nf, NW=Nw, k=K)

    # Compute weights from eigenvalues
    weights = eigenvalues / (np.arange(K) + 1).astype(float)

    # Align tapers with X
    tapers = np.tile(tapers.T, [m, 1, 1])
    tapers = np.swapaxes(tapers, 0, 1)

    # Compute tapered FFT of X
    # Note that X is assumed to be real, so that the negative frequencies can be discarded
    Xf = scipy.fft.rfft(np.multiply(tapers, X), 2 * NFFT, axis=-1)

    # Multitaper power spectrum (not scaled by weights.sum()!)
    Pk = np.abs(Xf) ** 2
    Pxx = np.sum(Pk.T * weights, axis=-1).T
    inv_Px = 1 / np.sqrt(Pxx)

    # If a specific frequency band is given
    if freq_band is not None:
        # Check if the sampling frequency is specified
        if fsamp is None:
            print("When a frequency band is selected, fsamp must be provided")
            return False
        # Compute the frequency range
        freqs = scipy.fft.rfftfreq(n=2 * NFFT, d=1.0 / fsamp)
        # Select the frequency band indices
        inds = (freqs >= freq_band[0]) & (freqs < freq_band[1])
        # Slice the vectors
        Xf = Xf[:, :, inds]
        inv_Px = inv_Px[:, inds]

    # Buffer for covariance matrix
    Ns = Xf.shape[1]
    # Vector for scaling
    scale_vec = inv_Px
    # Compute covariance matrix
    Cxy = compute_Cxy_jit(Xf, weights, scale_vec)
    # Make Cxy Hermitian
    Cxy = Cxy + np.transpose(Cxy.conj(), axes=[1, 0, 2])
    # Add ones to diagonal
    for i in range(Ns):
        Cxy[i, i] = 1

    return Cxy


def noise_space_projection(Rxx, A, sources=1):
    # (Nslow, Nch)
    Nf, Nslow, m = A.shape
    scale = 1.0 / (m * Nf)

    # Total projection onto noise space
    Pm = np.zeros(Nslow, dtype=complex)

    for f in range(Nf):
        Af = A[f]

        # Compute eigenvalues/vectors assuming Rxx is complex Hermitian (conjugate symmetric)
        # Eigenvalues appear in ascending order
        l, v = np.linalg.eigh(Rxx[:, :, f])
        M = sources
        # Extract noise space (size n-M)
        # NOTE: in original code, un was labelled "signal space"!
        un = v[:, : m - M]
        # Precompute un.un*
        Un = np.dot(un, un.conj().T)
        # Project steering vector onto subspace
        Pm += np.einsum("sn, nk, sk->s", Af.conj(), Un, Af, optimize=True)

    return np.real(Pm) * scale
