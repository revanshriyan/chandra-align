"""Phase-congruency front-end (Phase 14).

Log-Gabor phase congruency after Kovesi (1999), reimplemented from scratch
with numpy/cv2 (no torch). Phase congruency marks image structure by the
alignment of Fourier phases across scales instead of by gradient magnitude,
so it is invariant to global brightness/contrast changes: a crater rim is a
crater rim whether the sun lights it from the left or the right, and whether
the sensor gains differ.

That is exactly the IIRS<->TMC-2 modality gap: SIFT descriptors keyed on
gradient polarity collapse across sensors, while phase congruency keys on
structure without polarity. The PC map can be fed to ordinary detectors and
descriptors (see pc_arm), making the rest of the pipeline sensor-agnostic.

API
---
phase_congruency(img, nscale=4, norient=6) -> (pc, orientation)
    img : 2-D grayscale array (any dtype; normalized internally).
    pc  : 2-D float64 map, sum of per-orientation congruency in [0, ~nscale].
    orientation : 2-D float64 map of dominant structure orientation in [0, pi).

Algorithm notes
---------------
For each orientation and each scale, a complex Log-Gabor filter pair
(even/odd) is applied in the frequency domain. With (e, o) the responses and
A = hypot(e, o):

  meanE = sum(e*A)/sum(A),  meanO = sum(o*A)/sum(A)   (energy-weighted mean phase)
  A*dPhi = e*meanE + o*meanO - |e*meanO - o*meanE|     (phase deviation term)

  PC_o = sum_s W * max(A*dPhi - T, 0) / (sum_s A + eps)

W is a sigmoid weighting on the frequency spread of the response (narrow
frequency response = likely noise), T is a noise threshold estimated from
the smallest scale assuming Rayleigh-distributed noise energy. Defaults
follow Kovesi's: min wavelength 3 px, scale multiplier 2.1, sigmaOnf 0.55.
"""

import cv2
import numpy as np


def _log_gabor_bank(rows, cols, nscale, norient,
                    min_wavelength=3.0, mult=2.1, sigma_onf=0.55,
                    dtheta_on_sigma=1.5):
    """Build complex Log-Gabor filters in FFT layout (not fftshifted)."""
    # Normalized frequency grid in [-0.5, 0.5], fftshifted convention.
    yr = np.linspace(-0.5, 0.5, rows, endpoint=False)
    xr = np.linspace(-0.5, 0.5, cols, endpoint=False)
    x, y = np.meshgrid(xr, yr)
    radius = np.sqrt(x ** 2 + y ** 2)
    radius[rows // 2, cols // 2] = 1.0  # dummy: avoid log(0); zeroed below
    theta = np.arctan2(-y, x)

    theta_sigma = np.pi / (norient * dtheta_on_sigma)
    log_sigma = np.log(sigma_onf)

    bank = []
    for o in range(norient):
        theta0 = o * np.pi / norient
        ds = np.sin(theta) * np.cos(theta0) - np.cos(theta) * np.sin(theta0)
        dc = np.cos(theta) * np.cos(theta0) + np.sin(theta) * np.sin(theta0)
        dtheta = np.abs(np.arctan2(ds, dc))
        angular = np.exp(-dtheta ** 2 / (2.0 * theta_sigma ** 2))
        for s in range(nscale):
            wavelength = min_wavelength * mult ** s
            fo = 1.0 / wavelength
            log_radial = np.exp(
                -(np.log(radius / fo)) ** 2 / (2.0 * log_sigma ** 2))
            log_radial[radius > 0.5] = 0.0      # beyond Nyquist
            log_radial[rows // 2, cols // 2] = 0.0  # no DC response
            filt = log_radial * angular
            bank.append(np.fft.ifftshift(filt))
    return bank


def _rayleigh_noise_threshold(smallest_amplitude, k=2.0):
    """Noise threshold T from smallest-scale amplitudes.

    Assumes noise amplitudes follow a Rayleigh distribution: sigma is fit
    from the median (robust), then T = mean + k*std. Documented
    simplification of Kovesi's vector-sum noise model.
    """
    med = float(np.median(smallest_amplitude)) + 1e-12
    sigma = med / np.sqrt(2.0 * np.log(2.0))          # Rayleigh from median
    noise_mean = sigma * np.sqrt(np.pi / 2.0)
    noise_var = (2.0 - np.pi / 2.0) * sigma ** 2
    return noise_mean + k * np.sqrt(noise_var)


def phase_congruency(img, nscale=4, norient=6):
    """Compute Log-Gabor phase congruency of a grayscale image.

    Parameters
    ----------
    img : (H, W) array-like
        Single-channel image, any numeric dtype.
    nscale, norient : int
        Number of Log-Gabor scales and orientations.

    Returns
    -------
    pc : (H, W) float64
        Phase-congruency map (sum over orientations). Higher = stronger
        structure, independent of local contrast or brightness.
    orientation : (H, W) float64
        Dominant structure orientation in [0, pi), PC-weighted.
    """
    a = np.asarray(img, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError(f"phase_congruency expects a 2-D image, got {a.shape}")
    rows, cols = a.shape
    a = a - a.mean()

    bank = _log_gabor_bank(rows, cols, nscale, norient)
    ft = np.fft.fft2(a)

    pc_total = np.zeros((rows, cols))
    ox = np.zeros((rows, cols))
    oy = np.zeros((rows, cols))
    eps = 1e-9

    for o in range(norient):
        responses = []  # (even, odd, amplitude) per scale
        for s in range(nscale):
            resp = np.fft.ifft2(ft * bank[o * nscale + s])
            e, od = resp.real, resp.imag
            responses.append((e, od, np.hypot(e, od)))
        # energy-weighted mean phase vector over scales, NORMALIZED to unit
        # length: this normalization is what makes the deviation term carry
        # units of amplitude (not amplitude-squared), giving contrast
        # invariance. Without it, PC scales with image contrast.
        e_w = sum(e * amp for e, od, amp in responses)
        o_w = sum(od * amp for e, od, amp in responses)
        a_w = sum(amp for e, od, amp in responses)
        m_mag = np.hypot(e_w, o_w) + eps
        mean_e = (e_w / (a_w + eps)) / (m_mag / (a_w + eps))
        mean_o = (o_w / (a_w + eps)) / (m_mag / (a_w + eps))
        # i.e. mean_e, mean_o = unit vector along (e_w, o_w)
        # phase deviation via the mean-phase-vector trick
        dev_num = np.zeros((rows, cols))
        sum_a = np.zeros((rows, cols))
        max_a = np.zeros((rows, cols))
        for e, od, amp in responses:
            dev_num += (e * mean_e + od * mean_o
                        - np.abs(e * mean_o - od * mean_e))
            sum_a += amp
            max_a = np.maximum(max_a, amp)
        small_a = responses[0][2]
        # frequency-spread weighting: narrow response across scales ~= noise
        spread = sum_a / (max_a + eps)              # in [1, nscale]
        spread_frac = spread / nscale               # in (0, 1]
        weight = 1.0 / (1.0 + np.exp(10.0 * (0.4 - spread_frac)))
        thresh = _rayleigh_noise_threshold(small_a)
        pc_o = weight * np.maximum(dev_num - thresh, 0.0) / (sum_a + eps)
        pc_total += pc_o
        ang = o * np.pi / norient
        ox += pc_o * np.cos(2.0 * ang)
        oy += pc_o * np.sin(2.0 * ang)

    orientation = 0.5 * np.mod(np.arctan2(oy, ox), 2.0 * np.pi)
    orientation = np.mod(orientation, np.pi)
    return pc_total, orientation
