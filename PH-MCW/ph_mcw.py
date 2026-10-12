# -*- coding: utf-8 -*-
"""
PH-MCW: Prior-Histogram Marker-Controlled Watershed for SAR flood-water extraction.

Implementation accompanying the paper:
"Paddy Rice Inundation Mapping Based on Multi-Resolution Remote Sensing Imagery and
 Semantic Segmentation with a Dual-Attention and Edge-Aware Attention Network".

Algorithm (matching Section 3.2.3 of the manuscript):
  1. Water seeds:  pixels whose backscatter is below the prior water percentile T_water.
     T_water is defined from the prior water proportion of the study region
     (publicly released land-survey data, inland water bodies only, refined by
     visual inspection of the scene). The adopted prior percentiles are:
     Yingde 3.2%, Lechang 2.2%, Pingyuan 2.6%, Taishan 8.4%.
  2. Non-water seeds: pixels above T_non-water, the mode of the backscatter
     histogram (256 equi-width bins, smoothed with a 1-D Gaussian filter,
     sigma = 3 bins), which represents the typical backscatter of non-water
     surfaces occupying most pixels.
  3. Marker-controlled watershed on the backscatter INTENSITY image (no gradient
     image is used; water corresponds to low-elevation basins), applied
     bidirectionally: a forward pass on the intensity and a backward pass on its
     inverse (max - intensity), both with the same seeds and compactness = 0.
  4. Conflict arbitration: pixels classified as water by the forward pass but
     non-water by the backward pass are assigned to the nearer seed set
     (Euclidean distance transform).

Polarization: the cross-polarized channel was used (VH for Sentinel-1,
HV for GF-3), because the specular return of water is much weaker at
cross-polarization.

Usage:
  python ph_mcw.py --input sar_image.tif --prior 3.2 --output water.tif
  python ph_mcw.py --input sar_image.tif --prior 3.2 --aoi aoi_mask.tif --output water.tif

Dependencies: numpy, scipy, scikit-image, rasterio
"""

import argparse

import numpy as np
import rasterio
from rasterio.features import sieve
from scipy.ndimage import distance_transform_edt, gaussian_filter1d
from skimage.segmentation import watershed

BINS = 256          # equi-width histogram bins
SIGMA = 3.0         # Gaussian smoothing of the histogram, in bins
MIN_SEEDS = 50      # minimum number of water-seed pixels


def histogram_peak(values, bins=BINS, sigma=SIGMA):
    """Mode of the smoothed backscatter histogram (T_non-water)."""
    h, be = np.histogram(values, bins=bins)
    bc = (be[:-1] + be[1:]) / 2
    hs = gaussian_filter1d(h.astype(np.float64), sigma=sigma)
    return float(bc[np.argmax(hs)])


def ph_mcw(sar, valid, prior_percent):
    """
    Extract water from a SAR backscatter image.

    Parameters
    ----------
    sar : 2-D ndarray (float32)
        Backscatter image (DN for GF-3 L1A, dB for calibrated products).
    valid : 2-D boolean ndarray
        Valid-data mask (non-NaN / nodata pixels).
    prior_percent : float
        Prior water proportion in percent (e.g. 3.2 for Yingde).

    Returns
    -------
    water : 2-D boolean ndarray
        Extracted water mask.
    t_water : float
        Backscatter value at the prior percentile.
    t_nonwater : float
        Mode of the smoothed histogram.
    """
    t_water = float(np.percentile(sar[valid], prior_percent))
    t_nonwater = histogram_peak(sar[valid])

    markers = np.zeros(sar.shape, dtype=np.int32)
    markers[(sar < t_water) & valid] = 1      # water seeds
    markers[(sar > t_nonwater) & valid] = 2   # non-water seeds
    if np.sum(markers == 1) < MIN_SEEDS:
        return np.zeros(sar.shape, bool), t_water, t_nonwater

    # Intensity used directly as the watershed topographic surface.
    maxv = float(np.nanmax(sar[valid]))
    surface = np.where(np.isnan(sar), maxv + 0.5, sar)
    inverse = maxv - surface

    # Bidirectional watershed: forward on intensity, backward on its inverse.
    forward = watershed(surface, markers.copy(), mask=valid, compactness=0)
    backward = watershed(inverse, markers.copy(), mask=valid, compactness=0)
    water_f = (forward == 1) & valid
    nonwater_b = (backward == 2) & valid

    conflict = water_f & nonwater_b
    if np.sum(conflict) > 0:
        # Assign conflicting pixels to the nearer seed set.
        d_water = distance_transform_edt(~(markers == 1))
        d_nonwater = distance_transform_edt(~(markers == 2))
        water = water_f & ~conflict
        water[conflict] = d_water[conflict] < d_nonwater[conflict]
    else:
        water = water_f

    return water, t_water, t_nonwater


def extract(input_path, prior_percent, output_path, aoi_path=None,
            min_valid=-100.0, min_size=64):
    """Read a GeoTIFF, run PH-MCW, write the water mask as GeoTIFF."""
    with rasterio.open(input_path) as src:
        arr = src.read(1).astype(np.float32)
        profile = src.profile.copy()

    if arr.dtype == np.uint16:
        valid = arr > 0
    else:
        valid = arr > min_valid
        arr[~valid] = np.nan

    water, t_water, t_nonwater = ph_mcw(arr, valid, prior_percent)

    if aoi_path is not None:
        with rasterio.open(aoi_path) as src:
            aoi = src.read(1).astype(np.float32)
        # AOI: 1 = candidate flood-prone area, anything else excluded.
        water &= (aoi == 1)

    if min_size and min_size > 0:
        water = sieve(water.astype(np.uint8), size=min_size).astype(bool)

    profile.update(dtype=rasterio.uint8, count=1, compress="lzw", nodata=0)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(water.astype(np.uint8), 1)
        dst.set_band_description(1, "water mask (1 = water)")

    print("prior percentile : %.2f %%" % prior_percent)
    print("T_water          : %.4f" % t_water)
    print("T_non-water      : %.4f" % t_nonwater)
    print("water pixels     : %d" % int(np.sum(water)))
    print("output           : %s" % output_path)


def main():
    ap = argparse.ArgumentParser(description="PH-MCW SAR water extraction")
    ap.add_argument("--input", required=True, help="input SAR backscatter GeoTIFF")
    ap.add_argument("--prior", required=True, type=float,
                    help="prior water proportion in percent (e.g. 3.2)")
    ap.add_argument("--output", required=True, help="output water-mask GeoTIFF")
    ap.add_argument("--aoi", default=None,
                    help="optional AOI GeoTIFF (1 = candidate flood-prone area)")
    ap.add_argument("--min-valid", type=float, default=-100.0,
                    help="minimum valid backscatter (dB products only)")
    ap.add_argument("--min-size", type=int, default=64,
                    help="remove connected components smaller than this (pixels)")
    args = ap.parse_args()
    extract(args.input, args.prior, args.output, args.aoi,
            args.min_valid, args.min_size)


if __name__ == "__main__":
    main()
