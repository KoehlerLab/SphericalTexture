###############################################################################
# Does feature extraction by spherical projection texture
# Does a user-settable projection along rays from the centroid in gauss-legendre quadrature
# Mapping the data to a 1/2D spherical surface
# This is decomposed into a spherical harmonics power spectrum
# The resulting feature is an undersampled (for feature reduction) spectrum
###############################################################################

import numpy as np

# fourier transformations and speedup
from numba import jit, typeof, typed, types
from skimage.transform import resize
from skimage.filters import gaussian
from skimage import img_as_bool
import pyshtools as pysh
from pyshtools.backends.shtools import GLQGridCoord
from pyshtools.expand import SHExpandGLQ
from pyshtools.spectralanalysis import spectrum
import scipy
import ducc0

import threading
from functools import lru_cache

# saving/loading LUT
import pickle as pickle
from pathlib import Path
from appdirs import user_cache_dir

import time
import matplotlib.pyplot as plt

_condition = threading.RLock()
pysh.backends.select_preferred_backend(backend="ducc", nthreads=1)

class SphericalTextureGenerator():
    def __init__(self, ndim=3, projections=['Intensity'], output_types=['Condensed Spectrum'], scale=80 ):
        self.ndim = ndim
        self.optional_projections = [
            "Intensity", # mean intensity
            "Shape",
        ] 
        self.optional_output_types = [
            'Spectrum',
            'Condensed Spectrum',
            'Polarization Direction',
            'Full Projection',
            'Complex Decomposition'
        ]

        self.selected_projections = np.zeros(len(self.optional_projections), dtype=bool)  # contains selection of which projections should be done
        self.features = []
        for proj in projections:
            if proj not in self.optional_projections:
                raise ValueError(f"unexpected projection {projection}")
            self.selected_projections[self.optional_projections.index(proj)] = True
            for output_type in output_types:
                if output_type not in self.optional_output_types:
                    raise ValueError(f"unexpected output type {output_type}")
                self.features.append(f"{proj} {output_type}")
        self.raysLUT = None
        self.bin_start, self.bin_ends, self.n_coarse = None, None, None

        # Hyperparameters
        self.scale = scale  # transforms to cube of size scale by scale by scale - projections are scaled to sample π*scale
        self.reduced_spectrum_length = 20
        return

    def process_image(self, image, binary_mask):
        return self.unwrap_and_expand(image, binary_mask)

    def unwrap_and_expand(self, image, binary_bbox):
        t0 = time.time()
        rawbbox = image
        mask_object = binary_bbox

        if self.raysLUT == None:
            with _condition:
                self.raysLUT = self.get_ray_table(self.ndim)

        # resizing of data is also done in 3D for 2D data
        cube = resize(image, (self.scale, self.scale, self.scale), preserve_range=True, order=1)

        mask_cube = resize(mask_object != 0, tuple([self.scale] * len(image.shape)), order=0)

        segmented_cube = np.where(mask_cube, cube, -1)

        t1 = time.time()

        unwrapped = _st_lookup(segmented_cube, self.raysLUT, int(np.pi * self.scale), self.selected_projections)
        # print(unwrapped)
        t2 = time.time()

        result = {}
        projectedix = 0
        used_projections = [which_proj for which_proj, projected in enumerate(self.selected_projections) if projected]
        for projectedix, projection in enumerate(unwrapped):
            which_proj = self.optional_projections[used_projections[projectedix]]

            if self.ndim == 2:
                projection = projection[0, : int(self.scale * np.pi)]

            if np.max(projection) != 0:
                projection /= np.std(projection)
            projection -= np.mean(projection)

            projectedix += 1
            if self.ndim == 2:
                coeffs = scipy.fft.fft(projection)
                power = np.abs(coeffs * coeffs.conjugate() / len(projection) ** 2)
                power = power[: (len(projection) // 2) + 1] * 2  # keep area under curve 1, remove symmetry
            else:
                zero, w = pysh.expand.SHGLQ(int(np.pi * self.scale))
                coeffs = pysh.expand.SHExpandGLQ(projection, w=w, zero=zero)
                power = spectrum(coeffs, unit="per_l")
                

            # bin higher degrees in 2log spaced bins:
            if self.n_coarse is None:
                self.get_bins(len(power))
            means = [np.sum(power[s:e]) for s, e in zip(self.bin_start, self.bin_ends)]
            freqs = scipy.fft.fftfreq(251)[: (len(projection) // 2) + 1]
            

            for output_type in self.optional_output_types:
                projfeatname = f"{which_proj} {output_type}"
                if projfeatname in self.features:
                    if output_type == 'Full Projection':
                        result[projfeatname] = projection
                    if output_type == 'Polarization Direction':
                        # optional: add bandpassing here
                        peak = np.unravel_index(np.argmax(projection, axis=None), projection.shape)
                        peak /= np.array(projection.shape)
                        peak *= np.pi * 2
                        result[projfeatname] = peak
                    if output_type == 'Condensed Spectrum':
                        result[projfeatname] = np.concatenate([power[: self.n_coarse], np.array(means)])
                    if output_type == 'Spectrum':
                        result[projfeatname] = power
                    if output_type == 'Complex Decomposition':
                        result[projfeatname] = coeffs
        t3 = time.time()

        # print("time to do full unwrap and expand: \t", t3 - t0)
        return result
    
    def save_ray_table(self, fpath, rays):
        # save a pickle of the rayLUT, requires retyping of the dictionary
        # this is because the rays are ragged, and not single-length
        outLUT = {}  # need to un-type the dictionary for pickling
        for k, v in rays.items():
            outLUT[k] = v
        with open(fpath, "wb") as ofs:
            pickle.dump(outLUT, ofs, protocol=pickle.HIGHEST_PROTOCOL)
        return

    def get_bins(self, veclength):
        # increase bins until self.reduced_spectrum_length is hit with integer bins
        # all linearly scaled bins will be 'coarse features'
        n_bins = self.reduced_spectrum_length
        bins = np.unique(np.logspace(0, np.log2(veclength - 1), num=n_bins, base=2, endpoint=True).astype(int))
        while len(bins) < self.reduced_spectrum_length:
            n_bins += 1
            bins = np.unique(np.logspace(0, np.log2(veclength - 1), num=n_bins, base=2, endpoint=True).astype(int))

        self.n_coarse = np.argmax(bins - (np.arange(len(bins)) + 1) > 0)
        self.bin_ends = bins[self.n_coarse :]
        self.bin_start = np.roll(bins, 1)[self.n_coarse :]
        return
    
    @lru_cache
    def get_ray_table(self, ndim):
        if ndim is None:
            raise ValueError("ndim cannot be None")
        # try to load or generate new
        # loading requires retyping of the dictionary for numba, which slows it down (see: https://github.com/numba/numba/issues/8797)
        fpath = Path(user_cache_dir('SphericalTexture', "OG")) / f"sphericalLUT{self.scale}_{ndim}D.pickle"
        fpath.parent.mkdir(parents=True, exist_ok=True)
        try:
            t0 = time.time()
            with open(fpath, "rb") as handle:
                newLUT = pickle.load(handle) #change? 
            typed_rays = typed.Dict.empty(
                key_type=typeof((1, 1)),
                value_type=typeof(np.zeros((1, 3), dtype=np.int32)),
            )
            for k, v in newLUT.items():
                typed_rays[k] = v
            # print("loaded ray table in: ", time.time() - t0)
            return typed_rays
        except Exception as e:
            # print("recalculating LUT")
            t0 = time.time()
            rays = typed.Dict.empty(
                key_type=typeof((1, 1)),
                value_type=typeof(np.zeros((1, 3), dtype=np.int32)),
            )
            _st_fill_ray_table(self.scale, GLQGridCoord(int(np.pi * self.scale)), rays, ndim)
            self.save_ray_table(fpath, rays)
            t1 = time.time()
            # print("time to make ray table: ", t1 - t0)
            return rays



# All numba-accelerated functions cannot receive self, so are not class functions

# this traverses all rays
# Could theoretically be improved to a tree-data-structure, but this adds overhead
@jit(nopython=True, nogil=True)
def _st_lookup(img, raysLUT, fineness, projections):
    unwrapped = np.zeros((np.sum(projections), fineness + 1, fineness * 2 + 1), dtype=np.float64)
    for loc, ray in raysLUT.items():
        values = np.zeros(ray.shape[0])
        for ix, voxel in enumerate(ray):
            values[ix] = img[voxel[0], voxel[1], voxel[2]]
            if values[ix] < 0:  # quit when outside of object mask -  all outside of mask are set to -1
                if ix != 0:  # centroid is not outside of mask
                    values = values[:ix]
                    break
        proj = 0
        ray = ray.astype(np.float64)
        unwrapped[:, loc[1], loc[0]] = _st_project(ray, values, projections)
    return unwrapped


@jit(nopython=True, nogil=True)
def _st_project(ray, values, projections):
    vals = np.zeros(np.sum(projections), dtype=np.float64)
    proj = 0
    if projections[0]:  # MEAN
        vals[proj] = np.sum(values) / len(values)
        proj += 1
    if projections[1]:  # SHAPE
        vec = ray[0].astype(np.float64) - ray[len(values) - 1].astype(np.float64)
        vec -= vec < 0  # integer flooring issues
        vals[proj] = np.linalg.norm(vec)
        proj += 1
    return vals


# ---- only used in generating LUT ----


@jit(nopython=True)
def _st_fill_ray_table(scale, GLQcoords, rays, ndim):
    centroid = np.array([scale, scale, scale], dtype=np.float32) / 2.0
    if ndim == 2:
        glq_lon = np.linspace(0, 2 * np.pi, int(scale * np.pi))
        glq_lat = np.array([0.0])
    else:
        glq_lat, glq_lon = np.deg2rad(GLQcoords[0]), np.deg2rad(GLQcoords[1])

    for phi_ix, lon in enumerate(glq_lon):
        for theta_ix, lat in enumerate(glq_lat):
            ray = np.array(
                [
                    np.sin((np.pi / 2) - lat) * np.cos(lon),
                    np.sin((np.pi / 2) - lat) * np.sin(lon),
                    np.cos((np.pi / 2) - lat),
                ]
            )

            pixels = _st_march(ray, centroid, scale, marchlen=0.003).astype(np.int32).copy()

            # find unique pixels and keep order

            # NUMBA-fied version of: different = np.any(pixels[:-1, :] - pixels[1:, :],axis=1)
            different = np.array([np.any(difference) for difference in (pixels[:-1, :] - pixels[1:, :])])

            nz = np.nonzero(different)[0]
            unique_pixels = np.zeros((nz.shape[0] + 1, 3), dtype=np.int32)
            unique_pixels[0, :] = pixels[0]
            for ix, val in enumerate(nz):  # ix+1 this is because np.insert doesnt njit
                unique_pixels[ix + 1, :] = pixels[val + 1]

            rays[(phi_ix, theta_ix)] = unique_pixels

    return rays


@jit(nopython=True)
def _st_march(ray, centroid, scale, marchlen):
    increment = ray * marchlen
    distances = []
    normals = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])]
    bounds = [np.array([scale, scale, scale]).astype(np.float64) - 0.4, np.array([0.0, 0.0, 0.0])]
    for normal in normals:
        for bound in bounds:
            intersect = _st_isect_dist_line_plane(centroid, ray, bound, normal)
            distances.append(intersect)
    est_length = min(distances) / marchlen
    end = est_length * increment + centroid
    pixels = (
        np.linspace(centroid[0], end[0], int(est_length)),
        np.linspace(centroid[1], end[1], int(est_length)),
        np.linspace(centroid[2], end[2], int(est_length)),
    )
    pixels = np.stack(pixels).T
    return pixels


# intersection function edited from https://stackoverflow.com/questions/5666222/3d-line-plane-intersection
@jit(nopython=True)
def _st_isect_dist_line_plane(centroid, raydir, planepoint, planenormal, epsilon=1e-6):
    dot = np.dot(planenormal, raydir)
    if np.abs(dot) > epsilon:
        w = centroid - planepoint
        fac = -np.dot(planenormal, w) / dot
        if fac > 0:
            return fac
    return np.inf  # parallel ray and plane

