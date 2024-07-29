# Spherical Texture Extraction 

This toolkit extracts Spherical Textures: Angular projections of 2D or 3D image objects with subsequent spherical harmonics analysis. 

## Installing

Installing can best be done through `conda` with
```
conda install -c conda-forge sphericaltexture
```
or to create a dedicated environment:
```
conda create -n sphericaltexture_env -c conda-forge sphericaltexture
```
Note, this is a lot faster with an updated version of conda! If it takes over a minute, consider updating conda.


alternatively, one can also install through `pip` with
```
pip install SphericalTexture
```

## Usage 

Construct a `SphericalTextureGenerator` object, with the dimensionality of the data and the desired spherical projections and output types.
This stg.process_image() takes an image and binary mask, and returns a dictionary with numpy arrays for each projection and output type.

```
    from sphericaltexture import SphericalTextureGenerator
    # all outputs:
    stg = SphericalTextureGenerator(
            ndim=3, 
            projections=['Shape','Intensity'], 
            output_types=["Spectrum", "Condensed Spectrum", "Polarization Direction", "Full Projection", "Complex Decomposition"]
        )
    results = stg.process_image(imgdata, mask)

    # only 20-value Intensity spectrum:
    stg = SphericalTextureGenerator()
    results = stg.process_image(imgdata, mask)
```
Rays are projected from the center of a rescaled object to capture all angles to make a spherical or circular projection. The values are saved according to the selected projections.

Implemented projections:
- Intensity
    - Takes the average intensity along each ray
- Shape
    - Takes the distance from the center to the edge of the mask 

Subsequently, these are analyzed and saved according to the selected output type.

Implemented output types:
- Spectrum
    - 1D power spectrum of the Fourier/Spherical Harmonics decomposition of the spherical/circular projection.
- Condensed Spectrum
    - a 20-value version of the `Spectrum` binned along a log2 axis by integration.
- Polarization Direction
    - location in rad of the highest value in the projection
- Full Projection
    - The entire circular or spherical projection
- Complex Decomposition
    - The Fourier/Spherical Harmonics decomposition of the spherical/circular projection.

## Rescaling

Objects are rescaled to the size of the `scale` parameter in the creation of the `SphericalTextureGenerator`. This is by default 80: every object processed will be scaled to 80x80x80 pixels before projection to a sphere/circle.  Lower values give more speed, higher gives more resolution. However, for many applications in biology, going much higher than 80 seems anecdotally to give diminishing returns.

## Speed

The first time you run this with a new `shape` or `ndim` parameter a new projection map needs to be constructed. This is afterwards cached in a local folder, and should then subsequently be fast. 
Spherical Texture generation is optimized for parallel processing, such as this example:

```
from concurrent import futures
import numpy as np
from sphericaltexture import SphericalTextureGenerator

# 100 test images of 34x35x36 pixels
your_images = [np.random.randint(1,1e6, size=(34,35,36)) for i in range(100)]
your_masks = [np.ones((34,35,36)) for i in range(100)]

stg = SphericalTextureGenerator(
            projections=['Intensity'], 
            output_types=["Spectrum"]
        )

with futures.ThreadPoolExecutor(max_workers=10) as executor:
    jobs = [executor.submit(stg.process_image, imgdata, mask) for imgdata, mask in zip(your_images, your_masks)]
    all_results = [fut.result() for fut in futures.as_completed(jobs)]
```

