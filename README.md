# Spherical Texture Extraction 

This toolkit extracts Spherical Textures: Angular projections of 2D or 3D image objects with subsequent spherical harmonics analysis. As described in the [preprint](https://github.com/KoehlerLab/ilastik-sphericaltexture). 

This can be used easily in ilastik, or as a conda installable Python API.

## Usage in ilastik

Spherical textures are available in [ilastik](https://www.ilastik.org). This is available as a feature in the Object Classification pipeline. Further documentation on can be found on [ilastik](https://www.ilastik.org/documentation/objects/objectfeatures), and issues can be logged at the [image.sc forum](https://forum.image.sc).

In brief:
- Open any Object Classification pipeline
- Supply a label-mask/segmentation mask/pixel prediction map (depending on the pipeline)
- In the Feature Selection, select Spherical Texture

Development is done through the [ilastik-sphericaltexture bridge](https://github.com/KoehlerLab/ilastik-sphericaltexture). 


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

## Usage as API

Construct a `SphericalTextureGenerator` object, with the dimensionality of the data and the desired spherical projections and output types.
This stg.process_image() takes an image and binary mask, and returns a dictionary with numpy arrays for each projection and output type. Note that this function expects a single object at a time, and any mask management should be handled upstream from Spherical Texture generation.

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

### Plotting 

Some convenience functions for plotting are bundled, notably for gathering data into longform, with appended spherical harmonics degree, and for plotting condensed spectra. These require the added library `seaborn`. Here an example workflow that also utilizes `scipy`:
```
from scipy.ndimage import find_objects
import numpy as np
import sphericaltexture 
```

for separate objects in one image:
```
img = np.random.randint(0,1000, (10,10,25))
mask = np.concatenate([np.ones((10, 10, 5)) * ix for ix in range(5)], axis=-1).astype('uint32')

stg =  sphericaltexture.SphericalTextureGenerator()
results = []
for obj_id, objslice in enumerate(find_objects(mask)):
    result = stg.process_image(img[objslice], mask[objslice])
    result['obj_id'] = obj_id
    results.append(result)

df = sphericaltexture.list_of_dicts_to_dataframe(results, "Intensity Condensed Spectrum") # select which output
sphericaltexture.plot_condensed_spectra(df, groupkey=None, unitkey='obj_id', palette=None)
```

for separate objects in multiple images
```
# data with 5 objects per image, 3 images at different scales (5-pixel thick objects for convenience of generation)
imgs = []
masks = []
for scale in [10, 20, 30]:
    imgs.append(np.random.randint(0,1000, (scale,scale,25)))
    masks.append(np.concatenate([np.ones((scale, scale, 5)) * ix for ix in range(5)], axis=-1).astype('uint32'))


stg =  sphericaltexture.SphericalTextureGenerator()
results = []
# loop through images and objects and record all in list of dictionaries
for img_id, (img, mask) in enumerate(zip(imgs, masks)):
    for obj_id, objslice in enumerate(find_objects(mask)):
        result = stg.process_image(img[objslice], mask[objslice])
        result['obj_id'] = obj_id
        result['img_id'] = img_id
        results.append(result)

df = sphericaltexture.list_of_dicts_to_dataframe(results, "Intensity Condensed Spectrum") # select which output
sphericaltexture.plot_condensed_spectra(df, groupkey='img_id', unitkey='obj_id', palette=None)
```


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

