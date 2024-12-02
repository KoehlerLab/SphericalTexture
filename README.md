# Spherical Texture Extraction 

This toolkit extracts Spherical Textures: Angular projections of 2D or 3D image objects with subsequent spherical harmonics analysis. As described in the [preprint](https://github.com/KoehlerLab/ilastik-sphericaltexture). 

Spherical Textures can be used in ilastik, or as a conda installable Python API.

![Spherical harmonics illustration](./figures/fig2Artboard%201_1.png "Spherical Texture method design. A) A C. elegans meiotic nucleus in the pachytene stage, stained with DAPI, shown as maximum intensity projections over Z (left) and X, with the YZ view rescaled isotropically (center) and square pixels (right) about the XY view. B) Data from A rescaled to 80x80x80 pixels in XY (left) and YZ (right) views C) A graphic showing the mean intensity projection to spherical space, showing first a subset of the radial rays (left, red lines) used to generate the mean-intensity spherical projection as spherical data and as planar map projection (center). The mean intensities are normalized to mean=0 and variance=1 (right). D) Projections of the spherical harmonics basis functions on the sphere of the first 7 spherical harmonic degrees. E) The spherical harmonics power spectrum of the spherical projection in C shows a distinct peak around approx. the 10th harmonic degree. F) Rescaling the harmonic degrees to approximate wavelength yields a spherical harmonics spectrum, which shows a corresponding peak in the contribution to variance around a wavelength of approx. 0.1 rad/2π G) The standard output of the Spherical Texture method corresponds to the binned spectrum shown in F. Insets show the spherical projection bandpassed to fine, medium and coarse wavelengths, corresponding to the striped lines in the spectrum (7th and 27th harmonic degree). The bandpassed regions reflect the part of the signal quantified by each region of the plot, where the region that shows high variance in the quantification corresponds to the scale of the relative scale of the chromosomes of the data. H) The Spherical Texture extraction is implemented as a Python package and directly in ilastik, allowing for its adoption into the Object Classification workflow. In this workflow, users can interactively train a Random Forest machine learning classifier. Shown here is a part of a C. elegans gonad with segmented nuclei, where some nuclei were labeled as Class 1 and others as Class 2 (solid colors). Based on the Spherical Texture of these labels, ilastik predicts the class of all other nuclei (transparent colors).")
(caption under tooltip)
## Usage in ilastik

Spherical textures are available in [ilastik](https://www.ilastik.org), where it is available as a feature in the Object Classification pipeline. Further documentation for using Spherical Textures in ilastik is available at [ilastik](https://www.ilastik.org/documentation/objects/objectfeatures), and issues can be logged at the [image.sc forum](https://forum.image.sc).

In brief:
- Open any Object Classification pipeline
- Add a label-mask/segmentation mask/pixel prediction map (depending on the pipeline)
- In the Feature Selection, select Spherical Texture

Spherical Textures in ilastik is supported via the [ilastik-sphericaltexture bridge](https://github.com/KoehlerLab/ilastik-sphericaltexture). 

## Python API
### Installing

We recommend installing with `conda` from conda-forge
```
conda create -n sphericaltexture_env -c conda-forge sphericaltexture
```
Note: Consider updating conda if this takes over a minute.


alternatively, one can also install through `pip` with
```
pip install SphericalTexture
```

### Usage as API

Construct a `SphericalTextureGenerator` object, with the dimensionality of the data and the desired spherical projections and output types.
This stg.process_image() takes an image and binary mask, and returns a dictionary with numpy arrays for each projection and output type. Note that this function expects a **single object** at a time without padding, and any mask management should be handled upstream from Spherical Texture generation.

The center of projection can (optionally) be set, with a numpy array with the coordinate of the center of projection in image space. 

```
from sphericaltexture import SphericalTextureGenerator
# all outputs:
stg = SphericalTextureGenerator(
        ndim=3, 
        projections=['Shape','Intensity'], 
        output_types=["Spectrum", "Condensed Spectrum", "Polarization Direction", "Full Projection", "Complex Decomposition"]
    )
results = stg.process_image(imgdata, mask, center_of_projection=np.array([10,10,10]))

# only 20-value Intensity spectrum:
stg = SphericalTextureGenerator()
results = stg.process_image(imgdata, mask)
```
Rays are projected from the center of a rescaled object to capture all angles to make a spherical or circular projection. The values are saved according to the selected projections.

Implemented projections:
- Shape
    - Takes the distance from the center to the edge of the mask 
- Intensity
    - Takes the average intensity along each ray

Subsequently, these values are analyzed and saved according to the selected output type.

Implemented output types:
- Spectrum
    - 1D power spectrum of the Fourier/Spherical Harmonics decomposition of the spherical/circular projection.
- Condensed Spectrum
    - a 20-value version of the `Spectrum` binned along a log2 axis by integration.
- Polarization Direction
    - location of the highest value in the projection in radians
- Full Projection
    - The entire circular or spherical projection
- Complex Decomposition
    - The Fourier/Spherical Harmonics decomposition of the circular/spherical projection.

### Plotting 

Some convenience functions for plotting are bundled, notably for gathering data into long-form, with appended spherical harmonics degree, and for plotting condensed spectra. These require the added library `seaborn`. Here is an example workflow that also utilizes `scipy`:
```
from scipy.ndimage import find_objects
import numpy as np
import sphericaltexture 
```

for separate objects in one image:
```
# Generate example data 
img = np.random.randint(0,1000, (10,10,25))
mask = np.concatenate([np.ones((10, 10, 5)) * ix for ix in range(5)], axis=-1).astype('uint32')

# loop through objects and record all in a dictionary
stg =  sphericaltexture.SphericalTextureGenerator()
results = []
for obj_id, objslice in enumerate(find_objects(mask)):
    result = stg.process_image(img[objslice], mask[objslice])
    result['obj_id'] = obj_id
    results.append(result)

# Plotting
df = sphericaltexture.list_of_dicts_to_dataframe(results, "Intensity Condensed Spectrum") # select which output
sphericaltexture.plot_condensed_spectra(df, groupkey=None, unitkey='obj_id', palette=None)
```

for separate objects in multiple images
```
# Generate example data with 5 objects per image, 3 images at different scales (5-pixel thick objects for convenience of generation)
imgs = []
masks = []
for scale in [10, 20, 30]:
    imgs.append(np.random.randint(0,1000, (scale,scale,25)))
    masks.append(np.concatenate([np.ones((scale, scale, 5)) * ix for ix in range(5)], axis=-1).astype('uint32'))


# loop through images and objects and record all in a list of dictionaries
stg =  sphericaltexture.SphericalTextureGenerator()
results = []
for img_id, (img, mask) in enumerate(zip(imgs, masks)):
    for obj_id, objslice in enumerate(find_objects(mask)):
        result = stg.process_image(img[objslice], mask[objslice])
        result['obj_id'] = obj_id
        result['img_id'] = img_id
        results.append(result)

# Plotting
df = sphericaltexture.list_of_dicts_to_dataframe(results, "Intensity Condensed Spectrum") # select which output
sphericaltexture.plot_condensed_spectra(df, groupkey='img_id', unitkey='obj_id', palette=None)
```


### Rescaling

Objects are rescaled to the size of the `scale` parameter in the creation of the `SphericalTextureGenerator`. This is by default 80: every object processed will be scaled to 80x80x80 pixels before projection to a sphere/circle.  Higher values give more resolution but will take longer to calculate. However, for many applications in biology, going beyond 80 apppears to yield diminishing returns, according to our anecdotal evidence.

### Speed

The first time you run this with a new `shape` or `ndim` parameter a new projection map needs to be constructed. This projection map is cached in a local folder to increase the speed subsequently. 
Spherical Texture generation is optimized for parallel processing, such as in this example:

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

