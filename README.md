# SphericalTexture

This toolkit extracts Spherical Textures: Angular projections of 2D or 3D image objects with subsequent spherical harmonics analysis. 

## Usage 

Construct a `SphericalTextureGenerator` object, with the dimensionality of the data and the desired spherical projections and output types.
This stg.process_image() takes an image and binary mask, and returns a dictionary with numpy arrays for each projection and output type.

```
    from sphericaltexture import SphericalTextureGenerator
    stg = SphericalTextureGenerator(
            ndim=3, 
            projections=['Shape','Intensity'], 
            output_types=["Spectrum", "Polarization Direction", "Full Projection", "Complex Decomposition"]
        )
    results = stg.process_image(imgdata, mask)
```

## Installing

Still work in progress.