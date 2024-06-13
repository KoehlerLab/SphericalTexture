from sphericaltexture import SphericalTextureGenerator
import tifffile, os
from pathlib import Path
import pytest

test_folder = Path(os.path.abspath(Path(__file__).parent / 'test_data'))

@pytest.mark.parametrize('ndim', [2, 3])
@pytest.mark.parametrize('projections', [['Shape'], ['Intensity'], ['Shape', 'Intensity']])
def test_sphericaltexture(snapshot, ndim, projections):
    with tifffile.TiffFile(test_folder / 'data_2D.tif') as ifs:
        imgdata = ifs.asarray()
    with tifffile.TiffFile(test_folder /'data_2D.tif') as ifs:
        mask = ifs.asarray()

    stg = SphericalTextureGenerator(ndim=2, projections=['Shape','Intensity'], output_types=["Spectrum", "Polarization Direction", "Full Projection", "Complex Decomposition"])

    results = stg.process_image(imgdata, mask)

    for key in results:
        snapshot.assert_match(str(results[key]), f'{key}_{ndim}D')


