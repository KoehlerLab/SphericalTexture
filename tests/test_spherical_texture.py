from sphericaltexture import SphericalTextureGenerator
import tifffile, os
from pathlib import Path
import pytest

test_folder = Path(os.path.abspath(Path(__file__).parent / 'test_data'))

@pytest.mark.parametrize('ndim', [2, 3])
@pytest.mark.parametrize('projections', [['Shape'], ['Intensity'], ['Shape', 'Intensity']])
def test_sphericaltexture(snapshot, ndim, projections):
    with tifffile.TiffFile(test_folder / f'data_{ndim}D.tif') as ifs:
        imgdata = ifs.asarray()
    with tifffile.TiffFile(test_folder /f'mask_{ndim}D.tif') as ifs:
        mask = ifs.asarray()
    all_output_types = ["Spectrum", "Polarization Direction", "Full Projection", "Complex Decomposition", "Condensed Spectrum"]
    stg = SphericalTextureGenerator(ndim=ndim, projections=projections, output_types=all_output_types)


    results = stg.process_image(imgdata, mask)
    assert all([output_type in '\t'.join(results.keys()) for output_type in all_output_types])
    for key in results:
        snapshot.assert_match(str(results[key]), f'{key}_{ndim}D')




