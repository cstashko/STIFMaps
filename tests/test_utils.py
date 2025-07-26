import numpy as np
from STIFMaps.misc import norm_pic

def test_norm_pic_basic_behavior():
    # Make an image with known spread
    im = np.linspace(0, 100, 10000).reshape(100, 100)
    norm = norm_pic(im)

    assert norm.shape == im.shape
    assert norm.min() >= 0.0
    assert norm.max() <= 1.0

    # Check approximate location of clipped values
    # Bottom 1% should be ~0
    assert np.isclose(norm[0, 0], 0.0, atol=1e-2)
    # Top 1% should be ~1
    assert np.isclose(norm[-1, -1], 1.0, atol=1e-2)

def test_norm_pic_uniform_image():
    im = np.ones((50, 50)) * 42
    norm = norm_pic(im)
    assert np.all(norm == 0.0)  # hmax == hmin → divide-by-zero → clip to 0

def test_norm_pic_negative_values():
    im = np.array([[-5, 0], [5, 10]], dtype=float)
    norm = norm_pic(im)
    assert norm.shape == im.shape
    assert np.all(norm >= 0.0) and np.all(norm <= 1.0)