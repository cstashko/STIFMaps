import numpy as np
import tempfile
import torch
import os
from STIFMaps.STIFMap_generation import AlexNet, generate_STIFMap, collagen_paint, correlate_signals_with_stain
from PIL import Image

class DummyModel(torch.nn.Module):
    def forward(self, x):
        return torch.ones((x.shape[0], 1)) * 0.5  # constant prediction

def create_temp_image_pair():
    dapi = (np.random.rand(512, 512) * 255).astype(np.uint8)
    collagen = (np.random.rand(512, 512) * 255).astype(np.uint8)

    tmp_dir = tempfile.TemporaryDirectory()
    dapi_path = os.path.join(tmp_dir.name, 'dapi.png')
    collagen_path = os.path.join(tmp_dir.name, 'collagen.png')
    Image.fromarray(dapi).save(dapi_path)
    Image.fromarray(collagen).save(collagen_path)

    return dapi_path, collagen_path, tmp_dir

def test_generate_stifmap_dummy_model(monkeypatch):
    dapi, collagen, tmp = create_temp_image_pair()

    # Save dummy model
    model = DummyModel()
    model_path = os.path.join(tmp.name, 'dummy_model.pth')
    torch.save(model.state_dict(), model_path)

    monkeypatch.setattr('STIFMaps.STIFMap_generation.AlexNet', lambda: DummyModel())

    result = generate_STIFMap(
        dapi=dapi,
        collagen=collagen,
        name='test',
        step=32,
        models=[model_path],
        mask=False,
        batch_size=10,
        square_side=64,
        save_dir=False
    )

    assert isinstance(result, np.ndarray)
    assert result.ndim == 3  # 1 model layer, 2D spatial
    assert np.allclose(result, 0.5)

    tmp.cleanup()

def test_generate_STIFMap_runs_with_dummy_model(tmp_path):
    # Setup paths
    dapi_path = "tests/fixtures/no_stain/test1_DAPI.TIF"
    collagen_path = "tests/fixtures/no_stain/test1_collagen.TIF"
    model_path = tmp_path / "dummy_alexnet.pt"

    # Create and save a random-weight AlexNet
    model = AlexNet()
    torch.save(model.state_dict(), model_path)

    # Run the function
    result = generate_STIFMap(
        dapi=dapi_path,
        collagen=collagen_path,
        name="test",
        step=64,
        models=[str(model_path)],
        batch_size=4,
        save_dir=False
    )

    # Basic structural assertions
    assert isinstance(result, np.ndarray)
    assert result.ndim == 3  # One layer per model
    assert result.shape[0] == 1  # We passed one model



def test_collagen_paint_runs(tmp_path):
    dapi_path = "tests/fixtures/no_stain/test1_DAPI.TIF"
    collagen_path = "tests/fixtures/no_stain/test1_collagen.TIF"
    
    # Simulate a fake STIFMap output: shape (1, H, W)
    z = np.random.uniform(0.1, 0.9, size=(1, 10, 10))

    result = collagen_paint(
        dapi=dapi_path,
        collagen=collagen_path,
        z=z,
        name="test_output",
        step=64,
        square_side=224,
        scale_percent=10,  # downscale to make things lighter
        save_dir=str(tmp_path)
    )

    assert isinstance(result, np.ndarray)
    assert result.shape[2] == 3  # Should be RGB image

    output_file = tmp_path / "test_output_STIFMap.png"
    assert output_file.exists()



def test_correlate_signals_with_stain_runs():
    # Inline dummy input setup
    base_dir = os.path.join(os.path.dirname(__file__), 'fixtures', 'with_stain')
    dapi_path = os.path.join(base_dir, 'test1_DAPI.TIF')
    collagen_path = os.path.join(base_dir, 'test1_collagen.TIF')
    stain_path = os.path.join(base_dir, 'test1_stain.TIF')

    # Create a small dummy stiffness map (3 models, 5x5 output)
    z = np.random.rand(3, 5, 5).astype(np.float32)

    # Run the function
    result = correlate_signals_with_stain(
        dapi=dapi_path,
        collagen=collagen_path,
        z=z,
        stain=stain_path,
        step=32,
        square_side=64,
        scale_percent=50
    )

    # Assertions
    assert isinstance(result, tuple)
    assert len(result) == 3
    for corr in result:
        assert isinstance(corr, tuple)
        assert isinstance(corr[0], float)
        assert -1.0 <= corr[0] <= 1.0
