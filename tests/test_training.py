import torch
import pandas as pd
import os
import numpy as np
from torch.utils.data import Dataset
from STIFMaps.training import CustomImageDataset, AlexNet, train_model
from PIL import Image
import tempfile

def create_dummy_dataset_dir(num_samples=10):
    temp_dir = tempfile.TemporaryDirectory()

    sample_names = [f'img{i+1}' for i in range(num_samples)]
    file_names = [f'{name}.png' for name in sample_names]
    stiffness_vals = np.round(np.random.uniform(0.1, 1.0, size=num_samples), 2)

    df = pd.DataFrame({
        'Sample': sample_names,
        'Key': file_names,
        'Stiffness': stiffness_vals
    })

    for fname in df['Key']:
        # 3-channel RGB image
        img = (np.random.rand(224, 224, 3) * 255).astype(np.uint8)
        Image.fromarray(img).save(os.path.join(temp_dir.name, fname))

    return df, temp_dir


def test_custom_image_dataset_getitem():
    df, temp_dir = create_dummy_dataset_dir()

    dataset = CustomImageDataset(df=df, img_dir=temp_dir.name)

    # Should have 10 items
    assert len(dataset) == 10

    image, label = dataset[0]

    # Image shape should be (3, 224, 224) after moveaxis
    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, 224, 224)
    assert isinstance(label, float) or isinstance(label, np.floating)

    temp_dir.cleanup()


def test_alexnet_forward_pass():
    model = AlexNet(num_classes=2)
    dummy_input = torch.randn(1, 3, 224, 224)  # 1 image, RGB, 224x224
    output = model(dummy_input)
    assert output.shape == (1, 2)  # batch_size x num_classes


def test_train_model_runs_minimal_case():
    df, temp_dir = create_dummy_dataset_dir()

    try:
        train_model(
            df=df,
            img_dir=temp_dir.name,
            name='test_model',
            brightness_range=(0.9, 1.1),
            contrast_range=(0.9, 1.1),
            sharpness_range=(0.9, 1.1),
            batch_size=2,
            n_epochs=1,
            learning_rate=1e-4,
            weight_decay=0,
            save_directory=False,
            save_visualizations=False
        )
        assert True  # test passes if no exception is raised
    finally:
        temp_dir.cleanup()
