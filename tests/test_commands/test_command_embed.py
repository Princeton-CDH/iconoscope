import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np

from iconoscope.commands import embed
from iconoscope.dataset import ImageDataset
from iconoscope.models import DEFAULT_MODEL


@patch("iconoscope.commands.embed.extract_img_features")
def test_embed_existing_model_features_bail_out(
    mock_extract_features, tmp_path: Path, tmp_image_dir: Path, capsys
):
    # bail out early if requested features already present in the dataset file
    h5_file = tmp_image_dir / "img_dataset.h5"
    img_ds = ImageDataset.create(tmp_image_dir, h5_file)
    img_df = img_ds.get_image_data()
    df = img_df.with_columns(features=np.zeros((img_df.height, 4)))
    mock_extract_features.return_value = df
    embed.main(argparse.Namespace(dataset=h5_file, model=DEFAULT_MODEL.name))
    mock_extract_features.assert_called()
    output = capsys.readouterr().out
    assert "Successfully extracted features from 3 images in 0.00s" in output

    # when model features are present in dataset, should not call
    mock_extract_features.reset_mock()
    embed.main(argparse.Namespace(dataset=h5_file, model=DEFAULT_MODEL.name))
    mock_extract_features.assert_not_called()
    output = capsys.readouterr().out
    assert "already has features" in output
