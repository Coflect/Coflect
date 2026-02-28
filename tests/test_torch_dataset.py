from pytest import raises

from coflect.modules.hitl.common.torch_dataset import DatasetConfig, build_torch_dataset


def test_build_synthetic_dataset_has_expected_shape_attrs() -> None:
    ds = build_torch_dataset(
        DatasetConfig(
            name="synthetic",
            root="./data",
            split="train",
            download_data=False,
        )
    )
    assert len(ds) > 0
    assert int(ds.num_classes) == 10
    assert int(ds.image_size) == 64

    x, y, sample_idx = ds[0]
    assert tuple(x.shape) == (3, 64, 64)
    assert int(y.item()) >= 0
    assert int(sample_idx) == 0


def test_build_dataset_rejects_unknown_name() -> None:
    with raises(ValueError):
        build_torch_dataset(  # type: ignore[arg-type]
            DatasetConfig(
                name="unknown",  # type: ignore[arg-type]
                root="./data",
                split="train",
                download_data=False,
            )
        )
