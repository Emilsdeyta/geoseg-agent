import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("segmentation_models_pytorch")

from geoseg.models.siamese_unet import SiameseUNet  # noqa: E402

SMALL = {"encoder_name": "resnet18", "encoder_weights": None}


@pytest.mark.parametrize("fusion", ["diff", "concat"])
@pytest.mark.parametrize("size", [(64, 64), (96, 80)])
def test_output_shape(fusion: str, size: tuple[int, int]) -> None:
    model = SiameseUNet(**SMALL, fusion=fusion).eval()  # type: ignore[arg-type]
    image_a, image_b = torch.randn(2, 3, *size), torch.randn(2, 3, *size)
    with torch.no_grad():
        logits = model(image_a, image_b)
    assert logits.shape == (2, 1, *size)


def test_identical_inputs_give_spatially_constant_output_with_diff_fusion() -> None:
    """|f - f| = 0 at every level, so nothing in the output can depend on the image content."""
    model = SiameseUNet(**SMALL, fusion="diff").eval()
    image = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        logits = model(image, image)
    assert logits.std().item() < 1e-5


def test_diff_fusion_is_symmetric_in_eval_mode() -> None:
    model = SiameseUNet(**SMALL, fusion="diff").eval()
    image_a, image_b = torch.randn(1, 3, 64, 64), torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        forward = model(image_a, image_b)
        backward = model(image_b, image_a)
    assert torch.allclose(forward, backward, atol=1e-5)


def test_encoder_is_shared_between_dates() -> None:
    model = SiameseUNet(**SMALL)
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    total_params = sum(p.numel() for p in model.parameters())
    assert 0 < encoder_params < total_params  # one encoder, not two


def test_invalid_fusion_raises() -> None:
    with pytest.raises(ValueError, match="fusion"):
        SiameseUNet(**SMALL, fusion="sum")  # type: ignore[arg-type]
