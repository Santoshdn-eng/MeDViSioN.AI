import pytest
import numpy as np
from ai.mri.mri_model import monai_brats_pipeline
from backend.app.services.multi_model_router import MultiModelRouterService

def test_monai_brats_pipeline_initialization():
    assert monai_brats_pipeline is not None
    assert monai_brats_pipeline.model_loaded is True
    assert monai_brats_pipeline.device is not None

def test_monai_brats_inference_execution():
    # 512x512 synthetic brain slice
    dummy_slice = np.zeros((512, 512), dtype=np.uint8)
    y, x = np.ogrid[:512, :512]
    cranium = ((x - 256)**2 / 160**2 + (y - 256)**2 / 180**2) < 1.0
    dummy_slice[cranium] = 140
    lesion = ((x - 190)**2 / 28**2 + (y - 200)**2 / 28**2) < 1.0
    dummy_slice[lesion] = 230

    result = monai_brats_pipeline.predict(dummy_slice)

    assert result["model_name"] == "MONAI SegResNet (BraTS Pretrained)"
    assert "brats" in result["model_version"].lower()
    assert "GPU" in result["device_used"] or "CPU" in result["device_used"]
    assert result["confidence_score"] > 0.0
    assert len(result["findings"]) >= 3
    assert len(result["segmentations"]) == 3  # WT, TC, ET

def test_multi_model_router_mri_integration():
    dummy_slice = np.full((512, 512), 128, dtype=np.uint8)
    res = MultiModelRouterService.run_inference(dummy_slice, "MRI")
    assert res["modality"] == "MRI"
    assert "MONAI" in res["model_name"]
    assert len(res["segmentations"]) == 3
