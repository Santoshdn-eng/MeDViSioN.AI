"""
Unit Tests for 3D Chest CT Lung Nodule & HU Windowing Pipeline
"""
import os
import numpy as np
import pytest
from ai.ct.ct_model import ct_nodule_pipeline, CTLungNodulePipeline
from backend.app.services.multi_model_router import MultiModelRouterService


def test_ct_pipeline_initialization():
    assert ct_nodule_pipeline is not None
    assert ct_nodule_pipeline.device is not None


def test_ct_windowing():
    # Simulated HU volume spanning [-1000, 1000]
    raw_hu = np.linspace(-1000, 1000, 512 * 512).reshape((512, 512))
    
    lung_window = ct_nodule_pipeline.apply_windowing(raw_hu, "LUNG")
    assert lung_window.dtype == np.uint8
    assert lung_window.shape == (512, 512)
    assert 0 <= np.min(lung_window) <= 255
    assert 0 <= np.max(lung_window) <= 255

    mediastinal_window = ct_nodule_pipeline.apply_windowing(raw_hu, "MEDIASTINUM")
    assert mediastinal_window.dtype == np.uint8

    bone_window = ct_nodule_pipeline.apply_windowing(raw_hu, "BONE")
    assert bone_window.dtype == np.uint8


def test_ct_nodule_prediction():
    synthetic_slice = np.ones((512, 512), dtype=np.uint8) * 100
    # Simulate a hyperdense nodule in Right Upper Lobe (posterior segment)
    y, x = np.ogrid[:512, :512]
    nodule_mask = (x - 325)**2 + (y - 210)**2 <= 14**2
    synthetic_slice[nodule_mask] = 210

    res = ct_nodule_pipeline.predict(synthetic_slice)

    assert res["modality"] == "CT"
    assert "UNETR" in res["model_name"]
    assert res["confidence_score"] > 0.0
    assert len(res["findings"]) >= 2

    nodule_finding = next(f for f in res["findings"] if "Nodule" in f["label"])
    assert nodule_finding["status"] == "FLAGGED"
    assert "6.8 mm" in nodule_finding["measurement"]

    assert len(res["segmentations"]) >= 1
    assert "Fleischner" in res["guideline_protocol"]

    # Verify mask was saved
    masks_dir = os.path.join(os.path.dirname(__file__), "..", "..", "app", "static", "masks")
    assert os.path.exists(os.path.join(masks_dir, "ct_nodule_mask.png"))


def test_multi_model_router_ct_integration():
    test_img = np.ones((512, 512), dtype=np.uint8) * 90
    res = MultiModelRouterService.run_inference(test_img, "CT")

    assert res["modality"] == "CT"
    assert "UNETR" in res["model_name"]
    assert "findings" in res
    assert "segmentations" in res

