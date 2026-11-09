"""
Unit Tests for Chest X-Ray 14-Pathology Classifier & Grad-CAM Pipeline
"""
import os
import numpy as np
import pytest
from ai.xray.xray_model import chest_xray_pipeline, ChestXRayPipeline
from backend.app.services.multi_model_router import MultiModelRouterService


def test_xray_pipeline_initialization():
    assert chest_xray_pipeline is not None
    assert chest_xray_pipeline.model is not None
    assert len(chest_xray_pipeline.pathology_labels) == 14
    assert chest_xray_pipeline.device is not None


def test_xray_pipeline_predict():
    # Create synthetic radiograph with thoracic contrast
    y, x = np.ogrid[:512, :512]
    radiograph = np.clip(120 + 70 * np.sin(x / 50.0) + 40 * np.cos(y / 60.0), 10, 250).astype(np.uint8)
    
    result = chest_xray_pipeline.predict(radiograph)
    
    assert result["modality"] == "X-RAY"
    assert "DenseNet-121" in result["model_name"]
    assert result["confidence_score"] > 0.0
    assert len(result["findings"]) >= 4
    
    labels = [f["label"] for f in result["findings"]]
    assert "Pleural Effusion" in labels
    assert "Cardiomegaly" in labels
    assert "Pneumothorax" in labels
    
    # Check physical measurements
    effusion_f = next(f for f in result["findings"] if f["label"] == "Pleural Effusion")
    assert "mL" in effusion_f["measurement"]
    
    cardio_f = next(f for f in result["findings"] if f["label"] == "Cardiomegaly")
    assert "CTR" in cardio_f["measurement"]
    
    # Check Grad-CAM and segmentation masks were generated
    masks_dir = os.path.join(os.path.dirname(__file__), "..", "..", "app", "static", "masks")
    assert os.path.exists(os.path.join(masks_dir, "xray_gradcam.png"))
    assert os.path.exists(os.path.join(masks_dir, "xray_effusion_mask.png"))


def test_ctr_estimation_bounds():
    test_img = np.zeros((512, 512), dtype=np.uint8)
    test_img[200:400, 100:400] = 150 # Simulated thoracic cage
    test_img[250:350, 180:320] = 220 # Simulated cardiac opacity
    
    ctr = chest_xray_pipeline._estimate_ctr(test_img)
    assert 0.40 <= ctr <= 0.65


def test_multi_model_router_xray_integration():
    test_img = np.ones((512, 512), dtype=np.uint8) * 128
    res = MultiModelRouterService.run_inference(test_img, "X-RAY")
    
    assert res["modality"] == "X-RAY"
    assert "DenseNet-121" in res["model_name"]
    assert "findings" in res
    assert "segmentations" in res

