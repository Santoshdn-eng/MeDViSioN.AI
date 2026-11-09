"""
Unit Tests for Clinical Audit, OOD Rejection & Technical Adequacy Engine
"""
import numpy as np
import pytest
from backend.app.services.clinical_audit_service import ClinicalAuditService


def test_ood_flat_image_rejection():
    # Completely flat/uniform non-medical image (e.g. blank canvas or solid color)
    flat_pixels = np.zeros((512, 512), dtype=np.uint8) + 40
    report = ClinicalAuditService.validate_image_quality(flat_pixels, "X-RAY")
    
    assert report["is_valid"] is False
    assert report["quality_grade"] == "UNREADABLE"
    assert "flat, corrupted, or non-medical" in report["rejection_reason"]
    assert report["exposure"] == "DEFECTIVE"


def test_ood_low_variance_noise_rejection():
    # Extremely low variance noise (e.g., solid gray with tiny noise)
    low_var_pixels = (np.random.normal(128, 1.0, (512, 512))).astype(np.uint8)
    report = ClinicalAuditService.validate_image_quality(low_var_pixels, "MRI")
    
    assert report["is_valid"] is False
    assert report["quality_grade"] == "UNREADABLE"


def test_xray_technical_adequacy_valid():
    # Create realistic thoracic radiographic gradient
    y, x = np.ogrid[:512, :512]
    radiograph = np.clip(120 + 70 * np.sin(x / 50.0) + 40 * np.cos(y / 60.0), 10, 250).astype(np.uint8)
    
    report = ClinicalAuditService.validate_image_quality(radiograph, "X-RAY")
    
    assert report["is_valid"] is True
    assert "DIAGNOSTIC GRADE" in report["quality_grade"]
    assert "PA" in report["projection"] or "AP" in report["projection"]
    assert report["snr_db"] > 5.0
    assert "satisfactory" in report["technical_summary"].lower()


def test_mri_technical_adequacy_valid():
    # Create realistic brain-like contrast array
    y, x = np.ogrid[:512, :512]
    dist_from_center = np.sqrt((x - 256)**2 + (y - 256)**2)
    mri_slice = np.where(dist_from_center < 180, 140 + 50 * np.sin(dist_from_center / 20.0), 20).astype(np.uint8)
    
    report = ClinicalAuditService.validate_image_quality(mri_slice, "MRI")
    
    assert report["is_valid"] is True
    assert report["quality_grade"] == "DIAGNOSTIC GRADE A"
    assert "Axial" in report["projection"]
    assert "isotropic" in report["technical_summary"].lower()


def test_stat_critical_alert_tension_pneumothorax():
    findings = [
        {
            "label": "Tension Pneumothorax",
            "probability": 0.88,
            "location": "Right hemithorax",
            "measurement": "Complete lung collapse with contralateral tracheal shift",
            "status": "FLAGGED"
        },
        {
            "label": "Rib Fracture",
            "probability": 0.72,
            "location": "Right 5th rib",
            "measurement": "Displaced fracture",
            "status": "FLAGGED"
        }
    ]
    alert = ClinicalAuditService.check_critical_alerts(findings, "X-RAY")
    
    assert alert["is_stat"] is True
    assert alert["alert_level"] == "CRITICAL_IMMEDIATE"
    assert "TENSION PNEUMOTHORAX" in alert["emergency_condition"]
    assert "needle decompression" in alert["action_required"].lower()


def test_systematic_checklist_abcde_xray():
    findings = [
        {
            "label": "Pleural Effusion",
            "probability": 0.82,
            "location": "Right costophrenic angle",
            "measurement": "Moderate meniscus blunting (~200 mL)",
            "status": "FLAGGED"
        }
    ]
    adequacy = {
        "is_valid": True,
        "quality_grade": "DIAGNOSTIC GRADE A",
        "projection": "PA",
        "inspiration": "Adequate"
    }
    
    checklist = ClinicalAuditService.generate_systematic_checklist("X-RAY", findings, adequacy)
    
    assert len(checklist) == 5
    categories = [item["category"] for item in checklist]
    assert any("A - Airway" in cat for cat in categories)
    assert any("B - Breathing" in cat for cat in categories)
    assert any("C - Circulation" in cat for cat in categories)
    assert any("D - Diaphragm" in cat for cat in categories)
    assert any("E - Extrathoracic" in cat for cat in categories)
    
    # Check that breathing & diaphragm flagged the effusion
    breathing_item = next(item for item in checklist if "B - Breathing" in item["category"])
    assert breathing_item["status"] == "FLAGGED"
    assert "Pleural Effusion" in breathing_item["finding_text"]


def test_standardized_rads_stratification():
    # 1. CT Lung-RADS test (6.8 mm nodule)
    ct_findings = [
        {
            "label": "Solid Pulmonary Nodule",
            "probability": 0.91,
            "measurement": "6.8 mm diameter (Vol: 0.16 cm³)",
            "status": "FLAGGED"
        }
    ]
    ct_rads = ClinicalAuditService.calculate_standardized_rads("CT", ct_findings, [])
    assert ct_rads["system"] == "ACR Lung-RADS v2022"
    assert ct_rads["category"] == "Lung-RADS 3"
    assert "6 months" in ct_rads["recommendation"]
    
    # 2. MRI WHO CNS5 test (with enhancing tumor)
    mri_segs = [
        {
            "structure_name": "Enhancing Active Tumor (ET)",
            "volume_cm3": 3.6
        }
    ]
    mri_rads = ClinicalAuditService.calculate_standardized_rads("MRI", [], mri_segs)
    assert mri_rads["system"] == "WHO CNS5 Classification"
    assert "High-Grade Glioma" in mri_rads["category"]
    assert "Malignant" in mri_rads["malignancy_risk"]

