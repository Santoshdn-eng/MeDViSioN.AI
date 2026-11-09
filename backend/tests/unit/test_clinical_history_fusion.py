"""
Unit Tests for Patient Clinical History & Requisition Input (EHR Fusion)
"""
import pytest
from backend.app.services.mira_service import MIRAService


def test_mira_differential_smoker_elevated_malignancy_risk():
    study_context = {
        "modality": "CT",
        "findings": [
            {
                "label": "Solid Pulmonary Nodule",
                "probability": 0.91,
                "measurement": "6.8 mm diameter (Vol: 0.16 cm³)",
                "status": "FLAGGED"
            }
        ],
        "clinical_history": {
            "age": 64,
            "gender": "M",
            "smoking_status": "ACTIVE_SMOKER",
            "pack_years": 35,
            "chief_complaint": "Hemoptysis and chronic cough"
        }
    }

    res = MIRAService.generate_response(
        "What is the differential diagnosis and risk profile for this patient?",
        study_context
    )

    content = res["content"]
    assert "Bronchogenic Carcinoma" in content
    assert "high-risk" in content.lower()
    assert "Fleischner" in content


def test_mira_differential_non_smoker_benign_granuloma():
    study_context = {
        "modality": "CT",
        "findings": [
            {
                "label": "Solid Pulmonary Nodule",
                "probability": 0.91,
                "measurement": "6.8 mm diameter",
                "status": "FLAGGED"
            }
        ],
        "clinical_history": {
            "age": 32,
            "gender": "F",
            "smoking_status": "NEVER",
            "pack_years": 0,
            "chief_complaint": "Asymptomatic health screen"
        }
    }

    res = MIRAService.generate_response(
        "Provide differential diagnosis based on patient clinical history",
        study_context
    )

    content = res["content"]
    assert "Granuloma" in content or "Post-Inflammatory" in content
    assert "low-risk" in content.lower()


def test_mira_differential_cardiac_failure_orthopnea():
    study_context = {
        "modality": "X-RAY",
        "findings": [
            {
                "label": "Pleural Effusion",
                "probability": 0.84,
                "measurement": "Moderate meniscus blunting",
                "status": "FLAGGED"
            },
            {
                "label": "Cardiomegaly",
                "probability": 0.52,
                "measurement": "CTR ~ 0.52",
                "status": "FLAGGED"
            }
        ],
        "clinical_history": {
            "age": 72,
            "gender": "F",
            "chief_complaint": "Severe orthopnea and bilateral pedal edema"
        }
    }

    res = MIRAService.generate_response(
        "What are the top differentials for this chest X-ray?",
        study_context
    )

    content = res["content"]
    assert "Congestive Heart Failure" in content or "Heart" in content
    assert "transudative" in content.lower() or "edema" in content.lower()


def test_mira_differential_prior_systemic_cancer():
    study_context = {
        "modality": "MRI",
        "findings": [
            {
                "label": "Frontal Lobe Mass",
                "probability": 0.94,
                "measurement": "34.2 x 28.5 mm",
                "status": "FLAGGED"
            }
        ],
        "clinical_history": {
            "age": 58,
            "gender": "M",
            "prior_malignancy": "Stage III Lung Adenocarcinoma",
            "chief_complaint": "New onset focal seizure"
        }
    }

    res = MIRAService.generate_response(
        "What is the differential diagnosis?",
        study_context
    )

    content = res["content"]
    assert "Metastasis" in content
    assert "Glioma" in content

