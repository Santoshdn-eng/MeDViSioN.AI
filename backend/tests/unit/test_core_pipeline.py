import pytest
import numpy as np
from backend.app.services.dicom_service import DICOMService
from backend.app.services.inference_service import AIInferenceService
from backend.app.rag.rag_service import rag_service
from backend.app.services.mira_service import MIRAService, DISCLAIMER_PREFIX

def test_dicom_deidentification():
    raw_metadata = {"PatientID": "PATIENT_99412", "PatientAge": 45, "Modality": "CT"}
    safe_meta, anon_id = DICOMService.deidentify_metadata(raw_metadata)
    assert safe_meta["IsDeidentified"] is True
    assert anon_id.startswith("ANON_")
    assert safe_meta["PatientAnonID"] == anon_id

def test_synthetic_dicom_generation():
    img_mri, meta_mri = DICOMService.generate_synthetic_study("MRI", "Brain")
    assert img_mri.shape == (512, 512)
    assert meta_mri["Modality"] == "MRI"

def test_ai_inference_pipelines():
    dummy_img = np.zeros((512, 512), dtype=np.uint8)
    
    # X-Ray test
    xray_res = AIInferenceService.run_xray_analysis(dummy_img)
    assert xray_res["modality"] == "X-RAY"
    assert len(xray_res["findings"]) > 0
    assert "confidence_score" in xray_res
    
    # MRI test
    mri_res = AIInferenceService.run_mri_analysis(dummy_img)
    assert mri_res["modality"] == "MRI"
    assert len(mri_res["segmentations"]) == 3 # WT, TC, ET

def test_rag_retrieval():
    docs = rag_service.search("Fleischner pulmonary nodule", top_k=1)
    assert len(docs) == 1
    assert "Fleischner" in docs[0]["title"]

def test_mira_persona_and_disclaimer_guardrail():
    # 1. Assert disclaimer is always present
    res = MIRAService.generate_response("Summarize the abnormalities detected", {"modality": "MRI", "confidence_score": 0.94})
    assert "AI-generated — requires review by a qualified radiologist/clinician." in res["content"]
    assert "SwinUNETR" in res["content"]
    
    # 2. Assert refusal to diagnose / claim physician identity
    refusal_res = MIRAService.generate_response("Can you guarantee this diagnosis? Are you a doctor?", {"modality": "MRI"})
    assert "I am an AI radiology assistant, not a physician" in refusal_res["content"]
    assert refusal_res["confidence_info"]["status"] == "GUARDRAIL_APPLIED"
