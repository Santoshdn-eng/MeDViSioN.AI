import pytest
from httpx import AsyncClient, ASGITransport
import io
import numpy as np
from PIL import Image as PILImage
from backend.app.main import app

@pytest.mark.anyio
async def test_study_details_endpoint():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # First ensure a demo study exists
        demo_res = await client.post("/api/v1/studies/demo?modality=MRI&body_part=Brain")
        assert demo_res.status_code == 200
        study_id = demo_res.json()["id"]

        # Request full details
        res = await client.get(f"/api/v1/studies/{study_id}/details")
        assert res.status_code == 200
        data = res.json()
        assert data["study_id"] == study_id
        assert data["modality"] == "MRI"
        assert "technical_params" in data
        assert "raw_dicom_tags" in data
        assert len(data["raw_dicom_tags"]) > 10
        assert "image_metrics" in data
        assert "findings" in data
        assert "rads_assessment" in data

@pytest.mark.anyio
async def test_dicom_sr_export_endpoint():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Ensure a demo study exists
        demo_res = await client.post("/api/v1/studies/demo?modality=CT&body_part=Chest")
        assert demo_res.status_code == 200
        study_id = demo_res.json()["id"]

        # Export DICOM-SR
        res = await client.get(f"/api/v1/studies/{study_id}/export-sr")
        assert res.status_code == 200
        data = res.json()
        assert data["header"]["SOPClassUID"] == "1.2.840.10008.5.1.4.1.1.88.33"
        assert data["header"]["SOPClassName"] == "Comprehensive 3D SR Storage (TID 1500)"
        assert "qualitative_evaluations" in data
        assert "regulatory_disclaimer" in data

@pytest.mark.anyio
async def test_imaging_upload_returns_details_and_tags():
    # Create valid synthetic image using DICOMService
    from backend.app.services.dicom_service import DICOMService
    img, _ = DICOMService.generate_synthetic_study("MRI", "Brain")
    pil_img = PILImage.fromarray(img)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        files = {"file": ("test_brain_mri.png", buf.getvalue(), "image/png")}
        res = await client.post("/api/v1/imaging/upload", files=files)
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "SUCCESS"
        assert "technical_params" in data
        assert "raw_dicom_tags" in data
        assert len(data["raw_dicom_tags"]) > 10
        assert "image_metrics" in data
        assert "dicom_sr" in data
        assert data["dicom_sr"]["sop_class_name"] == "Comprehensive 3D SR Storage (TID 1500)"
