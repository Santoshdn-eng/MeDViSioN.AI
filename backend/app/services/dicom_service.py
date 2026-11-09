import os
import io
import time
import hashlib
import numpy as np
from PIL import Image as PILImage, ImageDraw, ImageFilter
from typing import Dict, Any, Tuple, Optional

class DICOMService:
    @staticmethod
    def deidentify_metadata(raw_metadata: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
        """De-identifies patient information according to Safe Harbor rules."""
        raw_id = str(raw_metadata.get("PatientID", "UNKNOWN_PATIENT"))
        anon_hash = hashlib.sha256(raw_id.encode('utf-8')).hexdigest()[:10].upper()
        anon_id = f"ANON_{anon_hash}"
        
        study_uid = raw_metadata.get("StudyInstanceUID")
        if not study_uid:
            study_uid = f"1.2.840.10008.{anon_hash}.{int(time.time()*1000)}"

        safe_metadata = {
            "PatientAnonID": anon_id,
            "PatientAge": raw_metadata.get("PatientAge", "N/A"),
            "PatientSex": raw_metadata.get("PatientSex", "O"),
            "Modality": raw_metadata.get("Modality", "UNKNOWN"),
            "BodyPartExamined": raw_metadata.get("BodyPartExamined", "UNKNOWN"),
            "StudyDescription": raw_metadata.get("StudyDescription", "Research Study"),
            "Manufacturer": raw_metadata.get("Manufacturer", "Generic Medical Imaging"),
            "StudyInstanceUID": study_uid,
            "WindowCenter": raw_metadata.get("WindowCenter", 40),
            "WindowWidth": raw_metadata.get("WindowWidth", 400),
            "IsDeidentified": True
        }
        return safe_metadata, anon_id

    @staticmethod
    def apply_windowing(pixel_array: np.ndarray, window_center: float, window_width: float) -> np.ndarray:
        """Applies window/level transformation for medical imaging visualization."""
        img_min = window_center - window_width // 2
        img_max = window_center + window_width // 2
        windowed = np.clip(pixel_array, img_min, img_max)
        windowed = (windowed - img_min) / (img_max - img_min) * 255.0
        return windowed.astype(np.uint8)

    @staticmethod
    def generate_synthetic_study(modality: str, body_part: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Generates anatomically plausible synthetic medical imaging studies for Research/Demo mode."""
        size = (512, 512)
        modality = modality.upper()
        
        if modality == "X-RAY":
            # Synthetic Chest X-Ray with rib cages and lung fields
            img = np.zeros(size, dtype=np.float32) + 30
            y, x = np.ogrid[:size[0], :size[1]]
            # Left & Right lung cavity shadows
            left_lung = ((x - 170)**2 / 60**2 + (y - 250)**2 / 120**2) < 1.0
            right_lung = ((x - 340)**2 / 60**2 + (y - 250)**2 / 120**2) < 1.0
            img[left_lung] = 160
            img[right_lung] = 175
            # Heart silhouette
            heart = ((x - 220)**2 / 55**2 + (y - 290)**2 / 65**2) < 1.0
            img[heart] = 70
            # Ribs
            for rib_y in range(160, 420, 35):
                rib = np.abs(y - rib_y - 0.05*(x-256)**2/10) < 6
                img[rib] = np.clip(img[rib] - 30, 10, 255)
            # Add anatomical noise
            img += np.random.normal(0, 5, size)
            pixel_array = np.clip(img, 0, 255).astype(np.uint8)
            metadata = {
                "Modality": "X-RAY",
                "BodyPartExamined": "CHEST",
                "StudyDescription": "Chest PA Research Scan",
                "PatientAge": 54,
                "PatientSex": "M",
                "WindowCenter": 128,
                "WindowWidth": 256
            }
            return pixel_array, metadata
            
        elif modality == "MRI":
            # Synthetic Brain MRI with ventricles and gray/white matter
            img = np.zeros(size, dtype=np.float32)
            y, x = np.ogrid[:size[0], :size[1]]
            # Cranium & Brain parenchyma
            cranium = ((x - 256)**2 / 160**2 + (y - 256)**2 / 180**2) < 1.0
            img[cranium] = 140
            # Ventricles
            ventricles = (((x - 235)**2 / 15**2 + (y - 240)**2 / 40**2) < 1.0) | \
                         (((x - 277)**2 / 15**2 + (y - 240)**2 / 40**2) < 1.0)
            img[ventricles] = 20
            # Synthetic lesion / mass in left frontal region
            lesion = ((x - 190)**2 / 28**2 + (y - 200)**2 / 28**2) < 1.0
            img[lesion] = 230 # Hyperintense
            img += np.random.normal(0, 4, size)
            pixel_array = np.clip(img, 0, 255).astype(np.uint8)
            metadata = {
                "Modality": "MRI",
                "BodyPartExamined": "BRAIN",
                "StudyDescription": "Brain MRI T2/FLAIR Research Protocol",
                "PatientAge": 48,
                "PatientSex": "F",
                "WindowCenter": 120,
                "WindowWidth": 200
            }
            return pixel_array, metadata

        elif modality in ("SPINE_MRI", "SPINE"):
            # Synthetic Lumbar Spine MRI (T2-FSE Sagittal)
            # Shows L1-S1 vertebral bodies, intervertebral discs, thecal sac, and L4-L5 focal extrusion
            img = np.zeros(size, dtype=np.float32)
            y, x = np.ogrid[:size[0], :size[1]]

            # Background soft tissue / paraspinal muscles
            paraspinal = ((x - 256)**2 / 90**2 + (y - 270)**2 / 200**2) < 1.0
            img[paraspinal] = 55

            # Vertebral bodies (L1-L5): high signal cortical shell, low-signal marrow on T2
            vertebra_centers = [125, 180, 235, 290, 345]  # y positions L1→L5
            for vy in vertebra_centers:
                body = ((x - 265)**2 / 38**2 + (y - vy)**2 / 20**2) < 1.0
                img[body] = 80  # Fatty marrow – moderate T2 signal
                cortex = ((x - 265)**2 / 38**2 + (y - vy)**2 / 20**2 > 0.80) & body
                img[cortex] = 30  # Dark cortical bone

            # Intervertebral discs (between vertebrae): bright T2 on healthy, dark on degenerated
            disc_centers = [152, 207, 262, 317, 374]  # L1-L2 through L5-S1
            disc_signals = [220, 215, 180, 110, 90]    # L4-L5 and L5-S1 desiccated (Pfirrmann IV-V)
            for dy, dsig in zip(disc_centers, disc_signals):
                disc = ((x - 265)**2 / 32**2 + (y - dy)**2 / 9**2) < 1.0
                img[disc] = dsig

            # Thecal sac / CSF (bright T2 fluid signal running through central canal)
            thecal_canal = ((x - 305)**2 / 11**2 + (y - 250)**2 / 155**2) < 1.0
            img[thecal_canal] = 240  # Bright CSF

            # Spinal cord within thecal sac (L1-L2 conus level only)
            cord = ((x - 305)**2 / 5**2 + (y - 145)**2 / 25**2) < 1.0
            img[cord] = 110  # Moderate T2 cord signal

            # L4-L5 Disc Extrusion: left paracentral posterior focal mass compressing thecal sac
            extrusion = ((x - 290)**2 / 16**2 + (y - 317)**2 / 12**2) < 1.0
            img[extrusion] = 95  # Dark (desiccated disc material) compressing bright CSF

            # Posterior elements – spinous processes (skeletal dark signal)
            for vy in [125, 180, 235, 290, 345]:
                spinous = ((x - 230)**2 / 8**2 + (y - vy)**2 / 10**2) < 1.0
                img[spinous] = 25

            # Add anatomical MRI noise (T2-FSE acquisition)
            img += np.random.normal(0, 5, size)
            pixel_array = np.clip(img, 0, 255).astype(np.uint8)
            metadata = {
                "Modality": "SPINE_MRI",
                "BodyPartExamined": "LUMBAR SPINE",
                "StudyDescription": "Lumbar Spine MRI T2-FSE Sagittal Protocol",
                "PatientAge": 46,
                "PatientSex": "M",
                "SliceThickness": 3.0,
                "PixelSpacing": 0.390,
                "FieldOfView": 260,
                "RepetitionTime": 3800,
                "EchoTime": 102,
                "WindowCenter": 300,
                "WindowWidth": 600
            }
            return pixel_array, metadata

        else: # CT
            # Synthetic Lung CT with high contrast HU representation
            img = np.zeros(size, dtype=np.float32) + 50 # Soft tissue
            y, x = np.ogrid[:size[0], :size[1]]
            body = ((x - 256)**2 / 190**2 + (y - 256)**2 / 170**2) < 1.0
            img[~body] = 0 # Air outside
            # Lung fields (low density)
            lungs = (((x - 170)**2 / 65**2 + (y - 250)**2 / 95**2) < 1.0) | \
                    (((x - 340)**2 / 65**2 + (y - 250)**2 / 95**2) < 1.0)
            img[lungs & body] = 20
            # Nodule in right upper lung
            nodule = ((x - 325)**2 / 12**2 + (y - 210)**2 / 12**2) < 1.0
            img[nodule] = 180
            img += np.random.normal(0, 3, size)
            pixel_array = np.clip(img, 0, 255).astype(np.uint8)
            metadata = {
                "Modality": "CT",
                "BodyPartExamined": "CHEST/LUNG",
                "StudyDescription": "Chest High-Resolution CT Nodule Protocol",
                "PatientAge": 62,
                "PatientSex": "M",
                "WindowCenter": -600,
                "WindowWidth": 1500
            }
            return pixel_array, metadata
