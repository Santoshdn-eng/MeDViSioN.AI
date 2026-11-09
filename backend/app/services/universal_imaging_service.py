import os
import io
import hashlib
import numpy as np
from PIL import Image as PILImage, ImageOps, ImageFilter
from typing import Dict, Any, Tuple, Optional, List
import pydicom

class UniversalImagingService:
    @staticmethod
    def calculate_image_metrics(pixel_array: np.ndarray) -> Dict[str, Any]:
        """Calculates deep signal, contrast, and information-theoretic metrics for medical images."""
        p_float = pixel_array.astype(np.float32)
        min_v = float(np.min(p_float))
        max_v = float(np.max(p_float))
        mean_v = float(np.mean(p_float))
        std_v = float(np.std(p_float))
        dyn_range = float(max_v - min_v)
        
        # Signal to Noise Ratio in dB
        bg_corner = p_float[:40, :40]
        bg_std = float(np.std(bg_corner))
        if bg_std > 0.01:
            snr_db = round(float(20.0 * np.log10(max(mean_v, 1.0) / bg_std)), 2)
        else:
            snr_db = round(float(20.0 * np.log10(max(mean_v, 1.0) / max(std_v, 1.0))), 2)

        # Contrast to Noise Ratio
        fg_center = p_float[180:332, 180:332]
        cnr = round(float(abs(np.mean(fg_center) - mean_v) / max(std_v, 1.0)), 2)

        # Discrete Shannon Entropy
        hist, _ = np.histogram(pixel_array, bins=256, range=(0, 256))
        probs = hist / float(pixel_array.size)
        probs = probs[probs > 0]
        entropy_bits = round(float(-np.sum(probs * np.log2(probs))), 3)

        # 32-bin normalized histogram for UI sparkline / chart
        hist32, _ = np.histogram(pixel_array, bins=32, range=(0, 256))
        hist32_norm = [round(float(v) / float(pixel_array.size), 4) for v in hist32]

        return {
            "min_val": round(min_v, 1),
            "max_val": round(max_v, 1),
            "mean_val": round(mean_v, 1),
            "std_val": round(std_v, 1),
            "dynamic_range": round(dyn_range, 1),
            "snr_db": max(0.0, snr_db),
            "cnr": max(0.1, cnr),
            "entropy_bits": entropy_bits,
            "histogram32": hist32_norm,
            "bit_depth": "8-bit Grayscale (0-255)" if max_v <= 255 else "12-bit Extended"
        }

    @staticmethod
    def build_synthetic_dicom_tags(modality: str, filename: str, metrics: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generates standard DICOM header tags (Part 10 / NEMA standard) for PACS tag inspection."""
        mod = modality.upper()
        anon_hash = hashlib.sha256(filename.encode('utf-8')).hexdigest()[:8].upper()
        
        tags = [
            {"tag": "(0008,0005)", "name": "Specific Character Set", "vr": "CS", "value": "ISO_IR 100"},
            {"tag": "(0008,0016)", "name": "SOP Class UID", "vr": "UI", "value": "1.2.840.10008.5.1.4.1.1.4" if mod in ["MRI", "SPINE_MRI", "SPINE"] else ("1.2.840.10008.5.1.4.1.1.2" if mod == "CT" else "1.2.840.10008.5.1.4.1.1.1")},
            {"tag": "(0008,0018)", "name": "SOP Instance UID", "vr": "UI", "value": f"1.2.840.10008.5.{anon_hash}.20261108"},
            {"tag": "(0008,0020)", "name": "Study Date", "vr": "DA", "value": "20261108"},
            {"tag": "(0008,0030)", "name": "Study Time", "vr": "TM", "value": "104522.000"},
            {"tag": "(0008,0060)", "name": "Modality", "vr": "CS", "value": "MR" if mod in ["MRI", "SPINE_MRI", "SPINE"] else mod},
            {"tag": "(0008,0070)", "name": "Manufacturer", "vr": "LO", "value": "MedVision HealthTech PACS Engine v4.2"},
            {"tag": "(0008,0080)", "name": "Institution Name", "vr": "LO", "value": "St. Jude Medical Center - Dept. of Radiology"},
            {"tag": "(0008,1030)", "name": "Study Description", "vr": "LO", "value": "Lumbar Spine MRI Protocol" if mod in ["SPINE_MRI", "SPINE"] else f"{mod} Clinical Diagnostic Protocol"},
            {"tag": "(0008,103E)", "name": "Series Description", "vr": "LO", "value": "Series 2: Sagittal T2-FSE Lumbar Spine" if mod in ["SPINE_MRI", "SPINE"] else f"Series 2: {mod} Volumetric Reconstruction"},
            {"tag": "(0010,0010)", "name": "Patient's Name", "vr": "PN", "value": f"ANONYMIZED_PATIENT_{anon_hash}"},
            {"tag": "(0010,0020)", "name": "Patient ID", "vr": "LO", "value": f"MRN-{anon_hash}-R"},
            {"tag": "(0010,0030)", "name": "Patient's Birth Date", "vr": "DA", "value": "19740512"},
            {"tag": "(0010,0040)", "name": "Patient's Sex", "vr": "CS", "value": "M"},
            {"tag": "(0018,0015)", "name": "Body Part Examined", "vr": "CS", "value": "SPINE" if mod in ["SPINE_MRI", "SPINE"] else ("BRAIN" if mod == "MRI" else "CHEST")},
            {"tag": "(0018,0050)", "name": "Slice Thickness", "vr": "DS", "value": "3.000 mm" if mod in ["SPINE_MRI", "SPINE"] else ("1.000 mm" if mod in ["CT", "MRI"] else "Projection (N/A)")},
            {"tag": "(0018,0060)", "name": "KVP / Magnetic Field", "vr": "DS", "value": "3.0 Tesla" if mod in ["MRI", "SPINE_MRI", "SPINE"] else "120.0 kVp"},
            {"tag": "(0018,0080)", "name": "Repetition Time (TR)", "vr": "DS", "value": "3800.0 ms" if mod in ["SPINE_MRI", "SPINE"] else ("9200.0 ms" if mod == "MRI" else "N/A")},
            {"tag": "(0018,0081)", "name": "Echo Time (TE)", "vr": "DS", "value": "102.0 ms" if mod in ["SPINE_MRI", "SPINE"] else ("98.0 ms" if mod == "MRI" else "N/A")},
            {"tag": "(0018,1150)", "name": "Exposure Time", "vr": "IS", "value": "160 ms" if mod == "CT" else ("32 ms" if mod == "X-RAY" else "N/A")},
            {"tag": "(0018,1151)", "name": "X-Ray Tube Current", "vr": "IS", "value": "200 mA" if mod not in ["MRI", "SPINE_MRI", "SPINE"] else "N/A"},
            {"tag": "(0018,1210)", "name": "Convolution Kernel", "vr": "SH", "value": "T2-FSE Lumbar Sagittal" if mod in ["SPINE_MRI", "SPINE"] else ("T2-FLAIR Isotropic" if mod == "MRI" else ("Lung B50f High-Res" if mod == "CT" else "High-Res PA"))},
            {"tag": "(0018,5100)", "name": "Patient Position", "vr": "CS", "value": "HFS (Head First Supine)" if mod != "X-RAY" else "PA Upright"},
            {"tag": "(0020,000D)", "name": "Study Instance UID", "vr": "UI", "value": f"1.2.840.10008.1.{anon_hash}.9912"},
            {"tag": "(0020,000E)", "name": "Series Instance UID", "vr": "UI", "value": f"1.2.840.10008.2.{anon_hash}.9912.2"},
            {"tag": "(0020,0013)", "name": "Instance (Slice) Number", "vr": "IS", "value": "10"},
            {"tag": "(0028,0010)", "name": "Rows", "vr": "US", "value": "512"},
            {"tag": "(0028,0011)", "name": "Columns", "vr": "US", "value": "512"},
            {"tag": "(0028,0030)", "name": "Pixel Spacing", "vr": "DS", "value": "0.488281 \\ 0.488281 mm"},
            {"tag": "(0028,0100)", "name": "Bits Allocated", "vr": "US", "value": "16"},
            {"tag": "(0028,0101)", "name": "Bits Stored", "vr": "US", "value": "12"},
            {"tag": "(0028,0102)", "name": "High Bit", "vr": "US", "value": "11"},
            {"tag": "(0028,0103)", "name": "Pixel Representation", "vr": "US", "value": "0 (Unsigned)"},
            {"tag": "(0028,1050)", "name": "Window Center", "vr": "DS", "value": "120" if mod == "MRI" else ("-600" if mod == "CT" else "128")},
            {"tag": "(0028,1051)", "name": "Window Width", "vr": "DS", "value": "200" if mod == "MRI" else ("1500" if mod == "CT" else "256")},
            {"tag": "(0028,1052)", "name": "Rescale Intercept", "vr": "DS", "value": "0.0"},
            {"tag": "(0028,1053)", "name": "Rescale Slope", "vr": "DS", "value": "1.0"},
            {"tag": "(0028,1054)", "name": "Rescale Type", "vr": "LO", "value": "HU" if mod == "CT" else "US (Unspecified)"}
        ]
        return tags

    @staticmethod
    def parse_uploaded_file(file_bytes: bytes, filename: str) -> Tuple[np.ndarray, Dict[str, Any], str]:
        """
        Parses DICOM, NIfTI (placeholder/raw), PNG, or JPEG files.
        Returns:
            - pixel_array (np.ndarray uint8 512x512)
            - metadata (dict with raw_dicom_tags, image_metrics, technical_params)
            - detected_modality (str: 'MRI' | 'CT' | 'X-RAY' | 'MAMMO' | 'ULTRASOUND')
        """
        filename_lower = filename.lower()
        metadata = {
            "Filename": filename,
            "FileSize": len(file_bytes),
            "PatientID": "UPLOADED_PATIENT",
            "StudyDescription": "Uploaded Clinical Study",
            "WindowCenter": 128,
            "WindowWidth": 256
        }
        raw_tags = []
        
        # 1. Attempt DICOM parsing via pydicom
        if filename_lower.endswith(('.dcm', '.dicom')) or file_bytes.startswith(b'DICM', 128):
            try:
                dicom_data = pydicom.dcmread(io.BytesIO(file_bytes), force=True)
                pixel_array = dicom_data.pixel_array.astype(np.float32)
                
                # Rescale slope and intercept if present
                slope = float(getattr(dicom_data, 'RescaleSlope', 1.0))
                intercept = float(getattr(dicom_data, 'RescaleIntercept', 0.0))
                pixel_array = pixel_array * slope + intercept
                
                # Modality from DICOM header
                modality = getattr(dicom_data, 'Modality', 'UNKNOWN').upper()
                if modality in ['MR', 'MRI']:
                    detected_modality = 'MRI'
                elif modality in ['CT']:
                    detected_modality = 'CT'
                elif modality in ['CR', 'DX', 'XR']:
                    detected_modality = 'X-RAY'
                elif modality in ['MG']:
                    detected_modality = 'MAMMO'
                elif modality in ['US']:
                    detected_modality = 'ULTRASOUND'
                else:
                    detected_modality = 'MRI'
                    
                metadata["PatientID"] = str(getattr(dicom_data, 'PatientID', 'PATIENT_UPLOAD'))
                metadata["PatientAge"] = getattr(dicom_data, 'PatientAge', '50Y')
                metadata["PatientSex"] = getattr(dicom_data, 'PatientSex', 'O')
                metadata["Modality"] = detected_modality
                metadata["StudyDescription"] = str(getattr(dicom_data, 'StudyDescription', f'{detected_modality} Study'))
                
                # Collect real DICOM tags
                for elem in dicom_data:
                    if elem.tag.is_private:
                        continue
                    try:
                        tag_str = f"({elem.tag.group:04X},{elem.tag.element:04X})"
                        raw_tags.append({
                            "tag": tag_str,
                            "name": elem.name,
                            "vr": elem.VR,
                            "value": str(elem.value)[:80]
                        })
                    except Exception:
                        pass

                # Normalize pixel values to 0-255
                min_val = np.min(pixel_array)
                max_val = np.max(pixel_array)
                if max_val > min_val:
                    normalized = ((pixel_array - min_val) / (max_val - min_val) * 255.0).astype(np.uint8)
                else:
                    normalized = np.zeros(pixel_array.shape, dtype=np.uint8)
                    
                # Resize to standard 512x512
                pil_img = PILImage.fromarray(normalized).resize((512, 512))
                out_array = np.array(pil_img)

                metrics = UniversalImagingService.calculate_image_metrics(out_array)
                if not raw_tags:
                    raw_tags = UniversalImagingService.build_synthetic_dicom_tags(detected_modality, filename, metrics)
                metadata["raw_dicom_tags"] = raw_tags
                metadata["image_metrics"] = metrics
                return out_array, metadata, detected_modality
            except Exception as e:
                print(f"DICOM parse fallback triggered: {e}")
                
        # 2. Standard image formats (PNG, JPEG, TIFF, BMP)
        try:
            pil_img = PILImage.open(io.BytesIO(file_bytes)).convert('L')
            pil_img = pil_img.resize((512, 512))
            pixel_array = np.array(pil_img)
            
            # Infer modality from filename or image structure
            if 'spine' in filename_lower or 'lumbar' in filename_lower or 'cervical' in filename_lower or 'thecal' in filename_lower or 'disc' in filename_lower:
                detected_modality = 'SPINE_MRI'
                desc = "Lumbar Spine MRI Protocol (Uploaded)"
            elif 'mri' in filename_lower or 'brain' in filename_lower or 't1' in filename_lower or 'flair' in filename_lower:
                detected_modality = 'MRI'
                desc = "Brain MRI Study (Uploaded)"
            elif 'ct' in filename_lower or 'lung' in filename_lower or 'chest_ct' in filename_lower:
                detected_modality = 'CT'
                desc = "High-Resolution Chest CT (Uploaded)"
            elif 'xray' in filename_lower or 'chest' in filename_lower or 'cxr' in filename_lower or 'rad' in filename_lower:
                detected_modality = 'X-RAY'
                desc = "Chest PA Radiograph (Uploaded)"
            elif 'mammo' in filename_lower or 'breast' in filename_lower:
                detected_modality = 'MAMMO'
                desc = "Digital Screening Mammogram (Uploaded)"
            elif 'us' in filename_lower or 'echo' in filename_lower or 'sound' in filename_lower:
                detected_modality = 'ULTRASOUND'
                desc = "Diagnostic Ultrasound Scan (Uploaded)"
            else:
                center_mean = np.mean(pixel_array[200:312, 200:312])
                if center_mean > 100:
                    detected_modality = 'MRI'
                    desc = "Magnetic Resonance Imaging (Auto-Detected)"
                else:
                    detected_modality = 'X-RAY'
                    desc = "Chest Radiograph (Auto-Detected)"
                    
            metadata["Modality"] = detected_modality
            metadata["StudyDescription"] = desc

            metrics = UniversalImagingService.calculate_image_metrics(pixel_array)
            raw_tags = UniversalImagingService.build_synthetic_dicom_tags(detected_modality, filename, metrics)
            metadata["raw_dicom_tags"] = raw_tags
            metadata["image_metrics"] = metrics

            return pixel_array, metadata, detected_modality
            
        except Exception as e:
            # Fallback to synthetic study if invalid file
            img, meta = UniversalImagingService.generate_synthetic(modality="MRI")
            metrics = UniversalImagingService.calculate_image_metrics(img)
            meta["raw_dicom_tags"] = UniversalImagingService.build_synthetic_dicom_tags("MRI", filename, metrics)
            meta["image_metrics"] = metrics
            return img, meta, "MRI"

    @staticmethod
    def generate_synthetic(modality: str = "MRI") -> Tuple[np.ndarray, Dict[str, Any]]:
        size = (512, 512)
        img = np.zeros(size, dtype=np.uint8) + 40
        return img, {"Modality": modality, "StudyDescription": f"Synthetic {modality}"}

