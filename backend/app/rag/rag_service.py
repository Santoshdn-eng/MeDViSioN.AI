from typing import List, Dict, Any
import numpy as np

class MedicalRAGService:
    def __init__(self):
        # Comprehensive curated medical guidelines knowledge base
        self.knowledge_base = [
            {
                "id": "KB-RAD-001",
                "title": "Fleischner Society Guidelines for Incidental Pulmonary Nodules",
                "content": "For solid pulmonary nodules in patients >= 35 years: Single solid nodule <= 6 mm requires no routine follow-up in low-risk patients; optional CT at 12 months in high-risk patients. Nodules 6-8 mm require CT follow-up at 6-12 months, then at 18-24 months if high risk. Nodules > 8 mm consider CT at 3 months, PET/CT, or tissue sampling.",
                "source": "Fleischner Society 2017 Guidelines",
                "category": "CT Chest"
            },
            {
                "id": "KB-RAD-002",
                "title": "ACR Appropriateness Criteria for Acute Chest Pain and Pleural Disease",
                "content": "Pleural effusion on upright chest radiograph is evidenced by blunting of the posterior costophrenic angle (> 50-75 mL fluid) or lateral costophrenic angle (> 175-200 mL). Lateral decubitus view or chest ultrasound can detect fluid volumes as small as 5-10 mL.",
                "source": "American College of Radiology (ACR) Guidelines",
                "category": "X-Ray"
            },
            {
                "id": "KB-RAD-003",
                "title": "WHO Classification of Tumors of the Central Nervous System (5th Edition)",
                "content": "Brain neoplasms are categorized based on histology and molecular biomarkers (IDH mutation, 1p/19q codeletion, MGMT methylation). On MRI, enhancing tumor (ET) on T1ce reflects breakdown of blood-brain barrier; non-enhancing high FLAIR signals demarcate infiltrative tumor and peritumoral vasogenic edema.",
                "source": "WHO CNS5 Classification & Neuro-oncology Protocol",
                "category": "MRI Brain"
            },
            {
                "id": "KB-RAD-004",
                "title": "Standard Cardiothoracic Ratio (CTR) Diagnostic Thresholds",
                "content": "The cardiothoracic ratio (CTR) is evaluated on a standard PA chest radiograph. A CTR exceeding 0.50 (50%) indicates cardiomegaly, though clinical correlation with body habitus, inspiratory effort, and AP vs PA projection is required.",
                "source": "Journal of Thoracic Imaging",
                "category": "X-Ray"
            }
        ]

    def search(self, query: str, top_k: int = 2) -> List[Dict[str, Any]]:
        """Simulates semantic vector search and re-ranking against trusted medical literature."""
        query_lower = query.lower()
        results = []
        for doc in self.knowledge_base:
            score = 0.0
            for word in query_lower.split():
                if word in doc["title"].lower() or word in doc["content"].lower():
                    score += 1.0
            if score > 0:
                results.append({**doc, "relevance_score": round(min(0.98, 0.65 + score * 0.1), 2)})
        
        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        if not results:
            results = [{**self.knowledge_base[0], "relevance_score": 0.70}]
        return results[:top_k]

rag_service = MedicalRAGService()
