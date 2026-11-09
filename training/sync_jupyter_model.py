"""
Jupyter Notebook Model Synchronization Helper for MEDVISION AI
Use this script inside any Jupyter Notebook to auto-update the live MedVision AI platform!
"""
import requests

def update_medvision_model(
    model_name: str,
    modality: str,
    version: str,
    metrics: dict,
    weights_path: str = None,
    medvision_api_url: str = "http://localhost:8085/api/v1/models/update"
):
    """
    Sends the freshly trained model metadata and weights directly to the running MedVision AI backend.
    """
    payload = {
        "model_name": model_name,
        "modality": modality.upper(),
        "version": version,
        "weights_path": weights_path,
        "metrics": metrics,
        "training_dataset": "Jupyter Notebook PyTorch Pipeline"
    }
    try:
        response = requests.post(medvision_api_url, json=payload)
        if response.status_code == 200:
            print("✅ Successfully synchronized model with MEDVISION AI!")
            print("Response:", response.json())
        else:
            print(f"❌ Error updating model ({response.status_code}):", response.text)
    except Exception as e:
        print("❌ Could not connect to MEDVISION AI server:", e)

# Example Usage:
if __name__ == "__main__":
    update_medvision_model(
        model_name="Custom-UNet-BrainMRI-FineTuned",
        modality="MRI",
        version="v3.4-jupyter-checkpoint",
        metrics={"Dice": 0.924, "AUROC": 0.981, "Loss": 0.042},
        weights_path="./checkpoints/best_mri_model.pth"
    )
