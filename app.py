import gradio as gr
import torch
import torch.nn as nn
import timm
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import warnings
warnings.filterwarnings('ignore')

# ════════════════════════════════════
# CONFIG
# ════════════════════════════════════
NUM_CLASSES  = 4
class_names  = ['Cataract', 'Diabetic Retinopathy',
                'Glaucoma', 'Normal']

descriptions = {
    'Cataract': 
        "La cataracte est un trouble de la vision causé "
        "par l'opacification du cristallin. "
        "Symptômes : vision floue, sensibilité à la lumière. "
        "Traitement : intervention chirurgicale simple. ",
    
    'Diabetic Retinopathy': 
        "La rétinopathie diabétique est une complication "
        "du diabète qui endommage les vaisseaux sanguins "
        "de la rétine. Peut causer la cécité si non traitée. "
        "Traitement : contrôle du diabète + laser. ",
    
    'Glaucoma': 
        "Le glaucome est une maladie qui endommage "
        "le nerf optique, souvent due à une pression "
        "oculaire élevée. Peut causer la cécité. "
        "Traitement : gouttes oculaires + chirurgie. ",
    
    'Normal': 
        "Aucune maladie oculaire détectée. "
        "L'œil semble en bonne santé. "
        "Continuez les examens réguliers chez "
        "votre ophtalmologue. "
}

device = torch.device('cpu')

# ════════════════════════════════════
# MODÈLE
# ════════════════════════════════════
class ResNet50Classifier(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.backbone = timm.create_model(
            'resnet50',
            pretrained  = False,
            num_classes = 0,
            global_pool = 'avg'
        )
        in_features = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.head(self.backbone(x))

# ── Charger modèle ──
model = ResNet50Classifier(NUM_CLASSES)
model.load_state_dict(
    torch.load('best_model_resnet.pth',
                map_location=device)
)
model.eval()
print("✅ Modèle chargé")

# ════════════════════════════════════
# CLAHE
# ════════════════════════════════════
def apply_clahe(image):
    img  = np.array(image)
    lab  = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0,
                              tileGridSize=(8, 8))
    l    = clahe.apply(l)
    lab  = cv2.merge((l, a, b))
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return Image.fromarray(result)

# ════════════════════════════════════
# TRANSFORMS
# ════════════════════════════════════
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Lambda(apply_clahe),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std= [0.229, 0.224, 0.225]
    )
])

transform_gradcam = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std= [0.229, 0.224, 0.225]
    )
])

# ════════════════════════════════════
# GRADCAM
# ════════════════════════════════════
def get_gradcam(image_pil):
    target_layers = [model.backbone.layer4[-1]]
    cam = GradCAM(model=model,
                   target_layers=target_layers)

    img_224 = image_pil.resize((224, 224))
    img_arr = np.array(img_224) / 255.0

    input_tensor  = transform_gradcam(
        image_pil
    ).unsqueeze(0)

    grayscale_cam = cam(input_tensor=input_tensor)[0]
    visualization = show_cam_on_image(
        img_arr.astype(np.float32),
        grayscale_cam,
        use_rgb=True
    )
    return Image.fromarray(visualization)

# ════════════════════════════════════
# PREDICTION
# ════════════════════════════════════
def predict(image):
    if image is None:
        return None, None, "Veuillez uploader une image", ""

    # PIL Image
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image).convert('RGB')
    else:
        image = image.convert('RGB')

    # Prédiction
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(input_tensor)
        probs   = torch.softmax(outputs, dim=1)[0]

    # Résultats
    probs_dict = {
        class_names[i]: float(probs[i])
        for i in range(NUM_CLASSES)
    }

    predicted_class = class_names[probs.argmax().item()]
    confidence      = float(probs.max()) * 100

    # GradCAM
    gradcam_img = get_gradcam(image)

    # Résultat texte
    result_text = (
        f"Maladie détectée : {predicted_class}\n"
        f"Confiance        : {confidence:.1f}%"
    )

    # Description
    description = descriptions[predicted_class]

    return gradcam_img, probs_dict, result_text, description

# ════════════════════════════════════
# INTERFACE GRADIO
# ════════════════════════════════════
with gr.Blocks(title="Eye Disease Detection") as app:

    gr.Markdown("""
    # 👁️ Eye Disease Detection
    ### Détection automatique de maladies oculaires
    **Modèle** : ResNet-50 | **Accuracy** : 88.15% | **F1** : 88.01%
    
    **Maladies détectées** : Cataract • Diabetic Retinopathy • Glaucoma • Normal
    """)

    with gr.Row():
        # Colonne gauche
        with gr.Column():
            input_image = gr.Image(
                label="Upload image de l'œil",
                type="pil"
            )
            predict_btn = gr.Button(
                "Analyser",
                variant="primary",
                size="lg"
            )

        # Colonne droite
        with gr.Column():
            gradcam_output = gr.Image(
                label="GradCAM — Zones analysées"
            )
            result_output = gr.Textbox(
                label="Résultat",
                lines=2
            )

    with gr.Row():
        probs_output = gr.Label(
            label="Probabilités par classe",
            num_top_classes=4
        )
        desc_output = gr.Textbox(
            label="Description de la maladie",
            lines=4
        )

    gr.Markdown("""
    ---
    ⚠️ **Avertissement** : Cet outil est à titre éducatif uniquement.
    Consultez un ophtalmologue pour un diagnostic médical.
    """)

    predict_btn.click(
        fn     = predict,
        inputs = [input_image],
        outputs= [gradcam_output, probs_output,
                  result_output, desc_output]
    )

    # Exemples
    gr.Examples(
        examples   = [],
        inputs     = [input_image]
    )

app.launch()