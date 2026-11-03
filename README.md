# OptiSight CDSS — Vitreoretinal Diabetic Retinopathy Diagnostic System

[![Vercel Deployment](https://img.shields.io/badge/Vercel-Live%20Demo-000000?style=for-the-badge&logo=vercel&logoColor=white)](https://optisight-cdss.vercel.app)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

OptiSight is an explainable multi-model clinical decision support system (CDSS) for 5-stage Diabetic Retinopathy (DR) grading adhering to the **International Clinical Diabetic Retinopathy (ICDR)** classification.

It fuses **Multi-Axis Vision Transformer (MaxViT-Tiny)** global attention and **Densely Connected Convolutional Network (DenseNet-121)** micro-texture feature reuse, achieving **85.45% accuracy** and **0.933 Quadratic Weighted Kappa (QWK)** on the APTOS 2019 benchmark and validated across 103 IDRiD cohorts.

🌐 **Live Web Application**: [https://optisight-cdss.vercel.app](https://optisight-cdss.vercel.app)

---

## 🔬 System Highlights

* **Full Uncropped Fundus Preservation**: Retains the complete, uncropped optical aperture (`max_dim=640`), preserving peripheral quadrants required for clinical rule validation.
* **Microvascular Edge-Guided Grad-CAM++**: Overcomes the multi-lesion limitation of standard Grad-CAM using second and third-order partial derivatives, bilateral edge-preserving filtering, and green-channel Canny edge modulation. Zero attention bleeding into dark margins.
* **AAO 4:2:1 Staging Rule Automation**: Automates the American Academy of Ophthalmology rule for Severe NPDR (4 quadrants of $\ge 20$ hemorrhages, 2 quadrants of venous beading, 1 quadrant of IRMA).
* **ETDRS 9-Zone Macular Grid Overlay**: Interactive concentric subfield overlay (1 mm central subfield, 3 mm inner ring, 6 mm outer ring) for evaluating Center-Involved Diabetic Macular Edema (CI-DME).
* **Biomarker-Integrated CDSS**: Synthesizes image-derived consensus grades with systemic patient biomarkers (HbA1c, diabetes duration, systolic blood pressure, DME) to produce a composite clinical risk score (0–100) and actionable triage timelines.
* **Dual-Mode Cloud Architecture**: Runs locally with high-performance PyTorch deep learning weights and deploys serverless 24/7 on Vercel with an edge-assisted computer vision fallback engine.

---

## 📐 Dual-Branch Hybrid Architecture

```
                    ┌─────────────────────────┐
                    │ Raw Retinal Fundus (RGB) │
                    └────────────┬────────────┘
                                 │
           ┌─────────────────────┴─────────────────────┐
           │                                           │
┌──────────▼───────────────┐               ┌───────────▼──────────────┐
│  MaxViT-Tiny (Multi-Axis)│               │ DenseNet-121 (Dense CNN) │
│  - MBConv Locality Bias  │               │ - Feature Reuse Highway  │
│  - Block Self-Attention  │               │ - 1024D Textural Repr    │
│  - Dilated Grid Attention│               │ - Micro-lesion Focus     │
└──────────┬───────────────┘               └───────────┬──────────────┘
           │ (512D)                                    │ (1024D)
           └─────────────────────┬─────────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │ Concatenation (1536D)   │
                    │ LayerNorm + Dropout(0.3)│
                    │ Linear(1536 -> 512)     │
                    │ GELU Activation         │
                    │ Dropout(0.2)            │
                    │ Linear(512 -> 5 Classes)│
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │ ICDR Grades 0 to 4      │
                    └─────────────────────────┘
```

### Feature Specialization
* **DenseNet-121**: Specializes in low-level micro-textures (microaneurysms, dot/blot hemorrhages, lipid exudates, cotton-wool spots, IRMA).
* **MaxViT-Tiny**: Specializes in global spatial context (optic disc to macula axis, vascular arcade continuity, multi-quadrant lesion symmetry, broad ischemic zones).

---

## 📊 Model Performance

| Model | Architecture | Accuracy | Quadratic Weighted Kappa (QWK) | Primary Specialty |
| :--- | :--- | :--- | :--- | :--- |
| **DenseNet-121** | Dense CNN | 81.20% | 0.898 | Microaneurysms & Exudates |
| **MaxViT-Tiny** | Multi-Axis Transformer | 83.15% | 0.912 | Global Arcades & Quadrants |
| **OptiSight Hybrid** | **Dual-Branch Late Fusion** | **85.45%** | **0.933** | **Optimal Balanced Staging** |

---

## 🛠️ Project Structure

```
├── optisight_app.py                      # Core CDSS Flask Application & Medical GUI
├── api/
│   └── index.py                          # Vercel Serverless Function Entry Point
├── requirements.txt                      # Lightweight dependencies for deployment
├── vercel.json                           # Vercel routing configuration
├── DEPLOYMENT_GUIDE.md                   # Cloud & local deployment guide
├── OptiSight_DR_Viva_Comprehensive_Guide.docx # Complete Viva Voce Guidebook (Word)
├── VIVA_QUESTIONS_AND_ANSWERS.md         # Comprehensive Viva Q&A (Markdown)
├── confusion_matrix_75.png               # Hybrid Confusion Matrix
├── densenet121_cm.png                    # DenseNet-121 Confusion Matrix
├── maxvit_cm.png                         # MaxViT Confusion Matrix
├── LICENSE                               # MIT License
└── README.md                             # Documentation
```

---

## 🚀 Quick Start (Local Run)

### 1. Clone the repository
```bash
git clone git@github.com:Virajmahale21/OPTISIGHT.git
cd OPTISIGHT
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
# For full local PyTorch deep learning weights:
pip install torch torchvision timm
```

### 3. Run the application
```bash
python optisight_app.py
```
Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

---

## 📚 Examination & Viva Resources

For students, researchers, and clinicians reviewing the technical defense:
* [OptiSight_DR_Viva_Comprehensive_Guide.docx](OptiSight_DR_Viva_Comprehensive_Guide.docx) — 34 exhaustive questions & answers formatted for printing.
* [VIVA_QUESTIONS_AND_ANSWERS.md](VIVA_QUESTIONS_AND_ANSWERS.md) — Quick-reference markdown version covering model comparisons, preprocessing mathematics, and clinical guidelines.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
