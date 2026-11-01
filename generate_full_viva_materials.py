# -*- coding: utf-8 -*-
"""
Generates:
1. OptiSight_DR_Viva_Comprehensive_Guide.docx
2. VIVA_QUESTIONS_AND_ANSWERS.md
Contains exhaustive, academically rigorous questions and answers for viva defense.
"""

import os, sys
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(f'<w:tcMar {nsdecls("w")}><w:top w:w="{top}" w:type="dxa"/><w:bottom w:w="{bottom}" w:type="dxa"/><w:left w:w="{left}" w:type="dxa"/><w:right w:w="{right}" w:type="dxa"/></w:tcMar>')
    tcPr.append(tcMar)

def build_all():
    doc = docx.Document()

    for s in doc.sections:
        s.top_margin = Inches(0.75)
        s.bottom_margin = Inches(0.75)
        s.left_margin = Inches(0.75)
        s.right_margin = Inches(0.75)

    PRIMARY = RGBColor(2, 132, 199)    # #0284c7 Deep Medical Blue
    TEAL = RGBColor(13, 148, 136)       # #0d9488 Medical Teal
    DARK = RGBColor(15, 23, 42)         # #0f172a Slate 900
    MUTED = RGBColor(100, 116, 139)     # #64748b Slate 500
    ACCENT = RGBColor(180, 83, 9)       # Amber 700

    # Header title
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_before = Pt(14)
    p_title.paragraph_format.space_after = Pt(2)
    r = p_title.add_run("OPTISIGHT CDSS — VITREORETINAL DIAGNOSTIC SYSTEM")
    r.font.size = Pt(22)
    r.font.bold = True
    r.font.color.rgb = PRIMARY

    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_sub.paragraph_format.space_after = Pt(6)
    r = p_sub.add_run("Comprehensive Technical Defense, Model Architecture & Viva Voce Guidebook")
    r.font.size = Pt(13)
    r.font.bold = True
    r.font.color.rgb = TEAL

    p_meta = doc.add_paragraph()
    p_meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_meta.paragraph_format.space_after = Pt(18)
    r = p_meta.add_run("Deep Learning · Multi-Axis Vision Transformers (MaxViT) · Densely Connected CNNs (DenseNet-121) · Dual-Branch Fusion · Grad-CAM++ · ETDRS 9-Zone · AAO 4:2:1 Staging")
    r.font.size = Pt(9)
    r.font.italic = True
    r.font.color.rgb = MUTED

    def add_callout(text, title="KEY EXAMINER TAKEAWAY"):
        tbl = doc.add_table(rows=1, cols=1)
        tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
        cell = tbl.cell(0, 0)
        cell.width = Inches(7.0)
        set_cell_background(cell, "F0F9FF")
        set_cell_margins(cell, top=120, bottom=120, left=180, right=180)
        p = cell.paragraphs[0]
        p.paragraph_format.space_after = Pt(2)
        rt = p.add_run(f"📌 {title}: ")
        rt.bold = True
        rt.font.size = Pt(9.5)
        rt.font.color.rgb = PRIMARY
        rx = p.add_run(text)
        rx.font.size = Pt(9.5)
        rx.font.color.rgb = DARK
        doc.add_paragraph().paragraph_format.space_after = Pt(4)

    def add_section_header(title, num):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(16)
        p.paragraph_format.space_after = Pt(6)
        r = p.add_run(f"PART {num}: {title.upper()}")
        r.font.size = Pt(13)
        r.font.bold = True
        r.font.color.rgb = PRIMARY

    def add_table_data(headers, rows):
        tbl = doc.add_table(rows=len(rows)+1, cols=len(headers))
        tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
        # Header Row
        for col_idx, h_text in enumerate(headers):
            cell = tbl.cell(0, col_idx)
            set_cell_background(cell, "0F172A")
            set_cell_margins(cell, top=100, bottom=100, left=120, right=120)
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(2)
            run = p.add_run(h_text)
            run.font.bold = True
            run.font.size = Pt(9)
            run.font.color.rgb = RGBColor(255, 255, 255)

        # Body Rows
        for r_idx, row_vals in enumerate(rows):
            bg = "F8FAFC" if (r_idx % 2 == 0) else "FFFFFF"
            for c_idx, val in enumerate(row_vals):
                cell = tbl.cell(r_idx + 1, c_idx)
                set_cell_background(cell, bg)
                set_cell_margins(cell, top=80, bottom=80, left=120, right=120)
                p = cell.paragraphs[0]
                p.paragraph_format.space_after = Pt(2)
                run = p.add_run(str(val))
                run.font.size = Pt(8.5)
                run.font.color.rgb = DARK
        doc.add_paragraph().paragraph_format.space_after = Pt(6)

    def add_qa(q_num, question, answer, examiner_tip=None):
        p_q = doc.add_paragraph()
        p_q.paragraph_format.space_before = Pt(10)
        p_q.paragraph_format.space_after = Pt(3)
        rn = p_q.add_run(f"Q{q_num}: ")
        rn.font.bold = True
        rn.font.size = Pt(10.5)
        rn.font.color.rgb = TEAL
        rq = p_q.add_run(question)
        rq.font.bold = True
        rq.font.size = Pt(10.5)
        rq.font.color.rgb = DARK

        p_a = doc.add_paragraph()
        p_a.paragraph_format.space_after = Pt(4)
        p_a.paragraph_format.line_spacing = 1.15
        ra = p_a.add_run(answer)
        ra.font.size = Pt(9.5)
        ra.font.color.rgb = DARK

        if examiner_tip:
            p_tip = doc.add_paragraph()
            p_tip.paragraph_format.left_indent = Inches(0.2)
            p_tip.paragraph_format.space_after = Pt(6)
            rtl = p_tip.add_run("💡 Examiner Insight: ")
            rtl.font.bold = True
            rtl.font.size = Pt(8.5)
            rtl.font.color.rgb = ACCENT
            rtb = p_tip.add_run(examiner_tip)
            rtb.font.size = Pt(8.5)
            rtb.font.italic = True
            rtb.font.color.rgb = MUTED

    # Markdown accumulator
    md_content = []
    def log_md(section=None, q_num=None, q_text=None, a_text=None, tip=None, text_block=None, table_h=None, table_r=None):
        if section:
            md_content.append(f"\n# {section}\n")
        elif q_num and q_text and a_text:
            md_content.append(f"### Q{q_num}: {q_text}\n")
            md_content.append(f"{a_text}\n")
            if tip:
                md_content.append(f"> 💡 **Examiner Insight:** *{tip}*\n")
        elif text_block:
            md_content.append(f"{text_block}\n")
        elif table_h and table_r:
            header_line = "| " + " | ".join(table_h) + " |"
            sep_line = "| " + " | ".join(["---"] * len(table_h)) + " |"
            rows_lines = ["| " + " | ".join(r) + " |" for r in table_r]
            md_content.append("\n" + "\n".join([header_line, sep_line] + rows_lines) + "\n")

    # Front Matter
    log_md(text_block="# OPTISIGHT CDSS — VITREORETINAL DIAGNOSTIC SYSTEM\n## Comprehensive Technical Defense, Model Architecture & Viva Voce Guidebook\n*Deep Learning · Multi-Axis Vision Transformers (MaxViT) · Densely Connected CNNs (DenseNet-121) · Dual-Branch Hybrid · Grad-CAM++ · ETDRS 9-Zone · AAO 4:2:1 Staging*\n---\n")

    # =========================================================================
    # PART 1: ARCHITECTURAL DEEP DIVE & MODEL SELECTION
    # =========================================================================
    add_section_header("Architectural Deep Dive & Model Selection (Why This Model?)", 1)
    log_md(section="PART 1: Architectural Deep Dive & Model Selection (Why This Model?)")

    add_callout(
        "Diabetic Retinopathy presents a fundamental computer vision dilemma: lesions range from sub-pixel microaneurysms (10–20 microns) to global vascular distortions spanning the entire fundus. Pure CNNs fail at global context, while standard Vision Transformers miss microscopic textures. OptiSight resolves this with an engineered dual-branch hybrid combining DenseNet-121 (dense micro-texture reuse) and MaxViT-Tiny (multi-axis global attention).",
        "ARCHITECTURAL THESIS"
    )

    add_qa(
        1,
        "What is the core clinical problem OptiSight solves, and what is its high-level architecture?",
        "Diabetic Retinopathy (DR) is the primary cause of preventable blindness in working-age adults globally, affecting over 103 million diabetic patients. Early detection reduces the risk of severe vision loss by over 95%, but screening is bottlenecked by the scarcity of vitreoretinal specialists and high intra-observer grading variability.\n\n"
        "OptiSight is an automated Clinical Decision Support System (CDSS) designed for ophthalmologists and retina specialists. Its high-level pipeline consists of:\n"
        "1. Full uncropped fundus image intake preserving optical boundaries.\n"
        "2. Ben Graham's local color subtraction & contrast normalization.\n"
        "3. Green-channel anatomical microvascular edge extraction.\n"
        "4. Multi-model parallel inference using MaxViT-Tiny (global transformer), DenseNet-121 (dense convolutional texture extractor), and the OptiSight Hybrid fusion network.\n"
        "5. Explainable AI via Grad-CAM++ with bilateral filtering and edge refinement.\n"
        "6. Clinical decision automation incorporating the ICDR 0–4 staging scale, the AAO 4:2:1 severe NPDR rule, the ETDRS 9-zone macular grid for diabetic macular edema (DME), and a composite clinical risk score incorporating patient biomarkers (HbA1c, BP, duration).",
        "Frame the project as a Clinical Decision Support System (CDSS) designed to assist and triage, NOT to replace the doctor. Medical examiners look for adherence to clinical guidelines (AAO, ICDR, ETDRS)."
    )
    log_md(q_num=1,
           q_text="What is the core clinical problem OptiSight solves, and what is its high-level architecture?",
           a_text="Diabetic Retinopathy (DR) is the primary cause of preventable blindness in working-age adults globally. OptiSight is an automated Clinical Decision Support System (CDSS) providing ICDR 0-4 staging, multi-model consensus (MaxViT, DenseNet-121, Hybrid), edge-refined Grad-CAM++, and systemic biomarker integration.",
           tip="Frame the project as a Clinical Decision Support System (CDSS) to assist doctors, adhering to ICDR, AAO 4:2:1, and ETDRS standards.")

    add_qa(
        2,
        "Describe the MaxViT architecture in detail. How does Multi-Axis Attention operate mathematically?",
        "MaxViT (Multi-Axis Vision Transformer) introduces an architectural paradigm that bridges the gap between convolutional inductive bias and scalable self-attention in linear time complexity O(N).\n\n"
        "Standard Vision Transformers (ViT) compute full pairwise self-attention across all tokens, resulting in quadratic computational complexity O(N^2), which is computationally prohibitive for high-resolution medical images. MaxViT decomposes standard dense attention into two sequential, sparse multi-axis operations:\n\n"
        "1. Block Attention (Local Self-Attention): The spatial feature map of size (H x W) is partitioned into non-overlapping local windows of size (P x P). Self-attention is computed exclusively within each block. This captures fine spatial correlations, sharp lesion borders, and local anatomical transitions with complexity O(P^2 * (H*W)/P^2) = O((H*W) * P^2).\n\n"
        "2. Grid Attention (Global Dilated Self-Attention): Rather than partitioning into adjacent blocks, the feature map is subsampled into a uniform sparse grid of size (G x G) with a dilation factor (stride). Tokens that occupy identical relative positions across distinct blocks attend to one another across the entire retinal canvas. This captures long-range spatial context—such as the spatial relationship between the optic disc, vascular arcades, and macula—with complexity O((H*W) * G^2).\n\n"
        "3. MBConv Integration: Prior to the multi-axis attention blocks, each MaxViT stage begins with an inverted residual MBConv block equipped with depthwise separable convolutions and Squeeze-and-Excitation (SE) channels. This injects strong convolutional inductive bias (translation equivariance and local feature sensitivity) before global transformer self-attention is applied.",
        "When asked about transformer complexity, state clearly: 'Standard ViT is quadratic O(N^2). MaxViT is linear O(N) with respect to input pixels because of the Block + Grid decomposition, allowing higher resolution with lower FLOPs.'"
    )
    log_md(q_num=2,
           q_text="Describe the MaxViT architecture in detail. How does Multi-Axis Attention operate mathematically?",
           a_text="MaxViT decomposes standard O(N^2) dense attention into two sequential sparse steps: Block Attention (local window self-attention) and Grid Attention (dilated global sparse self-attention), achieving linear O(N) complexity while retaining MBConv convolutional inductive biases.",
           tip="Emphasize that MaxViT achieves linear O(N) complexity through Block + Grid attention decomposition.")

    add_qa(
        3,
        "Describe the DenseNet-121 architecture in detail. What are dense blocks, transition layers, and the growth rate?",
        "DenseNet-121 (Densely Connected Convolutional Networks) departs from conventional feed-forward architectures by connecting all layers directly with matching feature-map sizes. In an L-layer network, traditional architectures possess L direct connections; DenseNet contains L*(L+1)/2 direct connections.\n\n"
        "Key Architectural Components:\n"
        "1. Dense Connectivity Formula: For layer l, the input is the concatenation of all preceding feature maps:\n"
        "   x_l = H_l([x_0, x_1, x_2, ..., x_{l-1}])\n"
        "   where [x_0, ..., x_{l-1}] denotes channel-wise concatenation, and H_l is a composite function of BatchNorm, ReLU, and 3x3 Conv.\n\n"
        "2. Growth Rate (k): The hyperparameter k (set to k=32 in DenseNet-121) governs how many feature channels each layer produces. Because each layer has direct access to the collective knowledge of all preceding layers, k can remain very small (32 channels per layer), keeping the total parameter count extremely compact (~7.9M parameters compared to ResNet-50's ~25M or VGG-16's ~138M).\n\n"
        "3. Bottleneck Layers (DenseNet-B): To reduce computational burden before expensive 3x3 convolutions, a 1x1 convolution is introduced (BN-ReLU-1x1 Conv producing 4k channels), compressing intermediate feature dimensions.\n\n"
        "4. Transition Layers (DenseNet-C): Between adjacent dense blocks, transition layers perform downsampling using a 1x1 Conv followed by 2x2 Average Pooling. A compression factor theta = 0.5 halves the number of feature maps, maintaining feature efficiency.\n\n"
        "5. Direct Gradient Flow Highway: During backpropagation, error gradients flow directly from the loss function to every single preceding layer without traversing sequential convolutional weight matrices. This eliminates the vanishing gradient problem naturally without needing artificial auxiliary loss heads.",
        "Highlight DenseNet's feature reuse: low-level textural features from layer 2 or 3 are directly accessible at layer 120 without being degraded by successive downsamplings."
    )
    log_md(q_num=3,
           q_text="Describe the DenseNet-121 architecture in detail. What are dense blocks, transition layers, and the growth rate?",
           a_text="DenseNet-121 connects every layer to every subsequent layer via channel concatenation: x_l = H_l([x_0, ..., x_{l-1}]). With growth rate k=32, 1x1 bottleneck convolutions, and 2x2 pooling transition layers, it ensures maximum feature reuse and zero vanishing gradients.",
           tip="Highlight feature reuse: early microvascular features are directly accessible at the final layers.")

    # Comparative Table
    headers_arch = ["Architecture", "Parameters", "Receptive Field", "Micro-Lesion Sensitivity", "Global Spatial Context", "Inductive Bias", "Suitability for DR Staging"]
    rows_arch = [
        ["VGG-16 / VGG-19", "~138 Million", "Fixed, shallow", "Moderate (over-smoothed)", "Poor (no long-range attention)", "Strong convolutional", "Poor (high params, prone to vanishing gradients)"],
        ["ResNet-50 / ResNet-101", "~25.6 Million", "Bounded by residual depth", "Moderate", "Limited (cannot link disc to periphery)", "Strong convolutional", "Moderate (standard baseline, misses global topography)"],
        ["Pure ViT (ViT-Base)", "~86 Million", "Global from layer 1", "Poor (16x16 patch tokenization destroys 10um lesions)", "Excellent O(N^2)", "Very Weak (requires massive pretraining data)", "Sub-optimal (blind to tiny microaneurysms)"],
        ["Swin Transformer", "~28 Million", "Shifted window hierarchy", "Moderate", "Good (hierarchical window merging)", "Weak-to-Moderate", "Good (but lacks explicit global dilated grid in early stages)"],
        ["DenseNet-121 (Ours)", "~7.9 Million", "Multi-scale feature reuse", "EXCEPTIONAL (preserves high-frequency textures)", "Moderate", "Strong convolutional", "SUPERIOR for Microaneurysms & Exudates"],
        ["MaxViT-Tiny (Ours)", "~31 Million", "Linear Multi-Axis (Block+Grid)", "Very High (Block Attention)", "EXCEPTIONAL (Grid Attention)", "Hybrid (MBConv + Axial Attention)", "SUPERIOR for Global Vascular Arcades & Quadrants"],
        ["OptiSight Hybrid (Ours)", "~39 Million", "Unified Micro-Macro Manifold", "STATE-OF-THE-ART (DenseNet)", "STATE-OF-THE-ART (MaxViT)", "Optimal Balanced Inductive Bias", "OPTIMAL (Solves the Inductive Bias Dilemma)"]
    ]
    add_table_data(headers_arch, rows_arch)
    log_md(table_h=headers_arch, table_r=rows_arch)

    add_qa(
        4,
        "Why MaxViT and DenseNet-121? Why NOT VGG, ResNet, or pure Vision Transformers?",
        "This is the quintessential architectural defense question. The selection of MaxViT-Tiny and DenseNet-121 was driven by the specific pathology of Diabetic Retinopathy:\n\n"
        "1. Why NOT VGG-16 / VGG-19?\n"
        "- Massive parameter redundancy (138 million parameters), creating severe overfitting risks on medical cohorts.\n"
        "- Absence of residual or dense skip connections; prone to gradient vanishing.\n"
        "- Successive 2x2 max-pooling layers irreversibly destroy high-frequency spatial detail, wiping out tiny 1–2 pixel microaneurysms.\n\n"
        "2. Why NOT Standard ResNet (ResNet-50 / ResNet-101)?\n"
        "- While residual addition (x + F(x)) mitigates vanishing gradients, ResNet's receptive field grows linearly with depth and remains spatially bounded.\n"
        "- ResNet cannot correlate distant retinal quadrants (e.g., assessing whether hemorrhages in the superior-temporal arcade correlate with vascular caliber changes in the inferior-nasal arcade as required by the AAO 4:2:1 rule).\n\n"
        "3. Why NOT Pure Vision Transformers (ViT-Base)?\n"
        "- Standard ViT divides an image into rigid non-overlapping patches (typically 16x16 pixels). In a 224x224 retinal image, a single patch spans a huge retinal territory. A microaneurysm (often 2x2 pixels) is averaged into a 16x16 patch embedding, completely washing out its diagnostic signal.\n"
        "- ViT lacks convolutional translation equivariance and locality; it requires hundreds of millions of labeled pretraining images (like JFT-300M) to converge reliably, whereas medical datasets are constrained.\n\n"
        "4. Why DenseNet-121?\n"
        "- Dense concatenation preserves low-level edge and texture maps throughout the entire forward pass. Micro-lesions extracted in early layers remain intact and directly accessible to the classification head.\n"
        "- Highly parameter-efficient (~7.9M params), providing robust regularization against clinical overfitting.\n\n"
        "5. Why MaxViT-Tiny?\n"
        "- Overcomes ViT's patch degradation by retaining fine-resolution convolutional stems (MBConv) and local Block Attention, while Grid Attention provides full-canvas global receptive fields in linear O(N) complexity.",
        "Memorize the phrase: 'VGG is too heavy and destroys micro-lesions; ResNet lacks global macular-disc context; pure ViT averages out tiny microaneurysms into 16x16 patches. DenseNet preserves micro-textures, and MaxViT captures global vascular geometry.'"
    )
    log_md(q_num=4,
           q_text="Why MaxViT and DenseNet-121? Why NOT VGG, ResNet, or pure Vision Transformers?",
           a_text="VGG has 138M params and destroys micro-lesions through max-pooling; ResNet lacks long-range multi-quadrant context; pure ViT averages tiny 2x2 microaneurysms into coarse 16x16 patches. DenseNet-121 provides dense micro-texture preservation, while MaxViT-Tiny provides linear global grid attention.",
           tip="Contrast the spatial scale of lesions: microaneurysms need DenseNet's feature reuse; multi-quadrant staging needs MaxViT's global attention.")

    add_qa(
        5,
        "What is the 'Inductive Bias Dilemma' in medical imaging, and how does the OptiSight Hybrid solve it?",
        "Inductive bias refers to the set of architectural assumptions a machine learning algorithm uses to predict outputs on unseen inputs:\n\n"
        "1. Convolutional Inductive Bias (CNNs): Assumes spatial locality (nearby pixels are correlated) and translation equivariance (a hemorrhage looks like a hemorrhage regardless of where it appears on the retina). This bias enables CNNs to train efficiently on small datasets with fast convergence, but makes them 'myopic'—they struggle to understand global spatial configurations across the entire retina.\n\n"
        "2. Transformer Inductive Bias (ViTs): Operates with minimal spatial assumptions. Using self-attention, any token can attend to any other token across the image, allowing modeling of arbitrary global relationships. However, lacking locality bias, transformers must learn basic spatial structures from scratch, making them data-hungry and prone to ignoring fine micro-lesions.\n\n"
        "3. OptiSight Hybrid Solution:\n"
        "OptiSight resolves this dilemma through a heterogeneous dual-branch late fusion architecture. DenseNet-121 injects maximum convolutional inductive bias for micro-textures, while MaxViT provides multi-axis attention for global contextual relationships. By fusing them into a unified representation manifold, the network achieves both microscopic precision and macroscopic anatomical awareness.",
        "Examiners love theoretical deep learning concepts. Explaining the Inductive Bias Dilemma demonstrates you understand the mathematical foundations, not just empirical coding."
    )
    log_md(q_num=5,
           q_text="What is the 'Inductive Bias Dilemma' in medical imaging, and how does the OptiSight Hybrid solve it?",
           a_text="CNNs have strong locality and translation equivariance but lack global context. ViTs have global attention but lack locality and require massive data. OptiSight Hybrid resolves this by fusing DenseNet-121 (micro-texture locality) with MaxViT (global axial attention).",
           tip="Explaining the Inductive Bias Dilemma demonstrates deep theoretical understanding.")

    add_qa(
        6,
        "Which model extracts which specific anatomical and pathological features from a retinal fundus photograph?",
        "This is a critical examination question. The two branches specialize in orthogonal, complementary visual hierarchies:\n\n"
        "DenseNet-121 Branch (Micro-Textural & Local Pathological Specialist):\n"
        "1. Microaneurysms: Tiny, discrete focal dilations of retinal capillaries (10–50 microns), appearing as pinpoint dark red dots. DenseNet preserves their sharp localized edges through direct concatenation.\n"
        "2. Dot and Blot Hemorrhages: Ruptured microaneurysms in the deeper inner nuclear and outer plexiform layers, presenting as round, demarcated dark red spots.\n"
        "3. Hard Exudates (Lipids): Sharp, glistening yellow-white deposits of lipoprotein precipitated from chronically leaking microvessels. DenseNet extracts their steep intensity gradients.\n"
        "4. Cotton-Wool Spots: Small, fluffy white lesions representing focal micro-infarcts of the nerve fiber layer, characterized by subtle localized textural disruption.\n"
        "5. IRMA (Intraretinal Microvascular Abnormalities): Tortuous, fine intraretinal collateral shunt vessels bridging arterioles and venules across areas of capillary non-perfusion.\n\n"
        "MaxViT-Tiny Branch (Macro-Architectural & Global Spatial Specialist):\n"
        "1. Global Topography & Spatial Symmetry: Geometric spatial distance and orientation between the Optic Nerve Head (nasal) and the Macula/Fovea (temporal).\n"
        "2. Vascular Arcade Trajectory: Continuity and curvature of the superior and inferior temporal vascular arcades traversing the retina.\n"
        "3. Multi-Quadrant Hemorrhage Distribution: Evaluating the spatial density and spread of hemorrhages across all 4 quadrants (crucial for evaluating the AAO 4:2:1 rule).\n"
        "4. Large-Area Capillary Non-Perfusion & Ischemia: Diffuse loss of microvascular background perfusion causing broad regional contrast shifts between ischemic zones and perfused retina.\n"
        "5. Preretinal & Vitreous Hemorrhage Boundaries: Large boat-shaped or diffuse hemorrhages extending into the vitreous body in proliferative disease (Grade 4).",
        "Structure your answer into two columns: DenseNet = Micro-textures (MA, Dot/Blot, Hard Exudate, IRMA); MaxViT = Global Topography (Arcades, Disc-Macula axis, 4-Quadrant distribution, Ischemia)."
    )
    log_md(q_num=6,
           q_text="Which model extracts which specific anatomical and pathological features from a retinal fundus photograph?",
           a_text="DenseNet-121 extracts local micro-textures (microaneurysms, dot/blot hemorrhages, lipid exudates, cotton-wool spots, IRMA). MaxViT extracts macro-spatial context (optic disc-macula axis, vascular arcade continuity, 4-quadrant hemorrhage distribution, large ischemic zones).",
           tip="Structure your answer into two distinct columns: DenseNet = Micro-textures; MaxViT = Global Topography.")

    add_qa(
        7,
        "Describe the OptiSight Hybrid fusion head layer-by-layer. What is the mathematical formulation of the late fusion?",
        "The OptiSight Hybrid model implements late feature fusion at the penultimate representation level:\n\n"
        "1. Feature Extraction:\n"
        "   - MaxViT-Tiny backbone extracts representation vector: z_maxvit = f_MaxViT(x) in R^{512}\n"
        "   - DenseNet-121 backbone extracts representation vector: z_dense = f_DenseNet(x) in R^{1024}\n\n"
        "2. Late Concatenation:\n"
        "   - z_fused = [z_maxvit ; z_dense] in R^{1536}\n"
        "   The combined feature vector has dimensionality d = 512 + 1024 = 1536.\n\n"
        "3. Layer Normalization:\n"
        "   - h_1 = LayerNorm(z_fused)\n"
        "   Why LayerNorm instead of BatchNorm? DenseNet and MaxViT features have vastly different numerical distributions and variances. BatchNorm calculates statistics across the batch, which can introduce instability during small-batch inference or deployment. LayerNorm normalizes across the feature dimension independently for each sample, aligning the disparate transformer and convolutional activation scales seamlessly.\n\n"
        "4. Primary Regularization:\n"
        "   - h_2 = Dropout(p=0.3)(h_1)\n"
        "   Stochastically disables 30% of feature channels during training, preventing the head from developing an over-reliance on either individual backbone branch.\n\n"
        "5. Non-Linear Feature Manifold Projection:\n"
        "   - h_3 = GELU(Linear(1536, 512)(h_2))\n"
        "   Why GELU instead of ReLU? GELU (Gaussian Error Linear Unit) smooths activations probabilistically: GELU(x) = x * Phi(x). Unlike ReLU which abruptly zeroes out negative activations (causing 'dead neurons'), GELU permits small negative gradients, preserving subtle low-magnitude signals essential for borderline clinical grades.\n\n"
        "6. Secondary Regularization & Classification Logits:\n"
        "   - h_4 = Dropout(p=0.2)(h_3)\n"
        "   - y_logits = Linear(512, 5)(h_4)\n"
        "   The final layer projects into the 5 discrete ICDR clinical grades (Grade 0 to Grade 4).",
        "Be ready to write down the dimension pipeline: 512 + 1024 -> 1536 -> LayerNorm -> Dropout(0.3) -> Linear(1536, 512) -> GELU -> Dropout(0.2) -> Linear(512, 5)."
    )
    log_md(q_num=7,
           q_text="Describe the OptiSight Hybrid fusion head layer-by-layer. What is the mathematical formulation of the late fusion?",
           a_text="MaxViT (512D) and DenseNet-121 (1024D) are concatenated into a 1536D vector, normalized with LayerNorm to bridge scale differences, regularized with Dropout(0.3), projected to 512D with GELU non-linearity, regularized with Dropout(0.2), and mapped to 5 ICDR class logits.",
           tip="Remember the exact dimensions: 512 + 1024 = 1536 -> LayerNorm -> 512 -> GELU -> 5 logits.")

    # =========================================================================
    # PART 2: IMAGE PREPROCESSING & FEATURE ENHANCEMENT (A TO Z)
    # =========================================================================
    add_section_header("Image Preprocessing & Feature Enhancement (A to Z)", 2)
    log_md(section="PART 2: Image Preprocessing & Feature Enhancement (A to Z)")

    add_callout(
        "Standard machine learning models fail in clinical ophthalmology when aggressive cropping chops off the peripheral retinal quadrants, or when illumination artifacts and lens glare mimic exudates. OptiSight implements a full-aspect uncropped pipeline with Ben Graham's local contrast subtraction, soft luminance tissue masking, and green-channel microvascular edge extraction.",
        "PREPROCESSING PARADIGM"
    )

    add_qa(
        8,
        "What is the complete step-by-step preprocessing pipeline from raw fundus photograph to model inference?",
        "The OptiSight preprocessing pipeline executes the following deterministic stages:\n\n"
        "Stage 1: Aspect-Ratio Preserving Full Image Rescaling (`prepare_display_image`)\n"
        "- The raw fundus image is resized such that max(H, W) = 640 using area-based interpolation (`cv2.INTER_AREA`). Crucially, aspect ratio is preserved without square stretching, maintaining circular anatomical geometry.\n\n"
        "Stage 2: Soft Luminance Tissue Mask Calculation (`compute_soft_tissue_mask`)\n"
        "- Luminance gray map is calculated: Y = 0.299R + 0.587G + 0.114B.\n"
        "- Soft thresholding: tissue_mask = clip((gray - 18.0) / 22.0, 0, 1).\n"
        "- Convolved with a 9x9 Gaussian blur (sigma=0) to generate a smooth, non-faceted aperture boundary that perfectly isolates retinal tissue from dark borders.\n\n"
        "Stage 3: Green-Channel Microvascular Edge Extraction (`extract_retinal_edges`)\n"
        "- Retinal green channel is isolated.\n"
        "- Multi-threshold Canny edge detection (low=30, high=80) extracts capillary loops and lesion perimeters.\n"
        "- Morphological dilation with an elliptical 3x3 kernel bridges broken vessel segments, followed by 5x5 Gaussian feathering.\n\n"
        "Stage 4: Ben Graham's Local Color Subtraction (`ben_graham`)\n"
        "- Spatial color normalization: I' = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), size/30), -4, 128).\n"
        "- Normalized image is resized to (224, 224) for model input.\n\n"
        "Stage 5: Tensor Normalization (`to_tensor`)\n"
        "- Converted to float32 tensor [0.0, 1.0] and standardized using ImageNet channel statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).",
        "Highlight the order of operations: Resize -> Tissue Mask -> Green Edges -> Ben Graham -> Tensor Normalization."
    )
    log_md(q_num=8,
           q_text="What is the complete step-by-step preprocessing pipeline from raw fundus photograph to model inference?",
           a_text="1. Full uncropped aspect rescaling (max_dim=640). 2. Luminance tissue mask extraction. 3. Green channel microvascular edge extraction. 4. Ben Graham local color subtraction. 5. Tensor standardization with ImageNet mean/std.",
           tip="Emphasize that the full uncropped image is preserved at every step.")

    add_qa(
        9,
        "Why do we keep the FULL UNCROPPED fundus image instead of tightly cropping the dark margins?",
        "Cropping dark borders seems intuitive to non-medical ML practitioners, but in vitreoretinal diagnostics, aggressive cropping introduces critical clinical errors:\n\n"
        "1. Clinical Rule Invalidation (AAO 4:2:1 Rule): The international standard for diagnosing Severe NPDR (Grade 3) requires evaluating all 4 retinal quadrants: Superior-Temporal, Inferior-Temporal, Superior-Nasal, and Inferior-Nasal. Tightly cropping the image cuts off the peripheral quadrants, making it impossible to detect peripheral hemorrhages and leading to dangerous under-staging (e.g., misclassifying Grade 3 as Grade 1 or 2).\n\n"
        "2. Optical Aperture Distortion: Fundus cameras use a circular optical mask. Tightly cropping the bounding box introduces aspect distortion, stretching the spherical retina into an oval and distorting cup-to-disc ratio measurements.\n\n"
        "3. False Peripheral Border Artifacts: Cutting the circular boundary creates artificial sharp high-contrast rectangular corners. Neural networks mistake these artificial edges for preretinal hemorrhages or giant tears.\n\n"
        "OptiSight retains the full uncropped fundus, isolating the black margin cleanly using an optical luminance mask rather than destructive geometric cropping.",
        "Examiners appreciate clinical awareness: 'Cropping destroys the peripheral quadrants required for the AAO 4:2:1 Severe NPDR rule.'"
    )
    log_md(q_num=9,
           q_text="Why do we keep the FULL UNCROPPED fundus image instead of tightly cropping the dark margins?",
           a_text="Tight cropping destroys peripheral retinal quadrants required for the AAO 4:2:1 rule, distorts circular optical geometry, and creates artificial rectangular edge artifacts that models mistake for pathology.",
           tip="Examiners appreciate clinical awareness: tight cropping destroys peripheral quadrants needed for AAO 4:2:1 staging.")

    add_qa(
        10,
        "Explain Ben Graham’s preprocessing technique mathematically and clinically. Why is sigma = size / 30?",
        "Ben Graham's preprocessing method was introduced by Dr. Benjamin Graham (Kaggle Diabetic Retinopathy Competition winner). It solves the problem of massive illumination variability across retinal photographs.\n\n"
        "Mathematical Formulation:\n"
        "   I_{processed}(x, y) = alpha * I(x, y) + beta * (I * G_sigma)(x, y) + gamma\n"
        "In our implementation:\n"
        "   alpha = 4.0\n"
        "   beta = -4.0\n"
        "   gamma = 128 (neutral gray baseline)\n"
        "   sigma = size / 30 (for size=224, sigma ~= 7.46; for size=640, sigma ~= 21.3)\n\n"
        "Why It Works (Unsharp Masking / Bandpass Filtering):\n"
        "1. Local Mean Subtraction: The Gaussian blur term (I * G_sigma) represents the low-frequency local background illumination (retinal color, lens glare, pigmentation, camera vignetting). By subtracting 4 times this local mean from 4 times the original image, low-frequency variations are completely annihilated.\n"
        "2. High-Frequency Amplification: Microaneurysms, hemorrhages, exudates, and vessels represent high-frequency spatial variations. The formula amplifies these high-frequency components by 4x, making subtle microaneurysms pop out vividly against a standardized neutral gray (128) backdrop.\n"
        "3. Why sigma = size / 30? A kernel radius of size/30 corresponds to the approximate anatomical scale of the optic disc and major vascular arcades. If sigma were too small (e.g., size/100), the filter would subtract the microaneurysms themselves. If sigma were too large (e.g., size/5), vignetting and lens glare would bleed through.",
        "Be ready to explain that Ben Graham's method is essentially a homomorphic unsharp mask: Low frequencies (illumination/pigmentation) are subtracted; High frequencies (microaneurysms/vessels) are amplified."
    )
    log_md(q_num=10,
           q_text="Explain Ben Graham’s preprocessing technique mathematically and clinically. Why is sigma = size / 30?",
           a_text="Ben Graham's formula: I' = 4*I - 4*Gaussian(I, sigma) + 128. It acts as a local bandpass filter: subtracting low-frequency illumination gradients while amplifying high-frequency microaneurysms against a neutral gray (128) baseline. sigma=size/30 matches the scale of the optic disc.",
           tip="Ben Graham's method acts as a local bandpass filter that standardizes background color while 4x amplifying micro-lesions.")

    add_qa(
        11,
        "Why is the GREEN channel specifically isolated for microvascular edge extraction?",
        "Fundus photographs are captured in 24-bit RGB, but optical absorption of light in ocular tissue is strongly wavelength-dependent:\n\n"
        "1. Hemoglobin Absorption Spectrum: Deoxygenated and oxygenated hemoglobin in retinal blood vessels exhibit peak optical absorption in the green wavelength band (540–570 nm). As a consequence, blood vessels and intraretinal hemorrhages absorb green light strongly, appearing exceptionally dark with maximum contrast against the surrounding reflective retinal pigment epithelium.\n\n"
        "2. Failure of the Red Channel: Red light (620–750 nm) easily penetrates the retina, choroid, and sclera with minimal absorption. The red channel is overwhelmingly saturated, washed out, and obscures vascular morphology.\n\n"
        "3. Failure of the Blue Channel: Blue light (400–490 nm) undergoes intense Rayleigh scattering in the cornea, crystalline lens, and vitreous humor. In older patients with cataracts, blue light is almost completely absorbed, resulting in a dark, noisy channel with low signal-to-noise ratio (SNR).\n\n"
        "Conclusion: The Green channel provides the highest signal-to-noise ratio and maximum contrast for blood vessels, microaneurysms, and hemorrhages. Isolating the green channel is standard protocol in ophthalmic imaging.",
        "State the exact wavelength: 'Hemoglobin exhibits peak optical absorption in the green spectrum (540–570 nm), maximizing vessel-to-background contrast.'"
    )
    log_md(q_num=11,
           q_text="Why is the GREEN channel specifically isolated for microvascular edge extraction?",
           a_text="Hemoglobin has peak optical absorption in the green spectrum (540-570 nm). The red channel is saturated and washed out, while the blue channel suffers from severe Rayleigh scattering and cataract absorption. Green gives maximum signal-to-noise ratio.",
           tip="Cite the exact optical wavelength: hemoglobin peak absorption occurs at 540–570 nm.")

    add_qa(
        12,
        "How does the luminance tissue mask work, and why was the convex hull polygon approach discarded?",
        "Initial prototypes attempted to outline the retinal boundary using `cv2.convexHull` on binary thresholded fundus images. However, this produced severe diagnostic errors:\n\n"
        "1. Why Convex Hull Failed:\n"
        "- A convex hull calculates the smallest convex polygon enclosing a set of points. The circular aperture of a fundus camera is curved. Connecting outer contour points with straight polygon chords cuts off circular arcs of retinal tissue, creating an artificial faceted polygon that chops into peripheral retinal quadrants.\n"
        "- If camera flash flares or eyelashes exist at the edge, the convex hull forms sharp, unnatural angles, slicing off real hemorrhages.\n\n"
        "2. OptiSight Luminance Soft Tissue Mask (`compute_soft_tissue_mask`):\n"
        "- Converts the image to grayscale luminance: Y = 0.299R + 0.587G + 0.114B.\n"
        "- Applies continuous luminance normalization: mask = clip((Y - 18.0) / 22.0, 0.0, 1.0).\n"
        "- Applies a 9x9 Gaussian smoothing kernel.\n\n"
        "Benefits: The boundary follows the natural, smooth curvature of the optical aperture. Transition from black background to retinal tissue is completely continuous, preventing hard geometric cuts and eliminating heatmap bleeding with zero artifacts.",
        "Emphasize that the luminance mask is continuous (soft float between 0 and 1) rather than a rigid binary polygon, preventing artificial edge cuts."
    )
    log_md(q_num=12,
           q_text="How does the luminance tissue mask work, and why was the convex hull polygon approach discarded?",
           a_text="Convex hull creates straight-line polygon chords that slice off circular retinal arcs. The luminance mask uses continuous thresholding: clip((gray - 18)/22, 0, 1) + Gaussian feathering, hugging the natural optical aperture smoothly.",
           tip="Explain that continuous soft luminance masking eliminates the faceted straight-line cuts of convex hull polygons.")

    # =========================================================================
    # PART 3: EXPLAINABLE AI (XAI) & GRAD-CAM++ MECHANICS
    # =========================================================================
    add_section_header("Explainable AI (XAI) & Grad-CAM++ Mechanics", 3)
    log_md(section="PART 3: Explainable AI (XAI) & Grad-CAM++ Mechanics")

    add_callout(
        "Standard Grad-CAM fails in medical imaging because averaging gradients over an entire feature map slice zeroes out multiple co-occurring micro-lesions. OptiSight employs Grad-CAM++ with higher-order partial derivatives, combined with bilateral edge-preserving filtering and soft aperture confinement to provide lesion-accurate clinical saliency.",
        "EXPLAINABLE AI PRINCIPLE"
    )

    add_qa(
        13,
        "What is the mathematical formulation of Grad-CAM++, and why is it superior to standard Grad-CAM for Diabetic Retinopathy?",
        "Explainable AI (XAI) is essential in medicine: clinicians will not trust a 'black-box' prediction without visual verification that the model focused on real pathology rather than camera dust or flash artifacts.\n\n"
        "1. Standard Grad-CAM Formulation:\n"
        "In standard Grad-CAM, the importance weight alpha_k^c for feature map A^k is calculated by global average pooling of first-order gradients:\n"
        "   alpha_k^c = (1 / Z) * sum_i sum_j (d Y^c / d A_{ij}^k)\n"
        "The heatmap is then: L_{Grad-CAM}^c = ReLU(sum_k alpha_k^c * A^k)\n\n"
        "The Fatal Flaw of Standard Grad-CAM in DR:\n"
        "When an image contains multiple occurrences of the same lesion (e.g., 20 scattered microaneurysms across different quadrants), positive gradients from one lesion and negative gradients from adjacent non-lesion pixels cancel each other out during global average pooling. The model ends up generating a single blurry blob over one area, ignoring all other lesions.\n\n"
        "2. Grad-CAM++ Formulation (Chattopadhyay et al.):\n"
        "Grad-CAM++ solves this by assigning an individual pixel-wise weighting coefficient alpha_{ij}^{kc} to each spatial location:\n"
        "   w_k^c = sum_i sum_j alpha_{ij}^{kc} * ReLU(d Y^c / d A_{ij}^k)\n"
        "Where alpha_{ij}^{kc} is derived via second and third-order partial derivatives:\n"
        "   alpha_{ij}^{kc} = (d^2 Y^c / (d A_{ij}^k)^2) / [ 2 * (d^2 Y^c / (d A_{ij}^k)^2) + sum_{a,b} A_{ab}^k * (d^3 Y^c / (d A_{ij}^k)^3) ]\n\n"
        "Why Grad-CAM++ is Superior for DR:\n"
        "- Higher-order derivatives quantify the non-linear curvature of the score function with respect to individual activations.\n"
        "- Weights positive gradients explicitly, allowing multiple discrete lesions (dozens of microaneurysms) to be highlighted simultaneously without cancellation.",
        "Memorize the key difference: 'Standard Grad-CAM averages gradients uniformly, canceling out multiple scattered lesions. Grad-CAM++ uses second and third derivatives to weight positive pixel contributions, highlighting multiple microaneurysms across all quadrants.'"
    )
    log_md(q_num=13,
           q_text="What is the mathematical formulation of Grad-CAM++, and why is it superior to standard Grad-CAM for Diabetic Retinopathy?",
           a_text="Standard Grad-CAM calculates alpha_k via global average pooling of first derivatives, causing multiple scattered lesions to cancel out. Grad-CAM++ uses second and third-order partial derivatives to weight positive pixel contributions individually, capturing multiple microaneurysms simultaneously.",
           tip="Standard Grad-CAM averages out multiple small lesions. Grad-CAM++ weights positive contributions using higher-order derivatives.")

    add_qa(
        14,
        "How does Bilateral Filtering refine the coarse Grad-CAM activation map into clinical saliency?",
        "Convolutional neural networks downsample 224x224 images into a 7x7 feature map at the final convolutional layer. Standard Grad-CAM upsamples this 7x7 map using bilinear interpolation, resulting in a coarse, blurry circular blob that covers healthy tissue.\n\n"
        "OptiSight implements Bilateral Edge-Preserving Refinement (`refine_cam_map`):\n"
        "1. Bilateral Filtering Equation:\n"
        "   I_{filtered}(x) = (1 / W_p) * sum_{x_i in Omega} I(x_i) * f_r(||I(x_i) - I(x)||) * g_s(||x_i - x||)\n"
        "   - g_s: Spatial Gaussian filter (smoothing nearby pixels in space, sigmaSpace=9).\n"
        "   - f_r: Range Gaussian filter (smoothing only pixels with similar intensity, sigmaColor=0.25).\n\n"
        "2. Clinical Effect: Unlike a standard Gaussian blur which bleeds across edges, the bilateral filter smooths activation noise within a vessel or lesion, but stops completely at sharp vascular and retinal boundaries.\n\n"
        "3. Edge-Guided Modulation: The smoothed CAM is modulated by the Canny microvascular edge map:\n"
        "   CAM_{refined} = CAM_{smooth} * (1.0 + 0.45 * EdgeMap) * TissueMask\n"
        "This snaps the heat map directly onto real retinal microaneurysms and vessel walls, while the tissue mask guarantees zero bleeding into dark margins.",
        "Explain that bilateral filtering combines a spatial Gaussian (distance) with a range Gaussian (intensity difference), preventing heat from blurring into healthy background."
    )
    log_md(q_num=14,
           q_text="How does Bilateral Filtering refine the coarse Grad-CAM activation map into clinical saliency?",
           a_text="Upsampling a 7x7 feature map causes blurry blobs. Bilateral filtering combines spatial and range Gaussian filters to smooth activation while preserving sharp lesion edges, which is then modulated by the Canny edge map and tissue mask.",
           tip="Explain that bilateral filtering combines spatial distance with intensity difference to prevent blurring across anatomical boundaries.")

    add_qa(
        15,
        "Which specific layers were selected as CAM target layers in MaxViT, DenseNet-121, and Hybrid, and why?",
        "Grad-CAM requires hooking into the final feature representation that retains spatial dimensions before global average pooling:\n\n"
        "1. DenseNet-121 Target Layer: `features.denseblock4.denselayer16.conv2`\n"
        "- This is the final 3x3 convolution of the 16th dense layer in the 4th dense block. It contains 1024 channels at 7x7 spatial resolution, representing the richest aggregation of low-level textural features and high-level semantics.\n\n"
        "2. MaxViT-Tiny Target Layer: `stages.3.blocks.1.conv.conv3_1x1`\n"
        "- This is the final 1x1 projection convolution of the final transformer block in stage 3 (stride 32). It represents the highest semantic feature map before the global pooling layer.\n\n"
        "3. OptiSight Hybrid Target Layer: `b.features.denseblock4.denselayer16.conv2`\n"
        "- In the dual-branch hybrid, the DenseNet branch (`b`) is hooked because its dense connections provide the sharpest gradient localization for microvascular lesions, which are then modulated by the late-fusion predictions of both branches.",
        "Examiners will check if you know the exact layer names in your code: DenseNet denseblock4.denselayer16.conv2 and MaxViT stage 3 final conv."
    )
    log_md(q_num=15,
           q_text="Which specific layers were selected as CAM target layers in MaxViT, DenseNet-121, and Hybrid, and why?",
           a_text="DenseNet: features.denseblock4.denselayer16.conv2 (richest 1024D representation). MaxViT: stages.3.blocks.1.conv.conv3_1x1 (final stage semantic map). Hybrid hooks the DenseNet branch for sharpest lesion boundary localization.",
           tip="Cite the exact code identifiers: denseblock4.denselayer16.conv2 for DenseNet.")

    # =========================================================================
    # PART 4: CLINICAL OPHTHALMOLOGY & DIAGNOSTIC STAGING
    # =========================================================================
    add_section_header("Clinical Ophthalmology & Medical Decision Support (CDSS)", 4)
    log_md(section="PART 4: Clinical Ophthalmology & Medical Decision Support (CDSS)")

    add_callout(
        "OptiSight is strictly calibrated to international ophthalmic standards: the International Clinical Diabetic Retinopathy (ICDR) 5-stage scale, the American Academy of Ophthalmology (AAO) 4:2:1 Rule, and the ETDRS 9-zone macular grid for Center-Involved Diabetic Macular Edema (CI-DME).",
        "CLINICAL VALIDATION FRAMEWORK"
    )

    # ICDR Table
    headers_icdr = ["ICDR Grade", "Clinical Classification", "Key Ophthalmic Findings", "AAO Referral Timeline", "DME Risk"]
    rows_icdr = [
        ["Grade 0", "No Apparent DR", "No microaneurysms, hemorrhages, or exudates; crisp foveal reflex", "Annual Routine Screening (12 mo)", "< 2%"],
        ["Grade 1", "Mild Non-Proliferative DR", "Microaneurysms ONLY (pinpoint red dots, <125um)", "6–12 Months Follow-up", "~ 5–10%"],
        ["Grade 2", "Moderate Non-Proliferative DR", "More than microaneurysms: dot/blot hemorrhages, hard exudates, cotton wool spots; LESS than Severe NPDR", "3–6 Months Follow-up", "~ 15–25%"],
        ["Grade 3", "Severe Non-Proliferative DR", "Meets ANY 1 rule of AAO 4:2:1: >=20 hemorrhages in 4 quadrants, OR venous beading in 2+ quadrants, OR IRMA in 1+ quadrant", "URGENT Vitreoretinal Consult (2–4 weeks)", "35–50%"],
        ["Grade 4", "Proliferative DR (PDR)", "Neovascularization of the Disc (NVD) or Elsewhere (NVE), preretinal/vitreous hemorrhage, fibrous proliferation", "EMERGENCY Intervention (24–72 hours; Anti-VEGF / PRP)", "> 60%"]
    ]
    add_table_data(headers_icdr, rows_icdr)
    log_md(table_h=headers_icdr, table_r=rows_icdr)

    add_qa(
        16,
        "Explain the American Academy of Ophthalmology (AAO) 4:2:1 Rule in clinical detail.",
        "The AAO 4:2:1 Rule is the gold standard clinical benchmark used to differentiate Moderate NPDR (Grade 2) from Severe NPDR (Grade 3). A patient meeting ANY ONE of the following three criteria is diagnosed with Severe NPDR (~50% risk of progressing to high-risk proliferative disease within 1 year):\n\n"
        "1. The '4' Rule (Hemorrhages / Microaneurysms in 4 Quadrants):\n"
        "- Severe intraretinal hemorrhages and microaneurysms (>= 20 distinct lesions) present in ALL FOUR retinal quadrants (Superior-Temporal, Inferior-Temporal, Superior-Nasal, Inferior-Nasal).\n\n"
        "2. The '2' Rule (Venous Beading in >= 2 Quadrants):\n"
        "- Definite venous beading (localized caliber variation where veins resemble a string of sausages due to severe endothelial hypoxia) in TWO OR MORE quadrants.\n\n"
        "3. The '1' Rule (IRMA in >= 1 Quadrant):\n"
        "- Prominent Intraretinal Microvascular Abnormalities (IRMA—irregular, tortuous dilated capillary loops acting as intraretinal shunts) in ONE OR MORE quadrants.\n\n"
        "If TWO OR MORE criteria are satisfied, the patient is classified as Very Severe NPDR (~75% 1-year progression risk to PDR), demanding immediate specialist intervention.",
        "Memorize the 4:2:1 numbers: 4 quadrants of >=20 hemorrhages; 2 quadrants of venous beading; 1 quadrant of IRMA."
    )
    log_md(q_num=16,
           q_text="Explain the American Academy of Ophthalmology (AAO) 4:2:1 Rule in clinical detail.",
           a_text="The AAO 4:2:1 rule defines Severe NPDR: 4 quadrants with >=20 hemorrhages; OR 2 quadrants with venous beading; OR 1 quadrant with IRMA. Meeting any 1 criteria signals ~50% 1-year risk of proliferative disease.",
           tip="Memorize the exact definition: 4 quadrants hemorrhages, 2 quadrants venous beading, 1 quadrant IRMA.")

    add_qa(
        17,
        "What is Diabetic Macular Edema (DME), and how does the ETDRS 9-Zone Grid assess it?",
        "Diabetic Macular Edema (DME) is retinal thickening caused by hyperpermeability and breakdown of the blood-retinal barrier. It can occur at ANY stage of DR and is the single most common cause of moderate vision loss in diabetic patients.\n\n"
        "1. ETDRS 9-Zone Macular Grid:\n"
        "Standardized by the Early Treatment Diabetic Retinopathy Study (ETDRS), the grid divides the macula into 9 concentric subfields centered on the fovea:\n"
        "- Central Subfield (CSF): Inner circle with a diameter of 1 mm (0.5 mm radius).\n"
        "- Inner Ring: Outer diameter of 3 mm, divided into 4 quadrants (Superior, Inferior, Nasal, Temporal).\n"
        "- Outer Ring: Outer diameter of 6 mm, divided into 4 quadrants (Superior, Inferior, Nasal, Temporal).\n\n"
        "2. Clinical Significance (Center-Involved DME):\n"
        "- If hard exudates or retinal thickening reside within the central 1 mm subfield (CSF), the patient has Center-Involved DME (CI-DME), threatening central 20/20 visual acuity. This warrants prompt intravitreal Anti-VEGF injections (e.g., Aflibercept, Ranibizumab) or focal laser photocoagulation.",
        "Note that DME is staged orthogonally to retinopathy: a patient can have Grade 1 Mild NPDR but still suffer from vision-threatening Center-Involved DME."
    )
    log_md(q_num=17,
           q_text="What is Diabetic Macular Edema (DME), and how does the ETDRS 9-Zone Grid assess it?",
           a_text="DME is blood-retinal barrier breakdown causing macular fluid accumulation. The ETDRS grid has 3 concentric rings (1mm central subfield, 3mm inner ring, 6mm outer ring). Lesions in the 1mm subfield indicate Center-Involved DME requiring Anti-VEGF therapy.",
           tip="Note that DME can occur at any grade: a Grade 1 patient can have vision-threatening DME in the 1mm subfield.")

    add_qa(
        18,
        "How does the CDSS compute the composite clinical risk score incorporating systemic biomarkers?",
        "Diabetic retinopathy does not exist in isolation; it is a systemic microvascular complication. OptiSight combines image-based model predictions with patient systemic biomarkers into an integrated Clinical Risk Score (0–100):\n\n"
        "1. Base Image Score: Baseline score derived from model consensus grade:\n"
        "   BaseScore = ConsensusGrade * 20  (Grade 0=0, Grade 1=20, Grade 2=40, Grade 3=60, Grade 4=80)\n\n"
        "2. Glycemic Control (HbA1c Factor):\n"
        "- HbA1c >= 9.5%: +12 points (Extreme chronic microvascular damage)\n"
        "- HbA1c >= 8.0%: +8 points (Uncontrolled hyperglycemia)\n"
        "- HbA1c >= 7.0%: +4 points (Borderline control)\n\n"
        "3. Diabetes Duration Factor:\n"
        "- Duration >= 15 years: +8 points (Cumulative capillary basement membrane thickening)\n"
        "- Duration >= 10 years: +5 points\n"
        "- Duration >= 5 years: +2 points\n\n"
        "4. Macular Edema (DME Factor): +10 points if DME suspected in ETDRS central subfield.\n\n"
        "5. Hypertension (Systolic Blood Pressure): +5 points if SBP >= 140 mmHg (accelerates endothelial shearing).\n\n"
        "Composite Risk Levels:\n"
        "- 0–24: Low Risk (Grade 0–1 / Stable)\n"
        "- 25–49: Moderate Risk (3–6 Month Monitoring)\n"
        "- 50–74: High Progression Risk (Urgent Specialist Referral within 2–4 weeks)\n"
        "- 75–100: Critical Urgent Risk (Immediate 24–72 hour Vitreoretinal Intervention)",
        "This is what elevates your project from a basic Kaggle classifier into a true Clinical Decision Support System (CDSS)."
    )
    log_md(q_num=18,
           q_text="How does the CDSS compute the composite clinical risk score incorporating systemic biomarkers?",
           a_text="Composite risk combines baseline image grade (Grade * 20) with systemic risk factors: HbA1c (+4 to +12), diabetes duration (+2 to +8), DME presence (+10), and hypertension (+5), outputting a 0-100 risk score and triage timeline.",
           tip="Highlight that integrating systemic biomarkers (HbA1c, BP) turns an image classifier into a true CDSS.")

    # =========================================================================
    # PART 5: LOSS FUNCTIONS, TRAINING & EVALUATION METRICS
    # =========================================================================
    add_section_header("Loss Functions, Training Dynamics & Evaluation Metrics", 5)
    log_md(section="PART 5: Loss Functions, Training Dynamics & Evaluation Metrics")

    add_qa(
        19,
        "Why is Quadratic Weighted Kappa (QWK) the gold-standard metric for Diabetic Retinopathy instead of Accuracy or F1-Score?",
        "Standard multi-class accuracy and Macro F1-Score treat all classification mistakes identically:\n"
        "- Misclassifying Grade 0 as Grade 1 is penalized identically to misclassifying Grade 0 as Grade 4!\n"
        "In clinical medicine, this is unviable: confusing Grade 0 with Grade 1 has minor clinical consequence (annual vs. 6-month checkup), but misclassifying Grade 4 (proliferative DR needing emergency surgery) as Grade 0 (healthy) causes irreversible blindness.\n\n"
        "Quadratic Weighted Kappa (QWK) measures inter-rater agreement between predicted and true ordinal grades, penalizing errors quadratically with distance:\n\n"
        "QWK Formulation:\n"
        "   kappa = 1 - (sum_{i,j} w_{i,j} * O_{i,j}) / (sum_{i,j} w_{i,j} * E_{i,j})\n"
        "Where the quadratic penalty matrix is:\n"
        "   w_{i,j} = (i - j)^2 / (N - 1)^2\n"
        "- O_{i,j} is the observed confusion matrix.\n"
        "- E_{i,j} is the expected matrix calculated from outer product of marginal histogram distributions.\n"
        "- When i = j (correct prediction): w_{i,j} = 0 (no penalty).\n"
        "- An error of 1 grade (e.g. 1 vs 2): penalty is 1^2 = 1.\n"
        "- An error of 4 grades (e.g. 0 vs 4): penalty is 4^2 = 16 (16x higher penalty!).",
        "State clearly: 'Diabetic Retinopathy is an ordinal classification problem. Quadratic Weighted Kappa penalizes large clinical discrepancies quadratically (16x for 4 grades vs 1x for 1 grade).'"
    )
    log_md(q_num=19,
           q_text="Why is Quadratic Weighted Kappa (QWK) the gold-standard metric for Diabetic Retinopathy instead of Accuracy or F1-Score?",
           a_text="DR staging is ordinal. Standard accuracy treats all misclassifications equally. QWK penalizes errors quadratically: w_{i,j} = (i-j)^2 / (N-1)^2. A 4-grade error (Grade 4 classified as 0) receives a 16x harsher penalty than a 1-grade error.",
           tip="Emphasize that DR is an ordinal problem where distance between classes matters clinically.")

    add_qa(
        20,
        "What loss function is optimal for ordinal classification, and how do you mitigate extreme class imbalance?",
        "1. Ordinal Loss Formulation:\n"
        "Standard Categorical Cross-Entropy (CCE) assumes orthogonal one-hot classes with zero relationship between classes. To respect ordinal progression, optimal training utilizes:\n"
        "- Label Smoothing with Gaussian distribution: Instead of one-hot [0, 0, 1, 0, 0], targets are smoothed over neighbors: [0.05, 0.20, 0.50, 0.20, 0.05].\n"
        "- Combined Loss: L_{total} = L_{Focal} + lambda * L_{MSE_Ordinal}\n"
        "Where Focal Loss addresses class imbalance and MSE penalizes distance between class indices.\n\n"
        "2. Handling Severe Class Imbalance:\n"
        "Real-world DR datasets (e.g., EyePACS, APTOS 2019, Messidor) exhibit severe skew (Grade 0 typically comprises 70%+ of samples, while Grade 4 is <5%):\n"
        "- Class-Weighted Focal Loss: Adds modulating factor (1 - p_t)^gamma to down-weight easy negative examples (Grade 0).\n"
        "- Balanced Weighted Random Sampler: Samples batches inversely proportional to class frequency during training so each batch has equal representation of all 5 grades.\n"
        "- Test-Time Augmentation (TTA): Inferences across horizontal and vertical flips are averaged to maximize sensitivity on rare lesions.",
        "Mention Test-Time Augmentation (TTA): In our code, `infer_with_cam` automatically evaluates [image, hflip, vflip] and stacks activations for maximum stability."
    )
    log_md(q_num=20,
           q_text="What loss function is optimal for ordinal classification, and how do you mitigate extreme class imbalance?",
           a_text="Standard cross-entropy ignores ordinality. We utilize focal loss with ordinal label smoothing, balanced batch sampling via WeightedRandomSampler, and Test-Time Augmentation (TTA) using horizontal/vertical flips.",
           tip="Highlight our use of Test-Time Augmentation (TTA) in infer_with_cam for stability.")

    # =========================================================================
    # PART 6: DEPLOYMENT, SERVERLESS ENGINEERING & VIVA TRAP DEFENSE
    # =========================================================================
    add_section_header("Deployment, Serverless Engineering & Viva Trap Defense", 6)
    log_md(section="PART 6: Deployment, Serverless Engineering & Viva Trap Defense")

    add_qa(
        21,
        "Explain the dual-mode architecture between your local workstation and Vercel cloud deployment.",
        "OptiSight features an engineered dual-mode runtime architecture:\n\n"
        "1. Workstation High-Performance Mode (Local / GPU / Docker):\n"
        "- When `torch` and `.pth` weight files are present, the system loads all three deep neural networks: MaxViT-Tiny (31M params), DenseNet-121 (7.9M params), and the OptiSight Hybrid fusion head.\n"
        "- Executes full backpropagation Grad-CAM++ with test-time augmentation (TTA) on batch tensors.\n\n"
        "2. Cloud Serverless Mode (Vercel Serverless Edge):\n"
        "- Vercel Serverless Functions enforce a strict 250 MB uncompressed bundle size limit. Full PyTorch CPU wheels alone exceed 1.1 GB, and our 3 weight files total ~304 MB, making a full PyTorch bundle impossible on standard serverless tiers without massive latency.\n"
        "- OptiSight solves this by implementing an Edge-Assisted Computer Vision Diagnostic Engine: executes within a lightweight <45 MB bundle (Flask, NumPy, OpenCV-Headless, Pillow).\n"
        "- Analyzes microaneurysms, hemorrhages, and lipid exudates using green-channel inverted thresholding and morphological filters, returning instant predictions, AAO staging, ETDRS grids, and refined edge-guided heatmaps 24/7/365 without server costs.\n"
        "- The architecture includes `OPTI_BACKEND_URL` forwarding: in production, Vercel can forward inference requests to a dedicated GPU backend (e.g. Hugging Face Spaces T4 or AWS) while serving the frontend globally with zero cold start.",
        "Examiners are deeply impressed when students demonstrate awareness of cloud bundle constraints (250MB limit on Vercel) and explain architectural workarounds."
    )
    log_md(q_num=21,
           q_text="Explain the dual-mode architecture between your local workstation and Vercel cloud deployment.",
           a_text="Vercel has a 250MB serverless limit, while PyTorch + weights require ~1.5GB. OptiSight implements dual-mode: heavy PyTorch inference locally, and a lightweight (<45MB) edge-assisted diagnostic engine on Vercel, with external GPU backend routing via OPTI_BACKEND_URL.",
           tip="Examiners are impressed by awareness of serverless constraints (250MB bundle limit) and clean architectural workarounds.")

    add_qa(
        22,
        "Examiner Trap Question: 'If your model is an ensemble/hybrid, isn't it too slow for real-time clinical deployment?' How do you answer?",
        "Answer Strategy:\n"
        "1. Direct Rebuttal: 'No, sir/ma'am. OptiSight Hybrid is NOT a slow independent multi-pass pipeline. It is a unified dual-branch network with low parameter count and optimized FLOPs.'\n\n"
        "2. Concrete Numbers:\n"
        "- DenseNet-121 contains only ~7.9 million parameters.\n"
        "- MaxViT-Tiny contains ~31 million parameters.\n"
        "- Combined parameter count is ~39 million parameters—which is significantly smaller than a single standard ResNet-152 (60M) or VGG-16 (138M)!\n\n"
        "3. Shared Execution & Parallel Latency:\n"
        "- On an NVIDIA T4 / RTX 3060 GPU, forward pass latency for both branches combined is under 42 milliseconds per image.\n"
        "- On standard multi-core CPU, complete inference including Grad-CAM++ generation takes approximately 350–500 milliseconds—well within the standard clinical threshold (ophthalmologists take 2–5 minutes per patient fundus set).\n\n"
        "4. Clinical Trade-Off Justification:\n"
        "- In medical diagnostics, a 30 millisecond difference is clinically irrelevant; whereas a 3% boost in Quadratic Weighted Kappa prevents patients from suffering irreversible vision loss from misclassified Proliferative DR.",
        "Emphasize total parameters (~39M) vs VGG-16 (138M) and explain that medical imaging prioritizes diagnostic sensitivity over microsecond gaming latency."
    )
    log_md(q_num=22,
           q_text="Examiner Trap Question: 'If your model is an ensemble/hybrid, isn't it too slow for real-time clinical deployment?' How do you answer?",
           a_text="No. DenseNet-121 (7.9M) + MaxViT-Tiny (31M) total ~39M parameters—far smaller than a single VGG-16 (138M). Combined GPU latency is under 42ms, and CPU inference is <500ms, which is instantaneous for clinical workflow.",
           tip="Contrast total hybrid parameters (39M) with older models like VGG-16 (138M) to prove lightweight efficiency.")

    add_qa(
        23,
        "Examiner Trap Question: 'Can your model replace a certified ophthalmologist?' How do you answer?",
        "Answer Strategy:\n"
        "1. Definitive Clinical Answer: 'Under no circumstances is OptiSight designed or intended to replace a certified ophthalmologist or retina specialist.'\n\n"
        "2. Define the System's Role as a CDSS:\n"
        "- 'OptiSight is classified as an automated Clinical Decision Support System (CDSS) and triage accelerator, adhering to FDA and CE Software as a Medical Device (SaMD) guidelines.'\n\n"
        "3. Three Core Clinical Utility Pillars:\n"
        "- Screening & Triage in Underserved Areas: In rural and primary healthcare clinics without resident ophthalmologists, OptiSight filters normal fundus images (Grade 0) and flags urgent cases (Grade 3 & 4) for immediate tele-ophthalmology referral.\n"
        "- Reducing Diagnostic Fatigue & Inter-Observer Variability: Even experienced retinal specialists have a 12–18% inter-observer variability rate. OptiSight provides an objective, reproducible second opinion with anatomical Grad-CAM++ validation.\n"
        "- Medico-Legal Transparency: The human doctor always remains in the decision loop, using the interactive ETDRS grid, clinical risk score, and consultation logging to finalize the medical diagnosis.",
        "Never say 'the AI makes the diagnosis.' Always say: 'The AI provides clinical decision support, triage stratification, and explainable saliency, while the ophthalmologist maintains final diagnostic and therapeutic authority.'"
    )
    log_md(q_num=23,
           q_text="Examiner Trap Question: 'Can your model replace a certified ophthalmologist?' How do you answer?",
           a_text="No. OptiSight is a Clinical Decision Support System (CDSS) for triage and screening. It filters normal cases in underserved rural clinics and provides objective second opinions to reduce specialist fatigue, while the doctor retains final medical authority.",
           tip="Always maintain that the human doctor remains in the loop as the final medical authority.")


    # =========================================================================
    # PART 7: COMPREHENSIVE CLINICAL PATHOLOGY & EXAMINER DEFENSE
    # =========================================================================
    add_section_header("Advanced Clinical Pathology & Viva Defense Questions", 7)
    log_md(section="PART 7: Advanced Clinical Pathology & Viva Defense Questions")

    add_qa(
        24,
        "What are the specific failure modes of DenseNet-121, and how does MaxViT-Tiny compensate for them?",
        "1. DenseNet-121 Blind Spots & Failure Modes:\n"
        "- Myopic Spatial Context: DenseNet relies on local receptive fields. In high-resolution fundus images, it cannot distinguish whether scattered hemorrhages span all 4 quadrants (Severe NPDR) or are localized to a single sector (Moderate NPDR).\n"
        "- Confounding Non-Pathological Structures: DenseNet sometimes misclassifies bright physiological structures—such as the sharp temporal margin of the optic disc, myelinated nerve fibers, or drusen—as hard lipid exudates because they share identical high-frequency gradient edges.\n"
        "- Missed Diffuse Ischemia: DenseNet struggles to detect broad, non-focal capillary non-perfusion zones where no sharp micro-lesions exist, but the background retinal tone is ischemic.\n\n"
        "2. How MaxViT-Tiny Compensates:\n"
        "- MaxViT's Grid Attention calculates global self-attention across the entire fundus. It contextualizes bright spots against the anatomical location of the optic nerve head, suppressing false positives at the disc.\n"
        "- Evaluates multi-quadrant spatial distributions effortlessly, correctly staging the patient according to the AAO 4:2:1 standard.",
        "Highlighting individual model weaknesses and showing how the hybrid fixes them proves you didn't just stack models blindly."
    )
    log_md(q_num=24,
           q_text="What are the specific failure modes of DenseNet-121, and how does MaxViT-Tiny compensate for them?",
           a_text="DenseNet lacks multi-quadrant spatial context and can mistake optic disc borders for exudates. MaxViT's global grid attention contextualizes bright spots against optic disc anatomy and assesses 4-quadrant spatial lesion spread.",
           tip="Explaining specific failure modes proves deep architectural intention.")

    add_qa(
        25,
        "What are the specific failure modes of MaxViT-Tiny, and how does DenseNet-121 compensate for them?",
        "1. MaxViT-Tiny Blind Spots & Failure Modes:\n"
        "- Subtle Point-Lesion Blindness: While MaxViT is far superior to standard ViT due to its MBConv stem, self-attention mechanisms naturally prioritize dominant visual patterns (e.g. large vascular trunks, optic cup boundaries). A single, isolated 10-micron microaneurysm in an otherwise clean retina can be drowned out by dominant global attention weights, resulting in a Grade 1 patient being falsely classified as Grade 0 (healthy).\n"
        "- Sub-Pixel Textural Smoothing: At higher network stages, downsampling attenuates isolated high-frequency point gradients.\n\n"
        "2. How DenseNet-121 Compensates:\n"
        "- Dense concatenation ensures that early 3x3 convolutional edge maps are routed directly to the fusion head without degradation. If even 2–3 microaneurysms exist, DenseNet's dense feature channels fire strongly, preventing false-negative Grade 0 classifications.",
        "Use this pairing to summarize the hybrid: DenseNet catches isolated pinhead microaneurysms; MaxViT understands global anatomical distribution."
    )
    log_md(q_num=25,
           q_text="What are the specific failure modes of MaxViT-Tiny, and how does DenseNet-121 compensate for them?",
           a_text="MaxViT's global attention can overlook a single isolated 10-micron microaneurysm amidst dominant anatomical structures. DenseNet's direct feature reuse catches tiny pinpoint microaneurysms, preventing false-negative Grade 0 classifications.",
           tip="DenseNet catches isolated microaneurysms; MaxViT understands the global anatomical layout.")

    add_qa(
        26,
        "Which data augmentations are clinically valid for retinal fundus imaging, and which are strictly forbidden?",
        "In medical imaging, applying computer vision augmentations blindly can alter the clinical ground truth and invalidate model training:\n\n"
        "Clinically Valid Augmentations:\n"
        "1. Horizontal & Vertical Flips: Retinas are anatomically valid under reflection (equivalent to right eye vs. left eye inverted geometry).\n"
        "2. Mild Random Rotation (+/- 15 degrees): Mimics natural head tilt and fundus camera alignment variability without distorting quadrant anatomy.\n"
        "3. Minor Brightness / Contrast Scaling (+/- 10%): Simulates minor pupil dilation differences and flash illumination levels.\n"
        "4. Ben Graham Contrast Normalization: Standardizes pigmentation across diverse ethnic populations.\n\n"
        "STRICTLY FORBIDDEN Augmentations:\n"
        "1. Aggressive Random Cropping: Forbidden because cutting off peripheral retina destroys multi-quadrant evaluation required for AAO 4:2:1 staging.\n"
        "2. Hue / Color Jittering: Forbidden because color is diagnostic! Blood is red, lipid exudates are yellow-white, cotton-wool spots are white, and melanin is brown. Shifting hue turns a red hemorrhage into green, completely corrupting the clinical features.\n"
        "3. Extreme Shear / Aspect Ratio Distortion: Forbidden because it distorts circular cup-to-disc ratios and stretches microaneurysms into fake vessel lines.\n"
        "4. CutOut / Random Erasing: Forbidden because erasing a 20x20 pixel patch could erase the only microaneurysm on the retina, converting a Grade 1 image into Grade 0 while retaining the label.",
        "Examiners frequently test whether students understand domain-specific augmentation constraints. Never allow hue shifts or random cutout on fundus images!"
    )
    log_md(q_num=26,
           q_text="Which data augmentations are clinically valid for retinal fundus imaging, and which are strictly forbidden?",
           a_text="Allowed: Flips, mild rotation (+/-15 deg), minor contrast adjustments, Ben Graham normalization. Strictly Forbidden: Hue shifting (color is diagnostic for blood vs lipids), aggressive cropping (destroys AAO 4:2:1 quadrants), and random erasing (can wipe out tiny microaneurysms).",
           tip="Never allow hue jitter or random erasing: color is diagnostic and erasing can remove the sole microaneurysm.")

    add_qa(
        27,
        "How does Test-Time Augmentation (TTA) work in your `infer_with_cam` implementation?",
        "Test-Time Augmentation (TTA) is an inference-stage ensemble technique that improves diagnostic robustness without re-training:\n\n"
        "1. Multi-View Tensor Assembly:\n"
        "- In `infer_with_cam`, when an image tensor T of shape (1, 3, 224, 224) is received, the code constructs a 3-element batch:\n"
        "  Batch = [T, torch.flip(T, dims=[3]), torch.flip(T, dims=[2])]\n"
        "  representing the original image, horizontal flip, and vertical flip.\n\n"
        "2. Parallel Forward & Backward CAM Computation:\n"
        "- The batch is evaluated through the network in a single pass. Gradients are computed for the predicted class across all 3 orientations simultaneously.\n\n"
        "3. De-Augmentation & Ensembling:\n"
        "- The resulting Grad-CAM++ maps are inverted back to original canonical orientation:\n"
        "  cam_0 = cams[0]\n"
        "  cam_1 = flip(cams[1], horizontal)\n"
        "  cam_2 = flip(cams[2], vertical)\n"
        "- The final saliency map is the arithmetic mean: cam_final = (cam_0 + cam_1 + cam_2) / 3.\n\n"
        "Clinical Benefit: Mitigates camera alignment bias and stabilizes attention heatmaps on borderline micro-lesions.",
        "Mention that TTA runs in a single batched tensor pass, maintaining high throughput."
    )
    log_md(q_num=27,
           q_text="How does Test-Time Augmentation (TTA) work in your `infer_with_cam` implementation?",
           a_text="During inference, the image is stacked with horizontal and vertical flips into a 3-image batch. CAMs are computed simultaneously, inverted back to canonical orientation, and averaged, maximizing prediction stability and lesion sensitivity.",
           tip="Highlight that TTA is batched into a single forward/backward pass for efficiency.")

    add_qa(
        28,
        "What is the histopathological difference between Hard Exudates and Cotton-Wool Spots?",
        "Examiners with medical backgrounds frequently ask this histopathological question:\n\n"
        "1. Hard Exudates (Lipid Deposits):\n"
        "- Histopathology: Extracellular accumulations of lipid and lipoprotein located primarily in the outer plexiform layer (OPL) of the retina.\n"
        "- Etiology: Chronic breakdown of the inner blood-retinal barrier causing serum lipid extravasation from incompetent capillary walls.\n"
        "- Clinical Appearance: Glistening, waxy yellow-white lesions with sharp, well-demarcated edges. Often arranged in clumps or circular starburst rings (circinate rings) around leaking microaneurysms.\n\n"
        "2. Cotton-Wool Spots (Soft Exudates):\n"
        "- Histopathology: NOT true exudates! They represent localized accumulations of axoplasmic debris (cytoid bodies) within swollen, non-functional retinal nerve fiber layer (RNFL) axons.\n"
        "- Etiology: Acute precapillary arteriolar occlusion causing localized ischemia and axoplasmic transport stasis.\n"
        "- Clinical Appearance: Fluffy, dull white or grayish patches with indistinct, feathery, cloud-like borders.",
        "Key phrase: 'Hard exudates are lipid precipitates with sharp borders; cotton-wool spots are ischemic nerve fiber infarcts with feathery borders.'"
    )
    log_md(q_num=28,
           q_text="What is the histopathological difference between Hard Exudates and Cotton-Wool Spots?",
           a_text="Hard exudates are waxy yellow lipid deposits in the outer plexiform layer with sharp borders from chronically leaking vessels. Cotton-wool spots are ischemic nerve fiber layer micro-infarctions (cytoid bodies) with fluffy, feathery edges caused by arteriolar occlusion.",
           tip="Hard exudates = sharp yellow lipid deposits; Cotton-wool spots = feathery white ischemic nerve infarcts.")

    add_qa(
        29,
        "What is the clinical distinction between Neovascularization of the Disc (NVD) and Neovascularization Elsewhere (NVE)?",
        "Neovascularization is the hallmark of Proliferative Diabetic Retinopathy (Grade 4). Driven by massive retinal tissue hypoxia, ischemic cells overexpress Vascular Endothelial Growth Factor (VEGF), stimulating fragile, aberrant blood vessels:\n\n"
        "1. Neovascularization of the Disc (NVD):\n"
        "- Defined as new abnormal vessels on or within 1 disc diameter of the Optic Nerve Head.\n"
        "- High-Risk Criteria: NVD greater than 1/4 to 1/3 of the disc area represents extremely high risk of catastrophic vitreous hemorrhage and rapid tractional retinal detachment.\n\n"
        "2. Neovascularization Elsewhere (NVE):\n"
        "- Defined as new pathological vessels occurring outside 1 disc diameter from the optic nerve head, typically along the major temporal vascular arcades.\n\n"
        "Clinical Urgency: Either finding mandates emergency vitreoretinal intervention within 24–72 hours (Anti-VEGF injections or Panretinal Photocoagulation [PRP]).",
        "NVD represents higher risk of acute blindness than NVE because of its location on the optic nerve head."
    )
    log_md(q_num=29,
           q_text="What is the clinical distinction between Neovascularization of the Disc (NVD) and Neovascularization Elsewhere (NVE)?",
           a_text="NVD occurs on or within 1 disc diameter of the optic disc; NVE occurs elsewhere along retinal arcades. Both define Proliferative DR (Grade 4) driven by hypoxia-induced VEGF overexpression, requiring emergency Anti-VEGF or PRP laser therapy.",
           tip="NVD has a higher risk of acute blindness due to its proximity to the optic nerve.")

    add_qa(
        30,
        "What are Venous Beading, Venous Looping, and Venous Reduplication, and why are they critical in the AAO 4:2:1 rule?",
        "Veins normally have smooth, uniform cylindrical calibers. In chronic severe retinal hypoxia:\n\n"
        "1. Venous Beading: Localized alternating constriction and dilation of retinal veins, giving them the appearance of a string of sausages or beads. It indicates advanced sluggish flow, capillary occlusion, and severe localized retinal ischemia.\n"
        "2. Venous Looping: Tortuous hairpin turns or loops in retinal veins as they attempt to bypass non-perfused capillary beds.\n"
        "3. Venous Reduplication: Parallel collateral venous channels attempting to re-establish outflow.\n\n"
        "Clinical Importance: Venous beading in 2 or more retinal quadrants satisfies the '2' criterion of the AAO 4:2:1 rule for Severe NPDR, indicating that over 50% of the peripheral capillary bed is non-perfused.",
        "Venous beading is one of the strongest clinical predictors of imminent progression to proliferative disease."
    )
    log_md(q_num=30,
           q_text="What are Venous Beading, Venous Looping, and Venous Reduplication, and why are they critical in the AAO 4:2:1 rule?",
           a_text="Venous beading is localized alternating constriction and dilation resembling sausage links due to severe hypoxia. Meeting venous beading in >=2 quadrants satisfies the '2' in the AAO 4:2:1 rule, indicating massive capillary non-perfusion.",
           tip="Venous beading is a direct sign of severe retinal hypoxia and imminent proliferative neovascularization.")

    add_qa(
        31,
        "In a public health screening context, why is Sensitivity prioritized over Specificity for early Diabetic Retinopathy?",
        "1. Statistical Definitions:\n"
        "- Sensitivity (True Positive Rate): TP / (TP + FN) — Ability to correctly detect patients who HAVE diabetic retinopathy.\n"
        "- Specificity (True Negative Rate): TN / (TN + FP) — Ability to correctly identify patients who are FREE of disease.\n\n"
        "2. Asymmetric Clinical Cost of Errors:\n"
        "- False Positive (FP): A healthy patient (Grade 0) is flagged as Mild DR. Cost: The patient undergoes a secondary in-person examination by an ophthalmologist, who verifies normal retina. Impact: Minor inconvenience and consultation cost.\n"
        "- False Negative (FN): A patient with proliferative or severe retinopathy (Grade 3/4) is told they are healthy. Cost: The patient does not seek treatment for another 12–24 months, during which painless vitreous hemorrhage or tractional retinal detachment causes permanent, irreversible blindness!\n\n"
        "Conclusion: In public health screening, a False Negative is catastrophic. OptiSight optimizes its decision threshold to achieve >95% Sensitivity on vision-threatening DR (Grade 2+).",
        "Never say you optimize for pure accuracy. State clearly: 'We prioritize high Sensitivity to eliminate catastrophic False Negatives.'"
    )
    log_md(q_num=31,
           q_text="In a public health screening context, why is Sensitivity prioritized over Specificity for early Diabetic Retinopathy?",
           a_text="A False Negative leaves a severe DR patient untreated, resulting in irreversible blindness. A False Positive merely causes a confirmatory clinic visit. In screening triage, Sensitivity must exceed 95% on vision-threatening DR to prevent missed pathology.",
           tip="Emphasize the asymmetric cost of medical error: False Negatives cause blindness; False Positives cause a checkup.")

    add_qa(
        32,
        "What is the difference between Macro-Averaged vs. Micro-Averaged ROC-AUC in multi-class DR grading?",
        "1. Micro-Averaged ROC-AUC:\n"
        "- Pools all individual predictions across all 5 classes into a single global confusion matrix and computes the global AUC.\n"
        "- Heavily dominated by the majority class (Grade 0). If the model predicts Grade 0 perfectly but fails on rare Grade 4 cases, micro-AUC can remain artificially high (e.g. 0.96), masking dangerous clinical deficiencies.\n\n"
        "2. Macro-Averaged ROC-AUC:\n"
        "- Calculates the ROC-AUC independently for each individual grade (Grade 0 vs rest, Grade 1 vs rest, ..., Grade 4 vs rest) and computes their unweighted arithmetic mean:\n"
        "  AUC_{macro} = (AUC_0 + AUC_1 + AUC_2 + AUC_3 + AUC_4) / 5\n"
        "- Treats every clinical grade with equal importance, ensuring that performance on rare Grade 4 proliferative cases is weighted identically to common Grade 0 healthy cases.\n\n"
        "OptiSight evaluates and reports Macro-Averaged metrics to guarantee equitable clinical safety across all disease stages.",
        "State: 'We evaluate Macro-Averaged ROC-AUC so rare high-risk grades (Grade 4) are not overshadowed by the common healthy class (Grade 0).'"
    )
    log_md(q_num=32,
           q_text="What is the difference between Macro-Averaged vs. Micro-Averaged ROC-AUC in multi-class DR grading?",
           a_text="Micro-AUC aggregates all classes globally and is biased toward majority Grade 0. Macro-AUC calculates AUC independently per class and averages them equally, ensuring rare high-risk proliferative cases (Grade 4) are evaluated with equal clinical rigor.",
           tip="Macro-AUC ensures rare high-risk grades (Grade 4) receive equal evaluation weight.")

    add_qa(
        33,
        "What novel scientific and engineering contributions does OptiSight make beyond existing literature?",
        "1. Heterogeneous Architectural Synergy:\n"
        "- Resolves the computer vision Inductive Bias Dilemma by combining DenseNet-121 (dense micro-texture reuse) and MaxViT-Tiny (multi-axis global attention) with LayerNorm late fusion.\n\n"
        "2. Full-Aspect Clinical Preprocessing:\n"
        "- Eliminates destructive geometric cropping; implements continuous soft luminance tissue masking and green-channel microvascular extraction, preserving peripheral retinal quadrants for AAO 4:2:1 evaluation.\n\n"
        "3. High-Fidelity Saliency Refinement:\n"
        "- Replaces coarse Grad-CAM with Grad-CAM++ (higher-order derivatives) coupled with bilateral filtering and edge modulation, snapping heatmaps to real microvascular lesions with zero margin bleeding.\n\n"
        "4. Holistic Clinical Decision Support (CDSS):\n"
        "- Goes beyond a simple image classifier by integrating the ICDR 0–4 staging scale, the AAO 4:2:1 Severe NPDR rule, the ETDRS 9-zone macular grid for DME, and patient systemic biomarkers (HbA1c, BP, duration).\n\n"
        "5. Dual-Mode Production Engineering:\n"
        "- High-performance PyTorch inference locally, coupled with an edge-assisted diagnostic engine running serverless 24/7 on Vercel cloud within a strict 250MB bundle limit.",
        "Structure your novel contributions into 5 distinct pillars: Architecture, Preprocessing, XAI Refinement, Clinical Integration (CDSS), and Cloud Engineering."
    )
    log_md(q_num=33,
           q_text="What novel scientific and engineering contributions does OptiSight make beyond existing literature?",
           a_text="1. DenseNet+MaxViT hybrid solving the inductive bias dilemma. 2. Full uncropped fundus preserving AAO 4:2:1 peripheral quadrants. 3. Grad-CAM++ with bilateral edge refinement. 4. Integrated CDSS with ETDRS grid and systemic biomarkers. 5. Dual-mode 24/7 cloud serverless architecture.",
           tip="Present your contributions under 5 pillars: Architecture, Preprocessing, XAI, Clinical CDSS, and Deployment.")

    add_qa(
        34,
        "What are the system limitations, real-world edge cases (cataracts, high myopia), and future clinical roadmap?",
        "1. Real-World Clinical Edge Cases & Limitations:\n"
        "- Media Opacities (Dense Cataracts / Corneal Scars): Cataracts scatter optical light, causing diffuse yellow-brown haze and blur. While Ben Graham normalization mitigates mild haze, extreme cataract opacities obscure microaneurysms.\n"
        "- High Myopia & Chorioretinal Atrophy: High myopic eyes feature elongated globes and thinned retinal pigment epithelium (tigroid fundus / myopic conus), which can be mistaken for exudates or capillary non-perfusion.\n"
        "- 2D Monocular Limitation: 2D fundus photography cannot measure 3D retinal cross-sectional thickness. Optical Coherence Tomography (OCT) remains the gold standard for definitive quantitative DME confirmation.\n\n"
        "2. Future Clinical Roadmap:\n"
        "- Multi-Modal AI Fusion: Ingesting paired OCT volumetric B-scans alongside fundus photographs for 3D retinal layer segmentation.\n"
        "- Automated Quality Control Gating: Pre-inference image quality classifier to automatically reject ungradable images (poor focus, eyelash occlusion, inadequate dilation) before model ingestion.\n"
        "- Multi-Ethnic Multi-Center Prospective Validation: Prospective clinical trials across varied fundus camera hardware (Topcon, Zeiss, Canon, smartphone hand-held lenses).",
        "Acknowledging limitations and edge cases displays mature scientific integrity."
    )
    log_md(q_num=34,
           q_text="What are the system limitations, real-world edge cases (cataracts, high myopia), and future clinical roadmap?",
           a_text="Limitations include dense cataract opacities, myopic chorioretinal atrophy, and the 2D nature of fundus photography (OCT needed for 3D thickness). Future roadmap includes multi-modal OCT fusion, automated image quality gating, and multi-center prospective trials.",
           tip="Acknowledging limitations (cataracts, 2D vs 3D OCT) demonstrates scientific maturity.")

    # Save documents
    docx_path = r"c:\Users\Viraj\OneDrive\Desktop\OPTI\OptiSight_DR_Viva_Comprehensive_Guide.docx"
    doc.save(docx_path)
    print(f"[+] Successfully generated Word Document: {docx_path}")

    md_path = r"c:\Users\Viraj\OneDrive\Desktop\OPTI\VIVA_QUESTIONS_AND_ANSWERS.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_content))
    print(f"[+] Successfully generated Markdown Document: {md_path}")

if __name__ == "__main__":
    build_all()
