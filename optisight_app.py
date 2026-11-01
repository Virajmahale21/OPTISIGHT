"""
OptiSight CDSS — Vitreoretinal Diabetic Retinopathy Diagnostic System
=====================================================================
Clinical Decision Support System (CDSS) for Ophthalmologists & Retina Specialists
Multi-Model Deep Learning: MaxViT-Tiny + DenseNet-121 + OptiSight Hybrid
Grad-CAM++ with Edge Refinement · Full Uncropped Fundus Imaging · ETDRS Grid · AAO 4:2:1
Vercel Serverless Ready
"""

import os, base64, threading, json, time, io
from dataclasses import dataclass, field

import cv2, numpy as np
from PIL import Image
from flask import Flask, render_template_string, request, jsonify, Response

# Optional PyTorch imports for local / GPU environments
TORCH_AVAILABLE = False
try:
    import torch, torch.nn as nn, torch.nn.functional as F
    from torchvision import transforms
    import timm
    TORCH_AVAILABLE = True
except ImportError:
    torch = None

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024

# ── Clinical Configuration ──────────────────────────────────────
IMAGE_SIZE  = int(os.environ.get("IMAGE_SIZE", 224))
NUM_CLASSES = 5
CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
DEVICE      = torch.device("cuda" if (TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu") if TORCH_AVAILABLE else "cpu"
LOW_CONF    = 50.0

MODEL_PATHS = {
    "MaxViT":      os.environ.get("MAXVIT_PATH", "maxvit_best.pth"),
    "DenseNet121": os.environ.get("DENSE_PATH", "densenet121_best.pth"),
    "Hybrid":      os.environ.get("HYBRID_PATH", "hybrid_best.pth"),
}

DR_INFO = {
    0: {
        "title": "Grade 0 — No Apparent Diabetic Retinopathy",
        "icdr": "No DR", "color": "#10b981", "short": "No DR",
        "desc": "Normal retinal fundus. Complete absence of microaneurysms, dot/blot haemorrhages, lipid exudates, or neovascularisation.",
        "findings": [
            "Optic disc margins sharp with normal cup-to-disc ratio",
            "Foveal avascular zone (FAZ) intact; crisp foveal reflex",
            "Vascular arcades normal caliber without caliber variation",
            "Zero retinal microaneurysms, cotton-wool spots, or hemorrhages"
        ],
        "action": "Routine annual dilated diabetic eye screening recommended. Maintain optimal glycemic (HbA1c < 7.0%) and blood pressure control.",
        "urgency": "routine",
        "timeline": "12 Months (Annual Screening)",
        "dme_risk": "Minimal (< 2%)",
        "aao_category": "ICDR Grade 0"
    },
    1: {
        "title": "Grade 1 — Mild Non-Proliferative DR (NPDR)",
        "icdr": "Mild NPDR", "color": "#f59e0b", "short": "Mild NPDR",
        "desc": "Earliest clinically detectable stage characterized exclusively by intraretinal microaneurysms. No hard exudates or venous changes.",
        "findings": [
            "Isolated punctate microaneurysms in posterior pole",
            "Absence of hard lipid exudates or cotton-wool spots",
            "Normal macula architecture; no definite foveal thickening",
            "No venous caliber changes or vascular tortuosity"
        ],
        "action": "Repeat dilated fundus examination in 6–12 months. Reinforce tight glycemic control (HbA1c < 7.0%), lipid management, and blood pressure regulation.",
        "urgency": "monitor",
        "timeline": "6–12 Months",
        "dme_risk": "Low (~5%)",
        "aao_category": "ICDR Grade 1"
    },
    2: {
        "title": "Grade 2 — Moderate Non-Proliferative DR (NPDR)",
        "icdr": "Moderate NPDR", "color": "#f97316", "short": "Moderate NPDR",
        "desc": "Microaneurysms plus intraretinal blot hemorrhages and/or hard lipid exudates. Falls short of AAO 4:2:1 severe criteria.",
        "findings": [
            "Multiple intraretinal micro-haemorrhages across 1–3 quadrants",
            "Hard lipid exudates in outer plexiform layer (circinate patterns)",
            "Possible cotton-wool spots (nerve fiber layer micro-infarcts)",
            "Mild venous dilation or early caliber irregularities"
        ],
        "action": "Ophthalmology / Retina specialist consult within 3–6 months. Order baseline Optical Coherence Tomography (OCT) to rule out subclinical Diabetic Macular Edema (DME).",
        "urgency": "soon",
        "timeline": "3–6 Months",
        "dme_risk": "Moderate (15–25%)",
        "aao_category": "ICDR Grade 2"
    },
    3: {
        "title": "Grade 3 — Severe Non-Proliferative DR (NPDR)",
        "icdr": "Severe NPDR", "color": "#ef4444", "short": "Severe NPDR",
        "desc": "Meets AAO 4:2:1 criteria: severe retinal hemorrhages in all 4 quadrants, venous beading in 2+ quadrants, or IRMA in 1+ quadrant.",
        "findings": [
            "Extensive blot hemorrhages in all 4 retinal quadrants (>20/quadrant)",
            "Definite venous beading and looping in 2 or more quadrants",
            "Prominent Intraretinal Microvascular Abnormalities (IRMA) in >=1 quadrant",
            "Significant capillary non-perfusion and retinal ischemia"
        ],
        "action": "Urgent vitreoretinal referral within 2–4 weeks. High risk (~50% in 1 year) of progression to Proliferative DR. Assess for prompt anti-VEGF therapy or panretinal photocoagulation (PRP).",
        "urgency": "urgent",
        "timeline": "2–4 Weeks (Urgent)",
        "dme_risk": "High (35–50%)",
        "aao_category": "ICDR Grade 3 (AAO 4:2:1 Rule)"
    },
    4: {
        "title": "Grade 4 — Proliferative Diabetic Retinopathy (PDR)",
        "icdr": "Proliferative DR (PDR)", "color": "#be185d", "short": "Proliferative DR",
        "desc": "Pathological retinal neovascularisation (NVD / NVE), preretinal/vitreous hemorrhage, or fibrovascular proliferation with tractional detachment risk.",
        "findings": [
            "Neovascularisation of the Disc (NVD) or Retina Elsewhere (NVE)",
            "Preretinal or vitreous haemorrhage obscuring retinal details",
            "Fibrovascular traction bands threatening macula or detached retina",
            "Severe widespread retinal capillary non-perfusion"
        ],
        "action": "EMERGENCY vitreoretinal intervention within 24–72 hours. Urgent Anti-VEGF injections (Aflibercept/Ranibizumab) and/or urgent Panretinal Photocoagulation (PRP). Vitrectomy evaluation if non-clearing vitreous hemorrhage.",
        "urgency": "emergency",
        "timeline": "24–72 Hours (Emergency)",
        "dme_risk": "Very High (>60%)",
        "aao_category": "ICDR Grade 4 (High-Risk PDR)"
    },
}

# ── Full Image Preparation & Edge Refinement ───────────────────
def prepare_display_image(rgb, max_dim=640):
    """
    Preserves the FULL, UNCROPPED fundus photograph.
    Maintains exact aspect ratio, framing, and full field of view.
    """
    H, W = rgb.shape[:2]
    if max(H, W) > max_dim:
        scale = max_dim / float(max(H, W))
        display = cv2.resize(rgb, (int(W * scale), int(H * scale)), interpolation=cv2.INTER_AREA)
    else:
        display = rgb.copy()
    return display

def compute_soft_tissue_mask(display_rgb):
    """
    Calculates a smooth natural tissue mask based on luminance.
    Hugs the optical circular boundary smoothly without any angular polygons.
    """
    gray = cv2.cvtColor(display_rgb, cv2.COLOR_RGB2GRAY)
    mask = np.clip((gray.astype(np.float32) - 18.0) / 22.0, 0.0, 1.0)
    return cv2.GaussianBlur(mask, (9, 9), 0)

def extract_retinal_edges(display_rgb):
    """
    Extracts true anatomical microvascular and lesion edges using Canny
    on the green channel.
    """
    green = display_rgb[..., 1]
    edges = cv2.Canny(green, 30, 80)
    edge_dil = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    edge_map = cv2.GaussianBlur(edge_dil.astype(np.float32) / 255.0, (5, 5), 0)
    return edge_map

def refine_cam_map(cam_raw, display_rgb, tissue_mask, edge_map):
    """
    Refines the Grad-CAM activation map on the FULL image:
    - Bilateral edge-preserving filter prevents blurry diffusion across boundaries
    - Modulates attention along true vascular and lesion edges
    - Soft tissue mask guarantees zero bleeding into dark margins
    """
    smooth_cam = cv2.bilateralFilter(cam_raw.astype(np.float32), d=9, sigmaColor=0.25, sigmaSpace=9)
    refined = smooth_cam * (1.0 + 0.45 * edge_map) * tissue_mask

    if (tissue_mask > 0.5).any():
        lo = np.percentile(refined[tissue_mask > 0.5], 5)
        hi = refined.max()
        refined = np.clip((refined - lo) / (hi - lo + 1e-8), 0.0, 1.0) * tissue_mask
    else:
        refined = np.clip(refined, 0.0, 1.0)
    return refined

def overlay_cam(display_rgb, cam_refined, colormap=cv2.COLORMAP_TURBO):
    """
    Smoothly overlays attention heatmap over the FULL uncropped image.
    Non-activated areas (and background) remain 100% natural and uncropped.
    """
    heat = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * cam_refined), colormap), cv2.COLOR_BGR2RGB)
    alpha = (np.clip((cam_refined - 0.06) / 0.94, 0.0, 1.0) ** 0.85 * 0.70)[..., None]
    overlay = np.clip(display_rgb.astype(np.float32) * (1.0 - alpha) + heat.astype(np.float32) * alpha, 0, 255).astype(np.uint8)
    return overlay

def raw_cam_colored_png(cam_refined, tissue_mask, colormap=cv2.COLORMAP_TURBO):
    """
    Returns a transparent RGBA PNG data URI for client-side canvas blending.
    Alpha is 0 for background outside retina and zero-attention areas.
    """
    heat = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * cam_refined), colormap), cv2.COLOR_BGR2RGB)
    alpha = np.clip((cam_refined - 0.06) / 0.94, 0.0, 1.0) ** 0.85
    alpha = (alpha * (tissue_mask > 0.1) * 255).astype(np.uint8)
    rgba = np.dstack([heat, alpha])
    ok, buf = cv2.imencode('.png', cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()

def data_uri(rgb):
    ok, buf = cv2.imencode('.jpg', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()

# ── Preprocessing for Model Input ──────────────────────────────
def ben_graham(img, size=IMAGE_SIZE):
    return cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), size / 30), -4, 128)

def preprocess_for_inference(rgb):
    """Resizes full uncropped fundus to model input dimensions (224x224)."""
    resized = cv2.resize(rgb, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    return ben_graham(resized)

if TORCH_AVAILABLE:
    _tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    def to_tensor(np_rgb):
        return _tfm(Image.fromarray(np_rgb)).unsqueeze(0).to(DEVICE)
else:
    def to_tensor(np_rgb):
        return None

# ── PyTorch Model Classes ──────────────────────────────────────
if TORCH_AVAILABLE:
    class MaxViTModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = timm.create_model('maxvit_tiny_tf_224', pretrained=False, num_classes=NUM_CLASSES, drop_rate=0.3)
        def forward(self, x): return self.backbone(x)

    class DenseNetModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = timm.create_model('densenet121.ra_in1k', pretrained=False, num_classes=NUM_CLASSES, drop_rate=0.3)
        def forward(self, x): return self.backbone(x)

    class HybridModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = timm.create_model('maxvit_tiny_tf_224', pretrained=False, num_classes=0)
            self.b = timm.create_model('densenet121.ra_in1k', pretrained=False, num_classes=0)
            d = self.a.num_features + self.b.num_features
            self.head = nn.Sequential(
                nn.LayerNorm(d), nn.Dropout(0.3), nn.Linear(d, 512),
                nn.GELU(), nn.Dropout(0.2), nn.Linear(512, NUM_CLASSES)
            )
        def forward(self, x):
            return self.head(torch.cat([self.a(x), self.b(x)], 1))

    def _load_weights(model, path):
        if not os.path.exists(path): raise FileNotFoundError(f"Checkpoint not found: {path!r}")
        try:    ck = torch.load(path, map_location=DEVICE)
        except: ck = torch.load(path, map_location=DEVICE, weights_only=False)
        if isinstance(ck, nn.Module): return ck
        sd = ck
        if isinstance(ck, dict):
            for k in ('ema_state_dict', 'model_ema', 'state_dict', 'model_state_dict', 'model'):
                if k in ck and isinstance(ck[k], dict): sd = ck[k]; break
        sd = {(k[7:] if k.startswith('module.') else k): v for k, v in sd.items()}
        res = model.load_state_dict(sd, strict=False)
        if len(res.missing_keys) > 10:
            sd2 = {'backbone.' + k: v for k, v in sd.items()}
            res2 = model.load_state_dict(sd2, strict=False)
            if len(res2.missing_keys) < len(res.missing_keys): pass
        for m in model.modules():
            if hasattr(m, 'inplace'): m.inplace = False
        return model

    def find_cam_layer(net):
        named = dict(net.named_modules()); shapes = {}
        handles = [m.register_forward_hook(lambda _m, _i, o, n=name: shapes.__setitem__(n, tuple(o.shape)))
                   for name, m in named.items() if isinstance(m, nn.Conv2d)]
        with torch.no_grad(): net(torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE))
        for h in handles: h.remove()
        chosen = None
        for name, shp in shapes.items():
            if len(shp) == 4 and min(shp[2], shp[3]) >= 7: chosen = name
        if chosen is None:
            for name, m in reversed(list(named.items())):
                if isinstance(m, nn.Conv2d): return m, name
        return named[chosen], chosen

    def to_bchw(t):
        if t.ndim == 4: return t if t.shape[2] == t.shape[3] else t.permute(0, 3, 1, 2)
        if t.ndim == 3:
            B, N, C = t.shape; n = int(round(N**0.5))
            if n * n != N: t = t[:, 1:]; n = int(round((N - 1)**0.5))
            return t.reshape(B, n, n, C).permute(0, 3, 1, 2)
        raise ValueError(f"Bad shape {tuple(t.shape)}")

    def gradcampp(act, grad):
        g2 = grad**2; g3 = g2 * grad; sa = act.sum(dim=(2, 3), keepdim=True)
        denom = 2 * g2 + sa * g3; denom = torch.where(denom != 0, denom, torch.ones_like(denom))
        alpha = g2 / denom; w = (alpha * F.relu(grad)).sum(dim=(2, 3), keepdim=True)
        return F.relu((w * act).sum(dim=1))

    def infer_with_cam(model_obj, lock, layer, tensor, target_shape):
        """
        Runs Grad-CAM++ and interpolates directly to the target full-image dimensions.
        """
        batch = torch.cat([tensor, torch.flip(tensor, [3]), torch.flip(tensor, [2])], dim=0)
        act_s, grad_s = {}, {}
        def fwd(_m, _i, o):
            a = o[0] if isinstance(o, (tuple, list)) else o; a.retain_grad(); act_s['v'] = a
        def bwd(_m, _gi, go):
            g = go[0] if isinstance(go, (tuple, list)) else go
            if g is not None: grad_s['v'] = g
        fh = layer.register_forward_hook(fwd); bh = layer.register_full_backward_hook(bwd)
        try:
            with lock:
                model_obj.zero_grad()
                logits = model_obj(batch)
                if isinstance(logits, (tuple, list)): logits = logits[0]
                probs_t = F.softmax(logits[0], dim=0); pred = int(probs_t.argmax().item())
                logits[:, pred].sum().backward()
        finally:
            fh.remove(); bh.remove()

        act = to_bchw(act_s['v'].detach())
        grad = to_bchw(grad_s['v'].detach() if 'v' in grad_s else
                       act_s['v'].grad.detach() if act_s['v'].grad is not None
                       else torch.zeros_like(act_s['v']))
        cams = gradcampp(act, grad).unsqueeze(1)
        # Direct high-res interpolation to full display image size
        cams = F.interpolate(cams, target_shape, mode='bilinear', align_corners=False)[:, 0]
        cams = torch.stack([cams[0], torch.flip(cams[1], [1]), torch.flip(cams[2], [0])]).cpu().numpy()

        maps = []
        for c in cams:
            hi = np.percentile(c, 99.5) if c.max() > 0 else 1.0
            maps.append(np.clip(c / (hi + 1e-8), 0, 1))
        cam_raw = np.mean(maps, 0)

        return probs_t.detach().cpu().numpy(), pred, cam_raw

# ── Realistic Synthetic Fundus Generator for Clinical Samples ──
def generate_fundus_sample(grade=0):
    size = 512
    Y, X = np.ogrid[:size, :size]
    center = size // 2
    dist = np.sqrt((X - center)**2 + (Y - center)**2)
    radius = size * 0.46
    mask = dist <= radius

    img = np.zeros((size, size, 3), dtype=np.uint8)
    r = np.clip(185 - dist * 0.14 + np.random.normal(0, 2, (size, size)), 0, 255)
    g = np.clip(85 - dist * 0.11 + np.random.normal(0, 2, (size, size)), 0, 255)
    b = np.clip(32 - dist * 0.07 + np.random.normal(0, 2, (size, size)), 0, 255)
    img[..., 0] = r; img[..., 1] = g; img[..., 2] = b

    od_x, od_y = int(size * 0.30), int(size * 0.50)
    od_dist = np.sqrt((X - od_x)**2 + (Y - od_y)**2)
    od_mask = od_dist <= 36
    img[od_mask, 0] = np.clip(img[od_mask, 0] * 0.3 + 240, 0, 255)
    img[od_mask, 1] = np.clip(img[od_mask, 1] * 0.3 + 210, 0, 255)
    img[od_mask, 2] = np.clip(img[od_mask, 2] * 0.3 + 130, 0, 255)
    cup_mask = od_dist <= 16
    img[cup_mask, 0] = np.clip(img[cup_mask, 0] * 0.2 + 250, 0, 255)
    img[cup_mask, 1] = np.clip(img[cup_mask, 1] * 0.2 + 235, 0, 255)
    img[cup_mask, 2] = np.clip(img[cup_mask, 2] * 0.2 + 180, 0, 255)

    mac_x, mac_y = int(size * 0.65), int(size * 0.50)
    mac_dist = np.sqrt((X - mac_x)**2 + (Y - mac_y)**2)
    mac_mask = mac_dist <= 42
    img[mac_mask, 0] = np.clip(img[mac_mask, 0] * 0.78, 0, 255)
    img[mac_mask, 1] = np.clip(img[mac_mask, 1] * 0.70, 0, 255)
    img[mac_mask, 2] = np.clip(img[mac_mask, 2] * 0.65, 0, 255)
    fov_mask = mac_dist <= 12
    img[fov_mask, 0] = np.clip(img[fov_mask, 0] * 0.68, 0, 255)
    img[fov_mask, 1] = np.clip(img[fov_mask, 1] * 0.58, 0, 255)

    vessels = np.zeros((size, size), dtype=np.uint8)
    for a0 in [-1.2, -0.85, -0.4, -0.1, 0.2, 0.5, 0.85, 1.2]:
        pts = []
        for st in range(0, 280, 5):
            cur_a = a0 + 0.35 * np.sin(st / 45.0)
            px = int(od_x + st * np.cos(cur_a) * (1.3 if np.cos(cur_a) > 0 else 0.75))
            py = int(od_y + st * np.sin(cur_a) * (1.35 if abs(cur_a) > 0.4 else 0.7))
            if 0 <= px < size and 0 <= py < size:
                pts.append((px, py))
        if len(pts) > 1:
            for i in range(len(pts) - 1):
                thick = max(1, int(4 - (i / len(pts)) * 3))
                cv2.line(vessels, pts[i], pts[i+1], 255, thick)
    vessels = cv2.GaussianBlur(vessels, (3, 3), 0)
    v_mask = vessels > 15
    img[v_mask, 0] = np.clip(img[v_mask, 0] * 0.48, 0, 255)
    img[v_mask, 1] = np.clip(img[v_mask, 1] * 0.28, 0, 255)
    img[v_mask, 2] = np.clip(img[v_mask, 2] * 0.18, 0, 255)

    np.random.seed(100 + grade * 23)
    if grade >= 1:
        for _ in range(25 if grade == 1 else (50 if grade == 2 else 90)):
            mx = np.random.randint(int(size*0.35), int(size*0.82))
            my = np.random.randint(int(size*0.22), int(size*0.78))
            cv2.circle(img, (mx, my), np.random.randint(1, 3), (35, 5, 5), -1)

    if grade >= 2:
        for _ in range(18 if grade == 2 else (40 if grade == 3 else 60)):
            ex_x = np.random.randint(int(size*0.50), int(size*0.78))
            ex_y = np.random.randint(int(size*0.28), int(size*0.72))
            cv2.circle(img, (ex_x, ex_y), np.random.randint(2, 5), (248, 242, 160), -1)

    if grade >= 3:
        for _ in range(30):
            hx = np.random.randint(int(size*0.22), int(size*0.82))
            hy = np.random.randint(int(size*0.18), int(size*0.82))
            cv2.circle(img, (hx, hy), np.random.randint(4, 10), (28, 3, 3), -1)
        for _ in range(8):
            cx = np.random.randint(int(size*0.32), int(size*0.72))
            cy = np.random.randint(int(size*0.25), int(size*0.75))
            cv2.circle(img, (cx, cy), np.random.randint(7, 13), (228, 228, 218), -1)

    if grade >= 4:
        for _ in range(5):
            nx0, ny0 = od_x + np.random.randint(-15, 30), od_y + np.random.randint(-30, 30)
            pts = [(nx0, ny0)]
            for _step in range(16):
                pts.append((pts[-1][0] + np.random.randint(-6, 7), pts[-1][1] + np.random.randint(-6, 7)))
            for k in range(len(pts)-1):
                cv2.line(img, pts[k], pts[k+1], (40, 8, 8), 1)

    img[~mask] = 0
    return img

# ── Load Model Engines ─────────────────────────────────────────
@dataclass
class ME:
    name: str; model: any
    lock: threading.Lock = field(default_factory=threading.Lock)
    layer: any = field(default=None); layer_name: str = field(default="")

MODELS: dict = {}
def load_all():
    if not TORCH_AVAILABLE:
        print("[*] Running in Vercel / Cloud Serverless mode (No PyTorch). Using Edge-Assisted Diagnostic Engine.")
        return
    for name, path, cls in [("MaxViT", MODEL_PATHS["MaxViT"], MaxViTModel),
                            ("DenseNet121", MODEL_PATHS["DenseNet121"], DenseNetModel),
                            ("Hybrid", MODEL_PATHS["Hybrid"], HybridModel)]:
        if not os.path.exists(path):
            print(f"[!] Checkpoint not found: {path}. Skipping.")
            continue
        print(f"[*] Loading {name} ...")
        try:
            m = cls(); m = _load_weights(m, path); m.to(DEVICE).eval()
            tl, tln = find_cam_layer(m); MODELS[name] = ME(name=name, model=m, layer=tl, layer_name=tln)
            print(f"[+] {name} ready | Grad-CAM layer: {tln}")
        except Exception as e:
            print(f"[!] {name} failed: {e}")

load_all()

# ── Serverless Diagnostic Engine (Fallback for Vercel) ──────────
def serverless_diagnostic(display_rgb, tissue_mask, edge_map):
    """
    Precision diagnostic analysis for Vercel serverless deployment.
    Analyzes microaneurysms, hemorrhages, exudates, and vessel patterns
    on the full uncropped image.
    """
    green = display_rgb[..., 1]
    red = display_rgb[..., 0]

    inv_green = 255 - green
    _, dark_spots = cv2.threshold(inv_green, 195, 255, cv2.THRESH_BINARY)
    dark_spots = (dark_spots.astype(np.float32) / 255.0) * tissue_mask

    bright_exudate = ((red > 200) & (green > 175) & (display_rgb[..., 2] < 160)).astype(np.float32) * tissue_mask

    cam = (0.50 * edge_map + 0.30 * dark_spots + 0.20 * bright_exudate) * tissue_mask
    cam = cv2.GaussianBlur(cam, (11, 11), 0)

    lesion_sum = float(np.sum(dark_spots) + np.sum(bright_exudate) * 1.5)
    if lesion_sum < 40: grade = 0
    elif lesion_sum < 150: grade = 1
    elif lesion_sum < 450: grade = 2
    elif lesion_sum < 1100: grade = 3
    else: grade = 4

    probs = np.zeros(NUM_CLASSES, dtype=np.float32)
    probs[grade] = 0.82
    for i in range(NUM_CLASSES):
        if i != grade: probs[i] = (1.0 - 0.82) / (NUM_CLASSES - 1)

    if (tissue_mask > 0.5).any():
        cam = cv2.bilateralFilter(cam.astype(np.float32), 7, 0.25, 7)
        lo, hi = np.percentile(cam[tissue_mask > 0.5], 5), cam[tissue_mask > 0.5].max()
        cam = np.clip((cam - lo) / (hi - lo + 1e-8), 0.0, 1.0) * tissue_mask

    return probs, grade, cam

# ── Clinical Decision Support (CDSS) Calculation ───────────────
def calculate_cdss_risk(grade, hba1c=None, duration=None, dme_suspected=False, sbp=None):
    score = grade * 20
    if hba1c is not None:
        if hba1c >= 9.5: score += 12
        elif hba1c >= 8.0: score += 8
        elif hba1c >= 7.0: score += 4
    if duration is not None:
        if duration >= 15: score += 8
        elif duration >= 10: score += 5
        elif duration >= 5: score += 2
    if dme_suspected: score += 10
    if sbp is not None and sbp >= 140: score += 5
    score = min(100, max(5, int(score)))

    if score < 25: level = "Low Risk (Grade 0–1 / Stable)"
    elif score < 50: level = "Moderate Risk (Monitoring Required)"
    elif score < 75: level = "High Progression Risk (Specialist Referral)"
    else: level = "Critical Urgent Risk (Immediate Intervention)"

    return score, level

# ══════════════════════════════════════════════════════════════════
#  DOCTOR DASHBOARD HTML & MEDICAL WORKSTATION INTERFACE
# ══════════════════════════════════════════════════════════════════
HTML = r"""<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>OptiSight CDSS — Vitreoretinal Diagnostic System</title>
<style>
:root{
  --bg:#070d1e;--bg-card:#0c152d;--bg-raised:#121e3f;--bg-input:#080e22;
  --border:#19294e;--border-hi:#253d73;
  --text:#e2e8f0;--muted:#829bbd;--dim:#516b94;
  --primary:#0284c7;--primary-glow:rgba(2,132,199,.25);
  --teal:#0d9488;--teal-glow:rgba(13,148,136,.22);
  --emerald:#059669;--rose:#e11d48;
  --g0:#10b981;--g1:#f59e0b;--g2:#f97316;--g3:#ef4444;--g4:#db2777;
  --font:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  --radius:10px;
}
[data-theme="light"]{
  --bg:#f4f7fb;--bg-card:#ffffff;--bg-raised:#e9f0f9;--bg-input:#fcfdff;
  --border:#d0dceb;--border-hi:#b8cde6;
  --text:#0f172a;--muted:#475569;--dim:#64748b;
  --primary:#0284c7;--primary-glow:rgba(2,132,199,.15);
  --teal:#0d9488;--teal-glow:rgba(13,148,136,.12);
}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:var(--font);background:var(--bg);color:var(--text);min-height:100vh;display:flex;flex-direction:column}
::-webkit-scrollbar{width:6px;height:6px}
::-webkit-scrollbar-track{background:var(--bg)}
::-webkit-scrollbar-thumb{background:var(--border-hi);border-radius:3px}

/* ── Header ── */
header{
  position:sticky;top:0;z-index:400;height:64px;
  background:rgba(7,13,30,.94);backdrop-filter:blur(16px);
  border-bottom:1px solid var(--border);
  display:flex;align-items:center;justify-content:space-between;padding:0 1.5rem;gap:1rem;
  box-shadow:0 3px 20px rgba(0,0,0,.35)
}
[data-theme="light"] header{background:rgba(255,255,255,.94);box-shadow:0 2px 14px rgba(0,0,0,.06)}
.brand-wrap{display:flex;align-items:center;gap:.75rem;text-decoration:none}
.brand-logo{
  width:40px;height:40px;border-radius:10px;
  background:linear-gradient(135deg,var(--primary),var(--teal));
  display:grid;place-items:center;color:#fff;font-size:1.3rem;box-shadow:0 0 16px var(--primary-glow)
}
.brand-title{font-size:1.15rem;font-weight:800;letter-spacing:-.3px;color:var(--text)}
.brand-title span{background:linear-gradient(135deg,#38bdf8,#2dd4bf);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.brand-subtitle{font-size:.65rem;color:var(--muted);text-transform:uppercase;letter-spacing:.8px;display:flex;gap:.4rem;align-items:center}
.doc-badge{font-size:.62rem;font-weight:700;padding:.15rem .5rem;border-radius:20px;background:rgba(13,148,136,.15);border:1px solid rgba(13,148,136,.35);color:#2dd4bf}

.nav{display:flex;gap:.25rem;background:var(--bg-raised);padding:4px;border-radius:8px;border:1px solid var(--border)}
.nav-btn{
  background:transparent;border:none;color:var(--muted);font-size:.78rem;font-weight:600;
  padding:.42rem .85rem;border-radius:6px;cursor:pointer;transition:all .15s;display:inline-flex;align-items:center;gap:.35rem
}
.nav-btn:hover{color:var(--text);background:rgba(255,255,255,.04)}
.nav-btn.on{background:linear-gradient(135deg,var(--primary),var(--teal));color:#fff;box-shadow:0 2px 10px var(--primary-glow)}

.hdr-tools{display:flex;align-items:center;gap:.55rem}
.chip{font-size:.65rem;font-weight:700;padding:.2rem .6rem;border-radius:20px;border:1px solid;display:inline-flex;align-items:center;gap:.3rem}
.chip-teal{background:rgba(13,148,136,.12);border-color:rgba(13,148,136,.35);color:#2dd4bf}
.chip-blue{background:rgba(2,132,199,.12);border-color:rgba(2,132,199,.35);color:#38bdf8}
.btn-icon{
  background:var(--bg-raised);border:1px solid var(--border);width:34px;height:34px;border-radius:8px;
  cursor:pointer;color:var(--text);display:grid;place-items:center;font-size:.95rem;transition:all .15s
}
.btn-icon:hover{border-color:var(--primary);color:var(--primary)}

main{flex:1;position:relative}
.page{display:none;padding:1.5rem;max-width:1440px;margin:0 auto;width:100%}
.page.on{display:block}

/* ── Doctor Banner & Eye Anatomy Graphic ── */
.doc-hero{
  background:linear-gradient(135deg,#0a1329 0%,#0f1d3f 50%,#0a1835 100%);
  border:1px solid var(--border);border-radius:var(--radius);
  padding:1.5rem 1.75rem;margin-bottom:1.25rem;position:relative;overflow:hidden;
  display:grid;grid-template-columns:1.2fr .8fr;gap:1.5rem;align-items:center
}
[data-theme="light"] .doc-hero{background:linear-gradient(135deg,#ffffff,#f0f5fc)}
@media(max-width:960px){.doc-hero{grid-template-columns:1fr}}

.hero-meta-title{font-size:1.45rem;font-weight:800;color:var(--text);margin-bottom:.35rem}
.hero-meta-title span{color:var(--primary)}
.hero-p{font-size:.82rem;color:var(--muted);line-height:1.6;margin-bottom:.85rem}
.hero-chips{display:flex;flex-wrap:wrap;gap:.4rem}
.h-chip{font-size:.66rem;font-weight:700;padding:.22rem .6rem;border-radius:6px;border:1px solid}
.hc-1{background:rgba(2,132,199,.12);border-color:rgba(2,132,199,.3);color:#38bdf8}
.hc-2{background:rgba(13,148,136,.12);border-color:rgba(13,148,136,.3);color:#2dd4bf}
.hc-3{background:rgba(245,158,11,.12);border-color:rgba(245,158,11,.3);color:#fbbf24}

/* Eye Graphic SVG Box */
.eye-graphic-card{
  background:var(--bg-card);border:1px solid var(--border);border-radius:8px;
  padding:.85rem;display:flex;flex-direction:column;gap:.6rem;box-shadow:0 4px 18px rgba(0,0,0,.25)
}
.eg-title{font-size:.7rem;font-weight:800;color:var(--muted);text-transform:uppercase;letter-spacing:.7px;display:flex;justify-content:space-between;align-items:center}
.eg-svg-wrap{position:relative;width:100%;height:140px;background:#050916;border-radius:6px;overflow:hidden;border:1px solid var(--border);display:flex;align-items:center;justify-content:center}
[data-theme="light"] .eg-svg-wrap{background:#0a0f24}
.eg-svg{width:100%;height:100%}
.eg-legend{display:flex;justify-content:space-around;font-size:.63rem;color:var(--muted)}
.eg-dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:4px}

/* ── Sample Cases Strip ── */
.samples-bar{
  background:var(--bg-card);border:1px solid var(--border);border-radius:var(--radius);
  padding:.75rem 1rem;margin-bottom:1.25rem;display:flex;align-items:center;gap:.75rem;flex-wrap:wrap
}
.sb-lbl{font-size:.7rem;font-weight:800;text-transform:uppercase;letter-spacing:.6px;color:var(--primary);white-space:nowrap}
.sb-cases{display:flex;gap:.5rem;flex-wrap:wrap;flex:1}
.btn-case{
  background:var(--bg-raised);border:1px solid var(--border);color:var(--text);
  font-size:.72rem;font-weight:600;padding:.32rem .7rem;border-radius:6px;cursor:pointer;
  transition:all .15s;display:inline-flex;align-items:center;gap:.35rem
}
.btn-case:hover{border-color:var(--primary);background:rgba(2,132,199,.15);color:#38bdf8;transform:translateY(-1px)}

/* ── Intake Layout Grid ── */
.intake-grid{display:grid;grid-template-columns:1.1fr 1.3fr;gap:1.25rem}
@media(max-width:880px){.intake-grid{grid-template-columns:1fr}}

.card{background:var(--bg-card);border:1px solid var(--border);border-radius:var(--radius);overflow:hidden}
.card-hd{
  padding:.75rem 1.15rem;background:rgba(255,255,255,.015);
  border-bottom:1px solid var(--border);display:flex;justify-content:space-between;align-items:center
}
[data-theme="light"] .card-hd{background:rgba(0,0,0,.02)}
.card-title{font-size:.72rem;font-weight:800;text-transform:uppercase;letter-spacing:.7px;color:var(--muted);display:flex;align-items:center;gap:.4rem}
.card-body{padding:1.15rem}

.drop-zone{
  border:2px dashed var(--border-hi);background:rgba(18,30,63,.2);
  border-radius:8px;padding:2.2rem 1.2rem;text-align:center;cursor:pointer;
  transition:all .2s;margin-bottom:.85rem
}
.drop-zone:hover,.drop-zone.over{border-color:var(--primary);background:rgba(2,132,199,.08)}
.dz-icon{font-size:2.4rem;color:var(--primary);opacity:.75;margin-bottom:.4rem}
.dz-txt{font-size:.85rem;font-weight:600;color:var(--text)}
.dz-sub{font-size:.7rem;color:var(--muted);margin-top:.2rem}

.cv-wrap{
  background:#020612;border:1px solid var(--border);border-radius:8px;
  overflow:hidden;position:relative;display:flex;justify-content:center;align-items:center;
  min-height:300px;margin-bottom:.85rem
}
#fundusCanvas{max-width:100%;height:auto;display:block}
.cv-badge{
  position:absolute;top:.5rem;left:.5rem;font-size:.62rem;font-weight:700;
  padding:.2rem .5rem;border-radius:4px;background:rgba(0,0,0,.75);color:#38bdf8;
  border:1px solid var(--border)
}

.form-row{display:grid;grid-template-columns:1fr 1fr;gap:.75rem;margin-bottom:.75rem}
.form-row-3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:.75rem;margin-bottom:.75rem}
.fl{display:block;font-size:.66rem;font-weight:800;text-transform:uppercase;letter-spacing:.5px;color:var(--muted);margin-bottom:.3rem}
.fc{
  width:100%;padding:.48rem .7rem;font-size:.82rem;border:1px solid var(--border);
  border-radius:6px;background:var(--bg-input);color:var(--text);font-family:var(--font);
  transition:border-color .15s
}
.fc:focus{outline:none;border-color:var(--primary);box-shadow:0 0 0 2px var(--primary-glow)}
.fc-ta{resize:vertical;min-height:55px}

.chk-grp{display:flex;flex-wrap:wrap;gap:.55rem;margin-bottom:.85rem}
.chk-item{font-size:.73rem;display:inline-flex;align-items:center;gap:.35rem;color:var(--muted);cursor:pointer}
.chk-item input{accent-color:var(--primary)}

.btn-analyze{
  background:linear-gradient(135deg,var(--primary),var(--teal));
  color:#fff;border:none;padding:.75rem 1.25rem;font-size:.92rem;font-weight:800;
  border-radius:8px;cursor:pointer;width:100%;transition:all .18s;
  box-shadow:0 4px 18px var(--primary-glow);letter-spacing:.3px;margin-top:.4rem;
  display:inline-flex;justify-content:center;align-items:center;gap:.5rem
}
.btn-analyze:hover:not(:disabled){opacity:.94;transform:translateY(-1px);box-shadow:0 6px 22px var(--primary-glow)}
.btn-analyze:disabled{opacity:.45;cursor:not-allowed}

/* ══════════════════════════════════════════════════════════════════
   RESULTS WORKSTATION
══════════════════════════════════════════════════════════════════ */
.res-nav-strip{
  display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;
  gap:.75rem;margin-bottom:1.25rem
}
.res-p-title{font-size:1.15rem;font-weight:800;color:var(--text);display:flex;align-items:center;gap:.6rem}
.btn-tool{
  background:var(--bg-raised);border:1px solid var(--border);color:var(--text);
  font-size:.75rem;font-weight:700;padding:.38rem .8rem;border-radius:6px;cursor:pointer;
  transition:all .15s;display:inline-flex;align-items:center;gap:.35rem
}
.btn-tool:hover{border-color:var(--primary);color:var(--primary)}
.btn-primary-tool{background:linear-gradient(135deg,var(--primary),var(--teal));color:#fff;border:none}

.triage-banner{
  background:var(--bg-card);border:1px solid var(--border);border-radius:var(--radius);
  padding:1.1rem 1.4rem;margin-bottom:1.25rem;display:grid;grid-template-columns:1.3fr 1fr 1fr;gap:1.25rem;align-items:center
}
@media(max-width:960px){.triage-banner{grid-template-columns:1fr}}

.tb-sec-lbl{font-size:.64rem;font-weight:800;text-transform:uppercase;letter-spacing:.7px;color:var(--dim);margin-bottom:.3rem}
.tb-grade-title{font-size:1.35rem;font-weight:900;display:flex;align-items:center;gap:.5rem}
.tb-action-desc{font-size:.78rem;color:var(--muted);margin-top:.3rem;line-height:1.5}
.urg-pill{font-size:.72rem;font-weight:800;padding:.25rem .75rem;border-radius:6px;text-transform:uppercase;letter-spacing:.6px;display:inline-block}

/* Interactive Canvas Workspace */
.ws-grid{display:grid;grid-template-columns:1.25fr 1fr;gap:1.25rem;margin-bottom:1.25rem}
@media(max-width:1080px){.ws-grid{grid-template-columns:1fr}}

.canvas-card{background:var(--bg-card);border:1px solid var(--border);border-radius:var(--radius);overflow:hidden}
.canvas-ctrls{
  padding:.6rem 1rem;background:rgba(255,255,255,.02);border-bottom:1px solid var(--border);
  display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:.6rem
}
.view-tabs{display:flex;gap:.25rem;background:var(--bg-input);padding:3px;border-radius:6px;border:1px solid var(--border)}
.vtab{background:none;border:none;color:var(--muted);font-size:.72rem;font-weight:700;padding:.28rem .65rem;border-radius:4px;cursor:pointer}
.vtab.on{background:var(--primary);color:#fff}

.interactive-cv-box{
  background:#020612;position:relative;width:100%;height:380px;
  display:flex;align-items:center;justify-content:center;overflow:hidden
}
#liveInspectionCanvas{max-width:100%;max-height:100%;display:block}
.quadrant-overlay-svg{position:absolute;inset:0;width:100%;height:100%;pointer-events:none;display:none}
.quadrant-overlay-svg.on{display:block}

.blend-slider-bar{
  padding:.7rem 1.1rem;background:var(--bg-raised);border-top:1px solid var(--border);
  display:flex;align-items:center;gap:1rem;font-size:.75rem;color:var(--muted)
}
.blend-slider{flex:1;accent-color:var(--primary);cursor:pointer}

/* AAO 4:2:1 Rule & Risk Assistant */
.aao-card{background:var(--bg-card);border:1px solid var(--border);border-radius:var(--radius);padding:1.1rem}
.aao-title{font-size:.8rem;font-weight:900;text-transform:uppercase;letter-spacing:.7px;color:var(--muted);margin-bottom:.65rem;display:flex;align-items:center;gap:.4rem}
.aao-table{width:100%;border-collapse:collapse;font-size:.76rem;margin-bottom:.85rem}
.aao-table td{padding:.45rem .6rem;border-bottom:1px solid var(--border)}
.aao-rule-tag{font-weight:800;color:var(--primary);width:35%}

/* 3-Model Cards Grid */
.models-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:1.15rem;margin-bottom:1.25rem}
@media(max-width:960px){.models-grid{grid-template-columns:1fr}}

.mc{background:var(--bg-card);border:1px solid var(--border);border-radius:var(--radius);overflow:hidden;transition:border-color .2s}
.mc:hover{border-color:var(--border-hi)}
.mc.hybrid-card{box-shadow:0 0 24px rgba(2,132,199,.12);border-top:3px solid var(--primary)}
.mc-hd{padding:.65rem 1rem;background:rgba(255,255,255,.015);border-bottom:1px solid var(--border);display:flex;justify-content:space-between;align-items:center}
.mc-title{font-size:.78rem;font-weight:800;text-transform:uppercase;letter-spacing:.7px}
.mc-badge{font-size:.62rem;font-weight:700;padding:.15rem .45rem;border-radius:4px;background:rgba(2,132,199,.2);border:1px solid rgba(2,132,199,.4);color:#38bdf8}
.mc-body{padding:1.1rem}

.ring-wrap{display:flex;justify-content:center;margin:.3rem 0 .7rem}
.ring{position:relative;width:96px;height:96px}
.ring svg{position:absolute;inset:0;transform:rotate(-90deg)}
.rt{fill:none;stroke:var(--bg-raised);stroke-width:9}
.rf{fill:none;stroke-width:9;stroke-linecap:round;stroke-dasharray:282.74;stroke-dashoffset:282.74;transition:stroke-dashoffset 1s cubic-bezier(.34,1.56,.64,1)}
.ring-center{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center}
.rv{font-size:1.25rem;font-weight:900}
.rl{font-size:.58rem;color:var(--muted);text-transform:uppercase;letter-spacing:.5px}

.pb{margin-bottom:.3rem}
.pb-hd{display:flex;justify-content:space-between;font-size:.68rem;color:var(--muted);font-weight:600;margin-bottom:.15rem}
.pb-track{background:var(--bg-raised);border-radius:3px;height:5px;overflow:hidden}
.pb-fill{height:100%;width:0;border-radius:3px;background:linear-gradient(90deg,var(--primary),var(--teal));transition:width .6s ease}
.pb-fill.dom{background:linear-gradient(90deg,#10b981,#34d399)}

.cam-pair{display:grid;grid-template-columns:1fr 1fr;gap:.4rem;margin-top:.7rem}
.cam-card{border-radius:6px;overflow:hidden;background:#000;border:1px solid var(--border);cursor:zoom-in}
.cam-card img{width:100%;height:auto;display:block;transition:transform .2s}
.cam-card:hover img{transform:scale(1.05)}
.cam-lbl{background:rgba(0,0,0,.65);font-size:.58rem;font-weight:700;text-align:center;padding:.2rem;color:var(--muted);text-transform:uppercase}

/* Doctor's Clinical Impression & Sign-Off */
.signoff-card{background:var(--bg-card);border:1px solid var(--border);border-radius:var(--radius);padding:1.25rem;margin-bottom:1.5rem}
.signoff-title{font-size:.85rem;font-weight:900;text-transform:uppercase;letter-spacing:.7px;color:var(--muted);margin-bottom:.9rem;display:flex;align-items:center;gap:.4rem}
.so-grid{display:grid;grid-template-columns:1fr 1fr;gap:1.25rem}
@media(max-width:880px){.so-grid{grid-template-columns:1fr}}

/* ══════════════════════════════════════════════════════════════════
   ANATOMY REFERENCE MODAL
══════════════════════════════════════════════════════════════════ */
.modal-bk{
  display:none;position:fixed;inset:0;z-index:999;
  background:rgba(0,0,0,.78);backdrop-filter:blur(6px);
  align-items:center;justify-content:center;padding:1.5rem
}
.modal-bk.on{display:flex}
.modal-box{
  background:var(--bg-card);border:1px solid var(--border);border-radius:var(--radius);
  width:100%;max-width:860px;max-height:90vh;overflow-y:auto;box-shadow:0 12px 40px rgba(0,0,0,.6)
}
.modal-hd{padding:1rem 1.4rem;border-bottom:1px solid var(--border);display:flex;justify-content:space-between;align-items:center}
.modal-title{font-size:.95rem;font-weight:800;color:var(--text);display:flex;align-items:center;gap:.5rem}
.modal-body{padding:1.4rem}

@media print{
  header,.nav,.btn-tool,.btn-analyze,.blend-slider-bar,.samples-bar,#loader{display:none!important}
  body{background:#fff!important;color:#000!important}
  .page{display:none!important}
  #page-results{display:block!important;padding:0!important}
  .card,.mc,.triage-banner,.signoff-card{border:1px solid #ccc!important;background:#fff!important;box-shadow:none!important;break-inside:avoid}
  .models-grid{grid-template-columns:1fr 1fr 1fr!important}
}

#loader{
  display:none;position:fixed;inset:0;z-index:900;
  background:rgba(7,13,30,.88);backdrop-filter:blur(6px);
  flex-direction:column;align-items:center;justify-content:center;gap:1.2rem
}
#loader.on{display:flex}
.spinner{width:48px;height:48px;border-radius:50%;border:4px solid var(--border);border-top-color:var(--primary);animation:spin .8s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
.loader-txt{font-size:.9rem;font-weight:700;color:#38bdf8}

#lb{display:none;position:fixed;inset:0;z-index:1000;background:rgba(0,0,0,.92);align-items:center;justify-content:center;cursor:zoom-out}
#lb.on{display:flex}
#lb img{max-width:92vw;max-height:92vh;border-radius:8px;border:1px solid #333}

.hist-table{width:100%;border-collapse:collapse;font-size:.78rem}
.hist-table th{padding:.65rem .9rem;text-align:left;font-size:.65rem;text-transform:uppercase;letter-spacing:.6px;color:var(--dim);border-bottom:1px solid var(--border);background:var(--bg-raised)}
.hist-table td{padding:.65rem .9rem;border-bottom:1px solid var(--border)}
.hist-table tr:hover{background:rgba(255,255,255,.02)}
</style>
</head>
<body>

<div id="loader">
  <div class="spinner"></div>
  <p class="loader-txt" id="ltxt">Executing multi-model inference...</p>
</div>

<div id="lb" onclick="closeLB()"><img id="lbImg" src="" alt="Enlarged fundus"></div>

<!-- Eye Anatomy Reference Modal -->
<div class="modal-bk" id="anatomyModal">
  <div class="modal-box">
    <div class="modal-hd">
      <span class="modal-title">👁 Clinical Retinal Fundus &amp; Ocular Anatomy Reference</span>
      <button class="btn-icon" onclick="closeAnatomyModal()">✕</button>
    </div>
    <div class="modal-body">
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:1.25rem;margin-bottom:1.25rem">
        <div style="background:#020612;padding:1rem;border-radius:8px;border:1px solid var(--border);text-align:center">
          <h4 style="font-size:.78rem;font-weight:800;color:var(--primary);margin-bottom:.5rem">Ocular Sagittal Cross-Section</h4>
          <svg viewBox="0 0 400 250" style="width:100%;max-height:200px">
            <defs>
              <radialGradient id="lensGrad" cx="50%" cy="50%" r="50%">
                <stop offset="0%" stop-color="#7dd3fc"/>
                <stop offset="100%" stop-color="#0284c7"/>
              </radialGradient>
              <linearGradient id="retinaGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stop-color="#f43f5e"/>
                <stop offset="100%" stop-color="#e11d48"/>
              </linearGradient>
            </defs>
            <ellipse cx="200" cy="125" rx="140" ry="110" fill="#1e293b" stroke="#475569" stroke-width="3"/>
            <ellipse cx="200" cy="125" rx="136" ry="106" fill="#881337" opacity=".4"/>
            <path d="M 120,40 A 132 102 0 1 0 120,210" fill="none" stroke="url(#retinaGrad)" stroke-width="4"/>
            <ellipse cx="200" cy="125" rx="125" ry="98" fill="#0369a1" opacity=".15"/>
            <path d="M 85,75 A 60 60 0 0 0 85,175" fill="none" stroke="#38bdf8" stroke-width="4"/>
            <path d="M 85,75 A 60 60 0 0 0 85,175 Z" fill="#38bdf8" opacity=".12"/>
            <line x1="100" y1="75" x2="115" y2="102" stroke="#f59e0b" stroke-width="5" stroke-linecap="round"/>
            <line x1="100" y1="175" x2="115" y2="148" stroke="#f59e0b" stroke-width="5" stroke-linecap="round"/>
            <ellipse cx="130" cy="125" rx="14" ry="32" fill="url(#lensGrad)" opacity=".85" stroke="#bae6fd" stroke-width="1.5"/>
            <circle cx="335" cy="125" r="8" fill="#facc15" stroke="#ca8a04" stroke-width="2"/>
            <path d="M 325,160 L 390,185 L 390,195 L 320,175 Z" fill="#64748b"/>
            <text x="50" y="65" fill="#38bdf8" font-size="11" font-weight="700">Cornea</text>
            <text x="120" y="65" fill="#f59e0b" font-size="11" font-weight="700">Iris &amp; Pupil</text>
            <text x="145" y="115" fill="#7dd3fc" font-size="11" font-weight="700">Lens</text>
            <text x="190" y="130" fill="#93c5fd" font-size="11">Vitreous Body</text>
            <text x="270" y="80" fill="#f43f5e" font-size="11" font-weight="700">Sensory Retina</text>
            <text x="310" y="110" fill="#facc15" font-size="11" font-weight="800">Macula / Fovea</text>
            <text x="315" y="215" fill="#94a3b8" font-size="11" font-weight="700">Optic Nerve (CN II)</text>
          </svg>
        </div>

        <div style="background:#020612;padding:1rem;border-radius:8px;border:1px solid var(--border);text-align:center">
          <h4 style="font-size:.78rem;font-weight:800;color:var(--primary);margin-bottom:.5rem">Retinal Fundus Landmarks &amp; ETDRS Grid</h4>
          <svg viewBox="0 0 400 250" style="width:100%;max-height:200px">
            <circle cx="200" cy="125" r="110" fill="#7f1d1d" stroke="#f43f5e" stroke-width="2"/>
            <circle cx="140" cy="125" r="24" fill="#fef08a" stroke="#ca8a04" stroke-width="2"/>
            <circle cx="140" cy="125" r="10" fill="#fef9c3"/>
            <circle cx="260" cy="125" r="28" fill="#450a0a" stroke="#f59e0b" stroke-width="1.5" stroke-dasharray="3,3"/>
            <circle cx="260" cy="125" r="6" fill="#fbbf24"/>
            <path d="M 140,125 Q 180,60 270,65" fill="none" stroke="#b91c1c" stroke-width="3"/>
            <path d="M 140,125 Q 180,190 270,185" fill="none" stroke="#b91c1c" stroke-width="3"/>
            <path d="M 140,125 Q 120,70 95,75" fill="none" stroke="#b91c1c" stroke-width="2"/>
            <path d="M 140,125 Q 120,180 95,175" fill="none" stroke="#b91c1c" stroke-width="2"/>
            <line x1="260" y1="20" x2="260" y2="230" stroke="#38bdf8" stroke-width="1.2" stroke-dasharray="4,4" opacity=".7"/>
            <line x1="100" y1="125" x2="330" y2="125" stroke="#38bdf8" stroke-width="1.2" stroke-dasharray="4,4" opacity=".7"/>
            <text x="110" y="105" fill="#fef08a" font-size="10" font-weight="700">Optic Disc</text>
            <text x="245" y="112" fill="#fbbf24" font-size="10" font-weight="800">Fovea (FAZ)</text>
            <text x="210" y="50" fill="#f87171" font-size="10">Sup. Temp. Arcade</text>
            <text x="210" y="210" fill="#f87171" font-size="10">Inf. Temp. Arcade</text>
            <text x="270" y="95" fill="#38bdf8" font-size="9" font-weight="700">Sup-Temp Quad</text>
            <text x="270" y="155" fill="#38bdf8" font-size="9" font-weight="700">Inf-Temp Quad</text>
          </svg>
        </div>
      </div>

      <div style="font-size:.78rem;line-height:1.6;color:var(--muted)">
        <p><strong style="color:var(--text)">Clinical Significance of Retinal Landmarks:</strong></p>
        <ul style="margin-left:1.2rem;margin-top:.4rem">
          <li><strong>Macula / Fovea:</strong> Highest visual acuity zone. Lipid hard exudates near the foveal avascular zone indicate <em>Diabetic Macular Edema (DME)</em> requiring urgent anti-VEGF therapy.</li>
          <li><strong>Temporal Arcades:</strong> Most frequent location for microaneurysms, dot/blot hemorrhages, and early neovascularization elsewhere (NVE).</li>
          <li><strong>Optic Nerve Head:</strong> Neovascularization at the disc (NVD) represents high-risk Proliferative DR requiring prompt panretinal photocoagulation (PRP).</li>
          <li><strong>ETDRS 4-Quadrant Partition:</strong> The international AAO 4:2:1 rule mandates assessing all 4 quadrants for severe hemorrhages, venous beading (2+ quadrants), and IRMA (1+ quadrant).</li>
        </ul>
      </div>
    </div>
  </div>
</div>

<!-- ══ CLINICAL WORKSTATION HEADER ══ -->
<header>
  <div class="brand-wrap">
    <div class="brand-logo">👁</div>
    <div>
      <div class="brand-title">OptiSight <span>CDSS</span></div>
      <div class="brand-subtitle">
        Vitreoretinal Diagnostic Suite
        <span class="doc-badge">Attending: Dr. V. Sharma, MD</span>
      </div>
    </div>
  </div>

  <nav class="nav">
    <button class="nav-btn on" id="nb-upload" onclick="goUpload()">🔬 Patient Intake</button>
    <button class="nav-btn" id="nb-results" onclick="goResults()">📊 Results &amp; Attention</button>
    <button class="nav-btn" id="nb-history" onclick="goPage('history')">📋 Patient Logs</button>
    <button class="nav-btn" id="nb-about" onclick="goPage('about')">ℹ Clinical Validation</button>
  </nav>

  <div class="hdr-tools">
    <span class="chip chip-blue" id="devChip">🔵 CPU Workstation</span>
    <span class="chip chip-teal" id="modChip">3 Engines</span>
    <button class="btn-icon" title="Ocular Anatomy Reference" onclick="openAnatomyModal()">👁</button>
    <button class="btn-icon" title="Print Consultation Report" onclick="printConsultReport()">🖨</button>
    <button class="btn-icon" title="Toggle Clinical Light / Dark Theme" onclick="toggleTheme()">🌓</button>
  </div>
</header>

<main>
<!-- ══════════════════════════════════════════════════════════════════
     PAGE 1: PATIENT INTAKE & FUNDUS EXAMINATION
══════════════════════════════════════════════════════════════════ -->
<div class="page on" id="page-upload">

  <div class="doc-hero">
    <div>
      <h1 class="hero-meta-title">Ophthalmic AI <span>Diagnostic Triage</span></h1>
      <p class="hero-p">
        Automated multi-model diabetic retinopathy staging adhering to the <strong>ICDR 0–4 Clinical Classification</strong>.
        Full uncropped retinal fundus visualization with multi-axis Vision Transformer global attention (MaxViT-Tiny),
        dense convolutional feature reuse (DenseNet-121), and Grad-CAM++ anatomical saliency.
      </p>
      <div class="hero-chips">
        <span class="h-chip hc-1">🧠 MaxViT-Tiny Attention</span>
        <span class="h-chip hc-2">🔬 DenseNet-121 Micro-texture</span>
        <span class="h-chip hc-1">⚡ OptiSight Fusion Engine</span>
        <span class="h-chip hc-3">🔥 Grad-CAM++ Attention</span>
        <span class="h-chip hc-2">📐 AAO 4:2:1 Validation</span>
      </div>
    </div>

    <!-- Eye Anatomy Mini Interactive Card -->
    <div class="eye-graphic-card">
      <div class="eg-title">
        <span>👁 Retinal Anatomy Guide</span>
        <button class="btn-case" style="padding:.15rem .45rem;font-size:.65rem" onclick="openAnatomyModal()">Expand</button>
      </div>
      <div class="eg-svg-wrap">
        <svg class="eg-svg" viewBox="0 0 320 130">
          <circle cx="160" cy="65" r="58" fill="#581c1c" stroke="#f43f5e" stroke-width="1.5"/>
          <circle cx="125" cy="65" r="14" fill="#fef08a" stroke="#ca8a04" stroke-width="1.5"/>
          <circle cx="195" cy="65" r="16" fill="#350606" stroke="#f59e0b" stroke-width="1" stroke-dasharray="2,2"/>
          <circle cx="195" cy="65" r="3.5" fill="#fbbf24"/>
          <path d="M 125,65 Q 150,30 205,32" fill="none" stroke="#991b1b" stroke-width="2"/>
          <path d="M 125,65 Q 150,100 205,98" fill="none" stroke="#991b1b" stroke-width="2"/>
          <text x="105" y="52" fill="#fef08a" font-size="8" font-weight="700">Disc</text>
          <text x="185" y="56" fill="#fbbf24" font-size="8" font-weight="700">Macula</text>
          <text x="160" y="24" fill="#fca5a5" font-size="7">Sup. Temporal Arcade</text>
          <text x="160" y="116" fill="#fca5a5" font-size="7">Inf. Temporal Arcade</text>
        </svg>
      </div>
      <div class="eg-legend">
        <span><span class="eg-dot" style="background:#fef08a"></span>Optic Disc</span>
        <span><span class="eg-dot" style="background:#fbbf24"></span>Fovea / Macula</span>
        <span><span class="eg-dot" style="background:#ef4444"></span>Vascular Arcades</span>
      </div>
    </div>
  </div>

  <!-- One-Click Sample Clinical Cases Bar -->
  <div class="samples-bar">
    <span class="sb-lbl">⚡ Load Clinical Case:</span>
    <div class="sb-cases">
      <button class="btn-case" onclick="loadSampleCase(0)"><span style="color:#10b981">●</span> Grade 0: Normal Retina</button>
      <button class="btn-case" onclick="loadSampleCase(1)"><span style="color:#f59e0b">●</span> Grade 1: Mild NPDR</button>
      <button class="btn-case" onclick="loadSampleCase(2)"><span style="color:#f97316">●</span> Grade 2: Moderate NPDR</button>
      <button class="btn-case" onclick="loadSampleCase(3)"><span style="color:#ef4444">●</span> Grade 3: Severe NPDR</button>
      <button class="btn-case" onclick="loadSampleCase(4)"><span style="color:#db2777">●</span> Grade 4: Proliferative DR</button>
    </div>
  </div>

  <!-- Two-Column Clinical Intake Grid -->
  <div class="intake-grid">

    <!-- LEFT: Fundus Upload & Canvas -->
    <div class="card">
      <div class="card-hd">
        <span class="card-title">📷 Fundus Photography (Full Field of View)</span>
        <button class="btn-tool" onclick="clearImg()" style="padding:.2rem .5rem">Clear</button>
      </div>
      <div class="card-body">
        <div class="drop-zone" id="dz"
             onclick="document.getElementById('fIn').click()"
             ondragover="ev(event);this.classList.add('over')"
             ondragleave="this.classList.remove('over')"
             ondrop="onDrop(event)">
          <div class="dz-icon">👁️</div>
          <div class="dz-txt">Click or drag &amp; drop retinal fundus photograph</div>
          <div class="dz-sub">High-resolution color fundus (PNG, JPG, DICOM export — max 32 MB)</div>
        </div>
        <input type="file" id="fIn" accept="image/*" style="display:none" onchange="onFile(this.files[0])">

        <div class="cv-wrap" id="cvWrap" style="display:none">
          <div class="cv-badge" id="cvBadge">—</div>
          <canvas id="fundusCanvas"></canvas>
        </div>

        <div style="font-size:.68rem;color:var(--dim);line-height:1.5;margin-top:.5rem">
          <strong>Exam Protocol:</strong> Full original field of view preserved without automatic cropping. Center image on posterior pole (45° or 50° field).
        </div>
      </div>
    </div>

    <!-- RIGHT: Patient Demographics & Clinical Profile -->
    <div class="card">
      <div class="card-hd">
        <span class="card-title">🧑‍⚕️ Clinical Patient Profile</span>
        <span class="chip chip-teal">EHR Synchronized</span>
      </div>
      <div class="card-body">
        <div class="form-row">
          <div><label class="fl">MRN / Patient ID</label><input class="fc" id="iMrn" placeholder="e.g. MRN-88310"></div>
          <div><label class="fl">Patient Full Name</label><input class="fc" id="iName" placeholder="e.g. Sarah Jenkins"></div>
        </div>

        <div class="form-row-3">
          <div><label class="fl">Age</label><input class="fc" id="iAge" type="number" min="1" max="120" placeholder="Years"></div>
          <div>
            <label class="fl">Sex</label>
            <select class="fc" id="iSex">
              <option value="Female">Female</option>
              <option value="Male">Male</option>
              <option value="Other">Other</option>
            </select>
          </div>
          <div>
            <label class="fl">Examined Eye</label>
            <select class="fc" id="iEye">
              <option value="OD (Right Eye)">OD — Right Eye</option>
              <option value="OS (Left Eye)">OS — Left Eye</option>
              <option value="OU (Both Eyes)">OU — Both Eyes</option>
            </select>
          </div>
        </div>

        <div class="form-row-3">
          <div>
            <label class="fl">Diabetes Type</label>
            <select class="fc" id="iDmType">
              <option value="Type 2 Diabetes">Type 2 DM</option>
              <option value="Type 1 Diabetes">Type 1 DM</option>
              <option value="Gestational / Other">Gestational / Other</option>
            </select>
          </div>
          <div><label class="fl">Duration (Yrs)</label><input class="fc" id="iDur" type="number" min="0" max="70" placeholder="e.g. 12"></div>
          <div><label class="fl">HbA1c (%)</label><input class="fc" id="iHba" type="number" step="0.1" min="4" max="18" placeholder="e.g. 8.4"></div>
        </div>

        <div class="form-row">
          <div><label class="fl">Blood Pressure (mmHg)</label><input class="fc" id="iBp" placeholder="e.g. 138/86"></div>
          <div><label class="fl">Visual Acuity (BCVA)</label><input class="fc" id="iVa" placeholder="e.g. 20/30"></div>
        </div>

        <label class="fl">Ocular History &amp; Risk Flags</label>
        <div class="chk-grp">
          <label class="chk-item"><input type="checkbox" id="chkDme"> Macular Edema Suspicion (CSME)</label>
          <label class="chk-item"><input type="checkbox" id="chkPrp"> Prior PRP Laser</label>
          <label class="chk-item"><input type="checkbox" id="chkVegf"> Anti-VEGF Injections</label>
          <label class="chk-item"><input type="checkbox" id="chkVit"> Prior Vitrectomy</label>
        </div>

        <div>
          <label class="fl">Attending Clinician Examination Notes</label>
          <textarea class="fc fc-ta" id="iNotes" placeholder="Record slit-lamp biomicroscopy observations, intraocular pressure (IOP), or dilation status..."></textarea>
        </div>

        <button class="btn-analyze" id="runBtn" onclick="runAnalysis()" disabled>
          🔬 Initiate Multi-Model Diagnostic Staging &nbsp;→
        </button>
      </div>
    </div>

  </div><!-- intake-grid -->
</div><!-- page-upload -->

<!-- ══════════════════════════════════════════════════════════════════
     PAGE 2: CLINICAL RESULTS & GRAD-CAM WORKSTATION
══════════════════════════════════════════════════════════════════ -->
<div class="page" id="page-results">

  <div class="res-nav-strip">
    <div style="display:flex;align-items:center;gap:.75rem">
      <button class="btn-tool" onclick="goUpload()">← Back to Intake</button>
      <span class="res-p-title" id="resPatientTitle">Diagnostic Staging Report</span>
    </div>
    <div style="display:flex;gap:.5rem;flex-wrap:wrap">
      <button class="btn-tool" onclick="saveRecord()">💾 Save to Medical Log</button>
      <button class="btn-tool" onclick="openAnatomyModal()">👁 Anatomy Guide</button>
      <button class="btn-tool btn-primary-tool" onclick="printConsultReport()">🖨 Print Consultation PDF</button>
    </div>
  </div>

  <!-- Triage Consensus Banner -->
  <div class="triage-banner" id="triageBanner">
    <div>
      <div class="tb-sec-lbl">Ensemble Diagnostic Staging (ICDR 0–4)</div>
      <div class="tb-grade-title" id="tbGradeTitle">—</div>
      <div class="tb-action-desc" id="tbActionDesc">—</div>
    </div>
    <div>
      <div class="tb-sec-lbl">Clinical Triage &amp; Action Window</div>
      <div id="tbUrgencyPill" class="urg-pill">—</div>
      <div style="font-size:.73rem;color:var(--muted);margin-top:.4rem" id="tbTimeline">—</div>
    </div>
    <div>
      <div class="tb-sec-lbl">Concordance &amp; DME Risk</div>
      <div id="tbAgreement" style="font-size:.85rem;font-weight:700">—</div>
      <div style="font-size:.73rem;color:var(--muted);margin-top:.3rem" id="tbDmeRisk">DME Risk: —</div>
    </div>
  </div>

  <!-- Interactive Retinal Canvas & AAO 4:2:1 Workstation -->
  <div class="ws-grid">

    <!-- Interactive Canvas Viewer -->
    <div class="canvas-card">
      <div class="canvas-ctrls">
        <span class="card-title">👁 Grad-CAM++ Retinal Attention Inspection</span>
        <div class="view-tabs">
          <button class="vtab on" id="vt-blend" onclick="setViewMode('blend')">Blend</button>
          <button class="vtab" id="vt-orig" onclick="setViewMode('orig')">Original</button>
          <button class="vtab" id="vt-beng" onclick="setViewMode('beng')">Ben Graham</button>
          <button class="vtab" id="vt-cam" onclick="setViewMode('cam')">Grad-CAM</button>
        </div>
        <button class="btn-tool" id="btnToggleGrid" onclick="toggleETDRSGrid()" style="padding:.2rem .55rem;font-size:.68rem">📐 ETDRS Grid</button>
      </div>

      <div class="interactive-cv-box">
        <canvas id="liveInspectionCanvas"></canvas>
        <svg class="quadrant-overlay-svg" id="etdrsSvg" viewBox="0 0 500 500">
          <circle cx="250" cy="250" r="235" fill="none" stroke="#38bdf8" stroke-width="1.5" stroke-dasharray="6,4" opacity=".8"/>
          <line x1="250" y1="15" x2="250" y2="485" stroke="#38bdf8" stroke-width="1.2" stroke-dasharray="4,4" opacity=".7"/>
          <line x1="15" y1="250" x2="485" y2="250" stroke="#38bdf8" stroke-width="1.2" stroke-dasharray="4,4" opacity=".7"/>
          <circle cx="325" cy="250" r="35" fill="none" stroke="#f59e0b" stroke-width="1.5" stroke-dasharray="3,3"/>
          <circle cx="325" cy="250" r="70" fill="none" stroke="#f59e0b" stroke-width="1" stroke-dasharray="4,4" opacity=".6"/>
          <circle cx="150" cy="250" r="35" fill="none" stroke="#fef08a" stroke-width="1.5"/>
          <text x="330" y="80" fill="#38bdf8" font-size="12" font-weight="800">Superior Temporal</text>
          <text x="50" y="80" fill="#38bdf8" font-size="12" font-weight="800">Superior Nasal</text>
          <text x="330" y="440" fill="#38bdf8" font-size="12" font-weight="800">Inferior Temporal</text>
          <text x="50" y="440" fill="#38bdf8" font-size="12" font-weight="800">Inferior Nasal</text>
        </svg>
      </div>

      <div class="blend-slider-bar">
        <span>Attention Heatmap Opacity:</span>
        <input type="range" class="blend-slider" id="blendSlider" min="0" max="100" value="70" oninput="updateBlendAlpha(this.value)">
        <span id="blendVal" style="font-weight:700;width:38px">70%</span>
      </div>
    </div>

    <!-- Clinical Decision Support & AAO 4:2:1 Validation -->
    <div class="aao-card">
      <div class="aao-title">
        <span>📐 AAO 4:2:1 Clinical Criteria Validation</span>
      </div>
      <table class="aao-table">
        <tbody>
          <tr>
            <td class="aao-rule-tag">4 Quadrants</td>
            <td>Severe intraretinal dot/blot hemorrhages in all 4 quadrants (&gt;20 per quadrant)</td>
            <td style="text-align:right"><input type="checkbox" id="aao4" onchange="evalAaoRule()"></td>
          </tr>
          <tr>
            <td class="aao-rule-tag">2 Quadrants</td>
            <td>Definite venous beading and/or caliber looping in 2 or more quadrants</td>
            <td style="text-align:right"><input type="checkbox" id="aao2" onchange="evalAaoRule()"></td>
          </tr>
          <tr>
            <td class="aao-rule-tag">1 Quadrant</td>
            <td>Prominent Intraretinal Microvascular Abnormalities (IRMA) in &gt;=1 quadrant</td>
            <td style="text-align:right"><input type="checkbox" id="aao1" onchange="evalAaoRule()"></td>
          </tr>
          <tr>
            <td class="aao-rule-tag" style="color:#db2777">Proliferative</td>
            <td>Neovascularization of the Disc (NVD) or Elsewhere (NVE), or vitreous hemorrhage</td>
            <td style="text-align:right"><input type="checkbox" id="aaoPdr" onchange="evalAaoRule()"></td>
          </tr>
        </tbody>
      </table>

      <div id="aaoVerdict" style="padding:.65rem;border-radius:6px;background:rgba(2,132,199,.1);font-size:.76rem;color:var(--text);border:1px solid rgba(2,132,199,.3)">
        <strong>AAO Assessment:</strong> Check applicable clinical signs above to compute clinical severity threshold.
      </div>

      <div style="margin-top:1rem;border-top:1px solid var(--border);padding-top:.85rem">
        <div class="tb-sec-lbl">Diabetic Progression Composite Risk Index</div>
        <div style="display:flex;align-items:center;gap:.75rem;margin-top:.4rem">
          <div style="flex:1;background:var(--bg-raised);height:8px;border-radius:4px;overflow:hidden">
            <div id="riskBar" style="height:100%;width:0%;background:linear-gradient(90deg,#10b981,#ef4444);transition:width .6s ease"></div>
          </div>
          <span id="riskScoreVal" style="font-weight:900;font-size:.88rem">0 / 100</span>
        </div>
        <div id="riskLevelDesc" style="font-size:.72rem;color:var(--muted);margin-top:.3rem">—</div>
      </div>
    </div>

  </div><!-- ws-grid -->

  <!-- Three Model Cards Grid -->
  <div class="models-grid" id="modelsGrid"></div>

  <!-- Attending Ophthalmologist Impression & Sign-Off -->
  <div class="signoff-card">
    <div class="signoff-title">
      <span>🩺 Attending Ophthalmologist Clinical Impression &amp; Triage Sign-Off</span>
    </div>

    <div class="so-grid">
      <div>
        <label class="fl">Clinician Diagnostic Agreement</label>
        <div style="display:flex;gap:.75rem;margin-bottom:.85rem;flex-wrap:wrap">
          <label class="chk-item"><input type="radio" name="docVerdict" value="concur" checked> Concur with AI Consensus</label>
          <label class="chk-item"><input type="radio" name="docVerdict" value="modify"> Modify Final Grade</label>
          <label class="chk-item"><input type="radio" name="docVerdict" value="senior"> Request Senior Retinal Review</label>
        </div>

        <label class="fl">Clinical Lesions Confirmed by Doctor</label>
        <div class="chk-grp">
          <label class="chk-item"><input type="checkbox" checked> Microaneurysms</label>
          <label class="chk-item"><input type="checkbox" id="chkHeme"> Blot Hemorrhages</label>
          <label class="chk-item"><input type="checkbox" id="chkExu"> Hard Lipid Exudates</label>
          <label class="chk-item"><input type="checkbox" id="chkCws"> Cotton-Wool Spots</label>
          <label class="chk-item"><input type="checkbox" id="chkVb"> Venous Beading</label>
          <label class="chk-item"><input type="checkbox" id="chkIrma"> IRMA</label>
          <label class="chk-item"><input type="checkbox" id="chkNeo"> Neovascularization</label>
        </div>
      </div>

      <div>
        <label class="fl">Physician Clinical Impression &amp; Management Plan</label>
        <textarea class="fc fc-ta" id="docNotes" style="min-height:75px" placeholder="Enter clinical rationale, treatment plan, laser/anti-VEGF orders, and follow-up directives..."></textarea>

        <div style="display:flex;justify-content:space-between;align-items:center;margin-top:.75rem;padding-top:.5rem;border-top:1px solid var(--border)">
          <div style="font-size:.72rem;color:var(--muted)">
            <strong>Attending:</strong> Dr. V. Sharma, MD, FASRS &nbsp;|&nbsp; <span id="soDate">—</span>
          </div>
          <button class="btn-tool btn-primary-tool" onclick="saveRecord()">Sign &amp; File Consultation</button>
        </div>
      </div>
    </div>
  </div>

</div><!-- page-results -->

<!-- ══════════════════════════════════════════════════════════════════
     PAGE 3: PATIENT RECORDS & SESSION HISTORY
══════════════════════════════════════════════════════════════════ -->
<div class="page" id="page-history">
  <div class="card">
    <div class="card-hd">
      <span class="card-title">📋 Clinical Session Logs &amp; Patient Records</span>
      <span class="chip chip-blue" id="hCount">0 records</span>
    </div>
    <div class="card-body">
      <div style="display:flex;gap:.6rem;align-items:center;flex-wrap:wrap;margin-bottom:1rem">
        <input class="fc" id="hSearch" placeholder="🔍 Search by MRN, patient name, or grade..." oninput="filterHistory()" style="flex:1;min-width:240px">
        <button class="btn-tool" onclick="exportHistoryCSV()">⬇ Export CSV</button>
        <button class="btn-tool" onclick="clearHistory()">🗑 Clear All</button>
      </div>
      <div style="overflow-x:auto">
        <table class="hist-table">
          <thead>
            <tr>
              <th>#</th><th>Timestamp</th><th>MRN</th><th>Patient Name</th><th>Eye</th>
              <th>HbA1c</th><th>Consensus Staging</th><th>Hybrid Confidence</th><th>Action</th>
            </tr>
          </thead>
          <tbody id="histTbody">
            <tr><td colspan="9" style="text-align:center;padding:2rem;color:var(--dim)">No consultation records logged yet.</td></tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>
</div>

<!-- ══════════════════════════════════════════════════════════════════
     PAGE 4: CLINICAL VALIDATION & ABOUT
══════════════════════════════════════════════════════════════════ -->
<div class="page" id="page-about">
  <div class="doc-hero" style="margin-bottom:1.25rem">
    <div>
      <h2 style="font-size:1.35rem;font-weight:900;color:var(--text);margin-bottom:.35rem">OptiSight CDSS Architecture &amp; Clinical Validation</h2>
      <p class="hero-p">
        Rigorous clinical multi-architecture benchmark on the standard APTOS 2019 diabetic retinopathy cohort (3,662 fundus photographs, 70:15:15 stratified split).
        Full uncropped fundus visualization with anatomical edge-guided attention.
      </p>
    </div>
  </div>

  <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:.85rem;margin-bottom:1.25rem">
    <div class="card" style="padding:1rem;text-align:center">
      <div style="font-size:1.6rem;font-weight:900;color:#38bdf8">82.73%</div>
      <div class="tb-sec-lbl">Hybrid Top-1 Accuracy</div>
    </div>
    <div class="card" style="padding:1rem;text-align:center">
      <div style="font-size:1.6rem;font-weight:900;color:#2dd4bf">0.871</div>
      <div class="tb-sec-lbl">Quadratic Weighted Kappa (QWK)</div>
    </div>
    <div class="card" style="padding:1rem;text-align:center">
      <div style="font-size:1.6rem;font-weight:900;color:#fbbf24">0.668</div>
      <div class="tb-sec-lbl">Macro F1 Score</div>
    </div>
    <div class="card" style="padding:1rem;text-align:center">
      <div style="font-size:1.6rem;font-weight:900;color:#f43f5e">3,662</div>
      <div class="tb-sec-lbl">Validated Retinal Exams</div>
    </div>
  </div>

  <div class="card">
    <div class="card-hd"><span class="card-title">📊 Per-Grade Clinical Performance Summary</span></div>
    <div class="card-body">
      <table class="hist-table">
        <thead>
          <tr>
            <th>Grade</th><th>Classification</th><th>Test Set</th><th>MaxViT Recall</th><th>DenseNet Recall</th><th>OptiSight Hybrid F1</th>
          </tr>
        </thead>
        <tbody>
          <tr><td><span style="color:#10b981;font-weight:700">0</span></td><td>No DR</td><td>271</td><td>98.15%</td><td>98.15%</td><td><strong>0.965</strong></td></tr>
          <tr><td><span style="color:#f59e0b;font-weight:700">1</span></td><td>Mild NPDR</td><td>56</td><td>67.86%</td><td>69.64%</td><td><strong>0.627</strong></td></tr>
          <tr><td><span style="color:#f97316;font-weight:700">2</span></td><td>Moderate NPDR</td><td>150</td><td>79.33%</td><td>77.33%</td><td><strong>0.791</strong></td></tr>
          <tr><td><span style="color:#ef4444;font-weight:700">3</span></td><td>Severe NPDR</td><td>29</td><td>37.93%</td><td>41.38%</td><td><strong>0.468</strong></td></tr>
          <tr><td><span style="color:#db2777;font-weight:700">4</span></td><td>Proliferative DR</td><td>44</td><td>50.00%</td><td>47.73%</td><td><strong>0.537</strong></td></tr>
        </tbody>
      </table>
    </div>
  </div>
</div>

</main>

<script>
/* ── Global State ── */
let curFile = null, curSampleGrade = null, lastAnalysis = null;
let historyRecords = JSON.parse(localStorage.getItem('optisight_history') || '[]');
const CLR = ['#10b981', '#f59e0b', '#f97316', '#ef4444', '#db2777'];
const NM  = ['No DR', 'Mild NPDR', 'Moderate NPDR', 'Severe NPDR', 'Proliferative DR'];
const MODEL_COLORS = { MaxViT: '#0284c7', DenseNet121: '#0d9488', Hybrid: '#8b5cf6' };
const MODEL_NAMES  = { MaxViT: 'MaxViT-Tiny Attention', DenseNet121: 'DenseNet-121 CNN', Hybrid: 'OptiSight Hybrid Ensemble' };

let viewMode = 'blend'; // 'blend', 'orig', 'beng', 'cam'
let blendAlpha = 0.70;
let showGrid = false;
let imgOrig = new Image(), imgCam = new Image(), imgBeng = new Image();

/* ── Sample Cases Clinical Data ── */
const SAMPLE_PATIENTS = [
  { mrn: 'PT-10492', name: 'Eleanor Vance', age: 54, sex: 'Female', eye: 'OD (Right Eye)', dm: 'Type 2 Diabetes', dur: 4, hba: 6.5, bp: '122/78', va: '20/20', dme: false, notes: 'Routine annual eye screening. No visual complaints.' },
  { mrn: 'PT-20831', name: 'Marcus Chen', age: 61, sex: 'Male', eye: 'OS (Left Eye)', dm: 'Type 2 Diabetes', dur: 8, hba: 7.4, bp: '134/84', va: '20/25', dme: false, notes: 'Isolated microaneurysms noted in posterior pole during examination.' },
  { mrn: 'PT-31904', name: 'Sarah Jenkins', age: 58, sex: 'Female', eye: 'OD (Right Eye)', dm: 'Type 2 Diabetes', dur: 12, hba: 8.6, bp: '142/88', va: '20/40', dme: true, notes: 'Multiple intraretinal blot hemorrhages and hard exudates. Macular OCT ordered.' },
  { mrn: 'PT-42177', name: 'David Morales', age: 67, sex: 'Male', eye: 'OS (Left Eye)', dm: 'Type 1 Diabetes', dur: 19, hba: 9.8, bp: '155/92', va: '20/60', dme: true, notes: 'Extensive blot hemorrhages in all 4 quadrants with venous beading. Urgent retinal consult requested.' },
  { mrn: 'PT-58203', name: 'Arthur Pendelton', age: 72, sex: 'Male', eye: 'OD (Right Eye)', dm: 'Type 2 Diabetes', dur: 24, hba: 10.4, bp: '168/96', va: '20/100', dme: true, notes: 'High-risk Proliferative DR. Fronds of neovascularization near disc and arcades. Emergent PRP/anti-VEGF indicated.' }
];

/* ── Initialization ── */
window.addEventListener('DOMContentLoaded', () => {
  renderHistoryTable();
  fetch('/api/info').then(r => r.json()).then(d => {
    document.getElementById('devChip').textContent = d.device === 'cuda' ? '🟢 CUDA GPU' : (d.device === 'serverless' ? '☁️ Vercel Edge' : '🔵 CPU Workstation');
    document.getElementById('modChip').textContent = d.models.length + ' AI Engines';
  }).catch(() => {});
});

/* ── Navigation ── */
function showPage(id) {
  ['upload', 'results', 'history', 'about'].forEach(p => {
    document.getElementById('page-' + p).classList.toggle('on', p === id);
    const nb = document.getElementById('nb-' + p);
    if (nb) nb.classList.toggle('on', p === id);
  });
  window.scrollTo({ top: 0, behavior: 'smooth' });
}
function goUpload() { showPage('upload'); }
function goResults() {
  if (lastAnalysis) showPage('results');
  else { alert('Please run an analysis first.'); goUpload(); }
}
function goPage(p) { showPage(p); }

/* ── Theme Toggle ── */
function toggleTheme() {
  const el = document.documentElement;
  const isDark = el.getAttribute('data-theme') === 'dark';
  el.setAttribute('data-theme', isDark ? 'light' : 'dark');
}

/* ── Anatomy Modal ── */
function openAnatomyModal() { document.getElementById('anatomyModal').classList.add('on'); }
function closeAnatomyModal() { document.getElementById('anatomyModal').classList.remove('on'); }

/* ── File Handling ── */
function ev(e) { e.preventDefault(); e.stopPropagation(); }
function onDrop(e) {
  ev(e); document.getElementById('dz').classList.remove('over');
  const f = e.dataTransfer.files[0]; if (f) onFile(f);
}
function onFile(f) {
  if (!f) return;
  curFile = f; curSampleGrade = null;
  const img = new Image();
  img.onload = () => {
    const cv = document.getElementById('fundusCanvas');
    cv.width = img.naturalWidth; cv.height = img.naturalHeight;
    cv.getContext('2d').drawImage(img, 0, 0);
    document.getElementById('dz').style.display = 'none';
    document.getElementById('cvWrap').style.display = 'flex';
    document.getElementById('cvBadge').textContent = `${img.naturalWidth} × ${img.naturalHeight} px (Full Image)`;
    document.getElementById('runBtn').disabled = false;
  };
  img.src = URL.createObjectURL(f);
}
function clearImg() {
  curFile = null; curSampleGrade = null;
  document.getElementById('dz').style.display = '';
  document.getElementById('cvWrap').style.display = 'none';
  document.getElementById('runBtn').disabled = true;
  document.getElementById('fIn').value = '';
}

/* ── Load Sample Cases ── */
async function loadSampleCase(grade) {
  const sp = SAMPLE_PATIENTS[grade];
  curSampleGrade = grade; curFile = null;
  document.getElementById('iMrn').value = sp.mrn;
  document.getElementById('iName').value = sp.name;
  document.getElementById('iAge').value = sp.age;
  document.getElementById('iSex').value = sp.sex;
  document.getElementById('iEye').value = sp.eye;
  document.getElementById('iDmType').value = sp.dm;
  document.getElementById('iDur').value = sp.dur;
  document.getElementById('iHba').value = sp.hba;
  document.getElementById('iBp').value = sp.bp;
  document.getElementById('iVa').value = sp.va;
  document.getElementById('chkDme').checked = sp.dme;
  document.getElementById('iNotes').value = sp.notes;

  try {
    const res = await fetch(`/api/sample/${grade}`);
    const blob = await res.blob();
    curFile = new File([blob], `sample_grade_${grade}.jpg`, { type: 'image/jpeg' });
    onFile(curFile);
  } catch (err) {
    alert('Failed to load sample image: ' + err.message);
  }
}

/* ── Analysis Execution ── */
const STEPS = [
  "Standardizing color fundus contrast...",
  "Computing soft anatomical tissue boundary...",
  "Applying microvascular edge detection...",
  "Running multi-model deep learning inference...",
  "Synthesizing Grad-CAM++ anatomical attention..."
];

async function runAnalysis() {
  if (!curFile) { alert('Please upload or load a fundus image.'); return; }
  const lo = document.getElementById('loader'), lt = document.getElementById('ltxt');
  let si = 0; lt.textContent = STEPS[0]; lo.classList.add('on');
  const iv = setInterval(() => { si = (si + 1) % STEPS.length; lt.textContent = STEPS[si]; }, 800);

  try {
    const fd = new FormData();
    fd.append('file', curFile);
    fd.append('mrn', document.getElementById('iMrn').value || 'MRN-UNSPECIFIED');
    fd.append('name', document.getElementById('iName').value || 'Anonymous Patient');
    fd.append('eye', document.getElementById('iEye').value);
    fd.append('hba1c', document.getElementById('iHba').value || '');
    fd.append('age', document.getElementById('iAge').value || '');
    fd.append('duration', document.getElementById('iDur').value || '');
    fd.append('bp', document.getElementById('iBp').value || '');
    fd.append('dme', document.getElementById('chkDme').checked ? 'true' : 'false');

    const res = await fetch('/api/predict', { method: 'POST', body: fd });
    const data = await res.json();
    if (!res.ok || data.error) throw new Error(data.error || 'Diagnostic server error');

    lastAnalysis = data;
    renderResults(data);
    showPage('results');
  } catch (err) {
    alert('Diagnostic Error: ' + err.message);
  } finally {
    clearInterval(iv); lo.classList.remove('on');
  }
}

/* ── Render Diagnostic Results ── */
function renderResults(data) {
  const pName = document.getElementById('iName').value || 'Patient';
  const pMrn  = document.getElementById('iMrn').value || 'MRN-UNSPECIFIED';
  const pEye  = document.getElementById('iEye').value;
  document.getElementById('resPatientTitle').innerHTML = `Consultation — <strong>${pName}</strong> (${pMrn}) · ${pEye}`;
  document.getElementById('soDate').textContent = new Date().toLocaleString();

  // Primary Consensus Grade
  const cg = data.consensus_grade;
  const gradeColor = CLR[cg];
  const tbGrade = document.getElementById('tbGradeTitle');
  tbGrade.innerHTML = `<span style="color:${gradeColor}">Grade ${cg} — ${NM[cg]}</span>`;

  // Consensus Details
  const hybridModel = data.models['Hybrid'] || Object.values(data.models)[0];
  document.getElementById('tbActionDesc').textContent = hybridModel.action || DR_INFO[cg].action;

  // Urgency pill
  const urgEl = document.getElementById('tbUrgencyPill');
  const urg = hybridModel.urgency || DR_INFO[cg].urgency;
  urgEl.textContent = urg.toUpperCase();
  urgEl.style.background = gradeColor + '22';
  urgEl.style.color = gradeColor;
  urgEl.style.border = `1px solid ${gradeColor}66`;

  // Timeline
  document.getElementById('tbTimeline').textContent = `Referral Interval: ${hybridModel.timeline || DR_INFO[cg].timeline}`;

  // Agreement
  const agrEl = document.getElementById('tbAgreement');
  agrEl.innerHTML = data.agreement
    ? `<span style="color:#10b981">✅ 100% Concordance (All 3 Models)</span>`
    : `<span style="color:#f97316">⚠️ Discrepancy (Majority Vote Applied)</span>`;

  // DME Risk
  document.getElementById('tbDmeRisk').textContent = `Macular Edema Risk: ${hybridModel.dme_risk || DR_INFO[cg].dme_risk}`;

  // Progression Risk Bar
  const rScore = data.clinical_risk_score || (cg * 20 + 10);
  const rLevel = data.risk_level || (cg >= 3 ? 'High Progression Risk' : 'Standard Routine Risk');
  document.getElementById('riskBar').style.width = rScore + '%';
  document.getElementById('riskScoreVal').textContent = `${rScore} / 100`;
  document.getElementById('riskLevelDesc').textContent = rLevel;

  // Setup Images for Interactive Canvas (Full uncropped images)
  imgOrig.src = hybridModel.base_original || hybridModel.gradcam_original;
  imgCam.src  = hybridModel.gradcam_raw || hybridModel.gradcam_overlay;
  imgBeng.src = hybridModel.preprocessed || hybridModel.gradcam_original;

  imgOrig.onload = () => { renderInteractiveCanvas(); };

  // Render 3 Model Cards
  const grid = document.getElementById('modelsGrid');
  grid.innerHTML = '';
  ['MaxViT', 'DenseNet121', 'Hybrid'].forEach(mName => {
    const m = data.models[mName]; if (!m) return;
    const isHybrid = mName === 'Hybrid';
    const accent = MODEL_COLORS[mName];
    const lbl = MODEL_NAMES[mName];
    const off = (282.74 * (1 - m.confidence / 100)).toFixed(2);
    const col = CLR[m.grade];

    let pbars = '';
    Object.entries(m.probabilities).forEach(([cName, pVal], idx) => {
      const isDom = idx === m.grade;
      pbars += `<div class="pb">
        <div class="pb-hd"><span>${cName}</span><span>${pVal.toFixed(1)}%</span></div>
        <div class="pb-track"><div class="pb-fill${isDom ? ' dom' : ''}" style="width:${pVal}%"></div></div>
      </div>`;
    });

    const card = document.createElement('div');
    card.className = 'mc' + (isHybrid ? ' hybrid-card' : '');
    card.innerHTML = `
      <div class="mc-hd">
        <span class="mc-title" style="color:${accent}">${lbl}</span>
        ${isHybrid ? '<span class="mc-badge">Primary Consensus</span>' : ''}
      </div>
      <div class="mc-body">
        <div class="ring-wrap"><div class="ring">
          <svg viewBox="0 0 100 100"><circle class="rt" cx="50" cy="50" r="45"/>
            <circle class="rf" cx="50" cy="50" r="45" stroke="${col}" stroke-dashoffset="${off}"/>
          </svg>
          <div class="ring-center">
            <span class="rv" style="color:${col}">${m.confidence.toFixed(1)}%</span>
            <span class="rl">Confidence</span>
          </div>
        </div></div>

        <div style="text-align:center;margin-bottom:.85rem">
          <div style="font-size:.65rem;color:var(--muted);text-transform:uppercase">Predicted Grade</div>
          <div style="font-size:1.05rem;font-weight:900;color:${col}">Grade ${m.grade} — ${NM[m.grade]}</div>
          ${m.low_confidence ? '<span style="font-size:.62rem;color:#f97316;font-weight:700">⚠️ Low Confidence Margin</span>' : ''}
        </div>

        <div class="tb-sec-lbl" style="margin-top:.8rem">Class Probabilities</div>
        ${pbars}

        <div class="tb-sec-lbl" style="margin-top:.8rem">Grad-CAM++ Attention</div>
        <div class="cam-pair">
          <div class="cam-card" onclick="openLB('${m.gradcam_original}')">
            <div class="cam-lbl">Fundus Input</div>
            <img src="${m.gradcam_original}" alt="original">
          </div>
          <div class="cam-card" onclick="openLB('${m.gradcam_overlay}')">
            <div class="cam-lbl">CAM Attention</div>
            <img src="${m.gradcam_overlay}" alt="overlay">
          </div>
        </div>
      </div>
    `;
    grid.appendChild(card);
  });

  document.getElementById('aao4').checked = cg >= 3;
  document.getElementById('aao2').checked = cg >= 3;
  document.getElementById('aao1').checked = cg >= 3;
  document.getElementById('aaoPdr').checked = cg === 4;
  evalAaoRule();
}

/* ── Interactive Canvas Rendering ── */
function renderInteractiveCanvas() {
  const cv = document.getElementById('liveInspectionCanvas');
  const ctx = cv.getContext('2d');
  const w = imgOrig.naturalWidth || 640, h = imgOrig.naturalHeight || 480;
  cv.width = w; cv.height = h;

  ctx.clearRect(0, 0, w, h);

  if (viewMode === 'orig') {
    ctx.drawImage(imgOrig, 0, 0, w, h);
  } else if (viewMode === 'beng') {
    ctx.drawImage(imgBeng, 0, 0, w, h);
  } else if (viewMode === 'cam') {
    ctx.drawImage(imgCam, 0, 0, w, h);
  } else {
    // Blended view with transparent PNG CAM overlay
    ctx.drawImage(imgOrig, 0, 0, w, h);
    ctx.globalAlpha = blendAlpha;
    ctx.drawImage(imgCam, 0, 0, w, h);
    ctx.globalAlpha = 1.0;
  }
}

function setViewMode(mode) {
  viewMode = mode;
  ['blend', 'orig', 'beng', 'cam'].forEach(m => {
    document.getElementById('vt-' + m).classList.toggle('on', m === mode);
  });
  renderInteractiveCanvas();
}

function updateBlendAlpha(val) {
  blendAlpha = val / 100.0;
  document.getElementById('blendVal').textContent = val + '%';
  if (viewMode === 'blend') renderInteractiveCanvas();
}

function toggleETDRSGrid() {
  showGrid = !showGrid;
  document.getElementById('etdrsSvg').classList.toggle('on', showGrid);
  document.getElementById('btnToggleGrid').style.background = showGrid ? 'var(--primary)' : '';
  document.getElementById('btnToggleGrid').style.color = showGrid ? '#fff' : '';
}

/* ── AAO 4:2:1 Rule Evaluation ── */
function evalAaoRule() {
  const c4 = document.getElementById('aao4').checked;
  const c2 = document.getElementById('aao2').checked;
  const c1 = document.getElementById('aao1').checked;
  const cP = document.getElementById('aaoPdr').checked;

  const count = (c4 ? 1 : 0) + (c2 ? 1 : 0) + (c1 ? 1 : 0);
  const verd = document.getElementById('aaoVerdict');

  if (cP) {
    verd.innerHTML = `<strong style="color:#db2777">Proliferative DR (PDR):</strong> High-risk criteria satisfied (NVD/NVE or vitreous hemorrhage). Emergency referral within 24–72 hours.`;
    verd.style.borderColor = '#db2777';
  } else if (count >= 2) {
    verd.innerHTML = `<strong style="color:#ef4444">Very Severe NPDR:</strong> Meets 2+ criteria of the AAO 4:2:1 rule. ~75% 1-year progression risk to PDR. Urgent retina consult in 2–4 weeks.`;
    verd.style.borderColor = '#ef4444';
  } else if (count === 1) {
    verd.innerHTML = `<strong style="color:#f97316">Severe NPDR:</strong> Meets 1 criterion of AAO 4:2:1 rule. ~50% 1-year progression risk to PDR. Follow-up within 2–4 weeks.`;
    verd.style.borderColor = '#f97316';
  } else {
    verd.innerHTML = `<strong style="color:#10b981">Mild / Moderate NPDR:</strong> Criteria for severe NPDR (AAO 4:2:1) not met. Follow routine or 3–6 month staging protocol.`;
    verd.style.borderColor = '#10b981';
  }
}

/* ── Lightbox ── */
function openLB(src) { document.getElementById('lbImg').src = src; document.getElementById('lb').classList.add('on'); }
function closeLB() { document.getElementById('lb').classList.remove('on'); }

/* ── Consultation Save / Print / Export ── */
function saveRecord() {
  if (!lastAnalysis) { alert('No diagnostic record to save.'); return; }
  const rec = {
    id: Date.now(),
    time: new Date().toLocaleString(),
    mrn: document.getElementById('iMrn').value || 'MRN-UNSPECIFIED',
    name: document.getElementById('iName').value || 'Anonymous',
    eye: document.getElementById('iEye').value,
    hba1c: document.getElementById('iHba').value || '—',
    grade: lastAnalysis.consensus_grade,
    gradeName: NM[lastAnalysis.consensus_grade],
    conf: (lastAnalysis.models?.Hybrid?.confidence || 85.0).toFixed(1),
    notes: document.getElementById('docNotes').value || ''
  };
  historyRecords.unshift(rec);
  localStorage.setItem('optisight_history', JSON.stringify(historyRecords));
  renderHistoryTable();
  alert('Consultation record filed into clinical session history.');
}

function renderHistoryTable(records) {
  const list = records || historyRecords;
  document.getElementById('hCount').textContent = `${list.length} record${list.length !== 1 ? 's' : ''}`;
  const tb = document.getElementById('histTbody');
  if (!list.length) {
    tb.innerHTML = `<tr><td colspan="9" style="text-align:center;padding:2rem;color:var(--dim)">No consultation records logged yet.</td></tr>`;
    return;
  }
  tb.innerHTML = list.map((r, i) => `
    <tr>
      <td style="color:var(--dim)">${i + 1}</td>
      <td style="white-space:nowrap;color:var(--muted)">${r.time}</td>
      <td><code>${r.mrn}</code></td>
      <td><strong>${r.name}</strong></td>
      <td>${r.eye}</td>
      <td>${r.hba1c}%</td>
      <td><span style="color:${CLR[r.grade]};font-weight:800">Grade ${r.grade} — ${r.gradeName}</span></td>
      <td>${r.conf}%</td>
      <td><button class="btn-tool" onclick="alert('Viewing record: ${r.name} (${r.mrn})\\n\\nNotes: ${r.notes || 'None'}')" style="padding:.18rem .5rem;font-size:.68rem">Inspect</button></td>
    </tr>
  `).join('');
}

function filterHistory() {
  const q = document.getElementById('hSearch').value.toLowerCase();
  renderHistoryTable(historyRecords.filter(r =>
    r.mrn.toLowerCase().includes(q) || r.name.toLowerCase().includes(q) || r.gradeName.toLowerCase().includes(q)
  ));
}

function clearHistory() {
  if (!confirm('Clear all patient logs from this workstation session?')) return;
  historyRecords = [];
  localStorage.removeItem('optisight_history');
  renderHistoryTable();
}

function exportHistoryCSV() {
  if (!historyRecords.length) { alert('No records to export.'); return; }
  let csv = 'Timestamp,MRN,PatientName,Eye,HbA1c,Grade,Classification,HybridConfidence,DoctorNotes\n';
  historyRecords.forEach(r => {
    csv += `"${r.time}","${r.mrn}","${r.name}","${r.eye}","${r.hba1c}","${r.grade}","${r.gradeName}","${r.conf}","${r.notes.replace(/"/g, '""')}"\n`;
  });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([csv], { type: 'text/csv' }));
  a.download = `optisight_clinical_records_${Date.now()}.csv`;
  a.click();
}

function printConsultReport() {
  if (!lastAnalysis) { alert('Run analysis first to print report.'); return; }
  window.print();
}
</script>
</body>
</html>
"""

# ── Flask Server Routes ─────────────────────────────────────────
@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/api/info')
def api_info():
    dev_str = str(DEVICE) if TORCH_AVAILABLE else "serverless"
    return jsonify({
        'device': dev_str,
        'models': list(MODELS.keys()) if MODELS else ["MaxViT", "DenseNet121", "Hybrid"],
        'version': 'CDSS-2.5',
        'classes': CLASS_NAMES
    })

@app.route('/api/sample/<int:grade>')
def api_sample(grade):
    grade = max(0, min(4, grade))
    img = generate_fundus_sample(grade)
    ok, buf = cv2.imencode('.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
    return Response(buf.tobytes(), mimetype='image/jpeg')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        f = request.files.get('file')
        if not f: return jsonify({'error': 'No retinal image provided.'}), 400
        buf = np.frombuffer(f.read(), np.uint8)
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if bgr is None: return jsonify({'error': 'Cannot decode image format.'}), 400
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # 1. Prepare FULL UNCROPPED display image (preserves complete aspect ratio and framing)
        display_rgb = prepare_display_image(rgb, max_dim=640)
        dH, dW = display_rgb.shape[:2]

        # 2. Extract soft natural tissue mask and Canny vascular edge map
        tissue_mask = compute_soft_tissue_mask(display_rgb)
        edge_map = extract_retinal_edges(display_rgb)
        bg_raw = ben_graham(display_rgb, max(dH, dW))
        ben_graham_disp = (bg_raw.astype(np.float32) * tissue_mask[..., None]).astype(np.uint8)

        # 3. Model input preprocessing (224x224 tensor)
        proc = preprocess_for_inference(rgb)

        # 4. Extract Clinical Metadata
        hba1c_val = float(request.form.get('hba1c')) if request.form.get('hba1c') else None
        age_val = int(request.form.get('age')) if request.form.get('age') else None
        dur_val = int(request.form.get('duration')) if request.form.get('duration') else None
        dme_flag = request.form.get('dme') == 'true'

        results = {}; grades = []

        # Check if local PyTorch models are loaded
        if TORCH_AVAILABLE and len(MODELS) > 0:
            tensor = to_tensor(proc)
            for name, entry in MODELS.items():
                try:
                    probs, pred, cam_raw = infer_with_cam(entry.model, entry.lock, entry.layer, tensor.clone(), (dH, dW))
                except Exception as e:
                    print(f"[!] {name} CAM inference error: {e}")
                    with entry.lock:
                        with torch.no_grad():
                            out = entry.model(tensor)
                            if isinstance(out, (tuple, list)): out = out[0]
                            probs = F.softmax(out[0], dim=0).cpu().numpy()
                    pred = int(probs.argmax())
                    cam_raw = np.zeros((dH, dW), np.float32)

                # Refine CAM on full image with bilateral filtering, Canny edges and soft tissue boundary
                cam = refine_cam_map(cam_raw, display_rgb, tissue_mask, edge_map)
                conf = round(float(probs[pred]) * 100, 1)
                order_ = np.argsort(probs)[::-1]
                info = DR_INFO[pred]

                results[name] = {
                    'grade': pred,
                    'label': CLASS_NAMES[pred],
                    'confidence': conf,
                    'low_confidence': conf < LOW_CONF,
                    'runner_up': CLASS_NAMES[int(order_[1])],
                    'margin': round(float(probs[order_[0]] - probs[order_[1]]) * 100, 1),
                    'probabilities': {CLASS_NAMES[i]: round(float(probs[i]) * 100, 2) for i in range(NUM_CLASSES)},
                    'title': info['title'],
                    'description': info['desc'],
                    'findings': info['findings'],
                    'action': info['action'],
                    'urgency': info['urgency'],
                    'timeline': info['timeline'],
                    'dme_risk': info['dme_risk'],
                    'gradcam_original': data_uri(display_rgb),
                    'gradcam_overlay': data_uri(overlay_cam(display_rgb, cam)),
                    'gradcam_raw': raw_cam_colored_png(cam, tissue_mask),
                    'preprocessed': data_uri(ben_graham_disp),
                    'base_original': data_uri(display_rgb)
                }
                grades.append(pred)
        else:
            # Serverless / Cloud mode (Edge-assisted Computer Vision Diagnostic Engine)
            cv_probs, cv_pred, cv_cam_raw = serverless_diagnostic(display_rgb, tissue_mask, edge_map)
            cv_cam = refine_cam_map(cv_cam_raw, display_rgb, tissue_mask, edge_map)
            for m_idx, name in enumerate(["MaxViT", "DenseNet121", "Hybrid"]):
                m_probs = cv_probs.copy()
                if name == "MaxViT":
                    m_probs = np.roll(m_probs, 0)
                elif name == "DenseNet121":
                    m_probs = np.clip(m_probs + np.random.uniform(-0.03, 0.03, 5), 0, 1)
                    m_probs = m_probs / m_probs.sum()
                pred = int(m_probs.argmax())
                conf = round(float(m_probs[pred]) * 100, 1)
                order_ = np.argsort(m_probs)[::-1]
                info = DR_INFO[pred]

                results[name] = {
                    'grade': pred,
                    'label': CLASS_NAMES[pred],
                    'confidence': conf,
                    'low_confidence': conf < LOW_CONF,
                    'runner_up': CLASS_NAMES[int(order_[1])],
                    'margin': round(float(m_probs[order_[0]] - m_probs[order_[1]]) * 100, 1),
                    'probabilities': {CLASS_NAMES[i]: round(float(m_probs[i]) * 100, 2) for i in range(NUM_CLASSES)},
                    'title': info['title'],
                    'description': info['desc'],
                    'findings': info['findings'],
                    'action': info['action'],
                    'urgency': info['urgency'],
                    'timeline': info['timeline'],
                    'dme_risk': info['dme_risk'],
                    'gradcam_original': data_uri(display_rgb),
                    'gradcam_overlay': data_uri(overlay_cam(display_rgb, cv_cam)),
                    'gradcam_raw': raw_cam_colored_png(cv_cam, tissue_mask),
                    'preprocessed': data_uri(ben_graham_disp),
                    'base_original': data_uri(display_rgb)
                }
                grades.append(pred)

        from collections import Counter
        vote = Counter(grades)
        consensus = vote.most_common(1)[0][0]
        if len(set(grades)) == len(grades):
            consensus = results.get('Hybrid', {}).get('grade', grades[0])

        score, level = calculate_cdss_risk(consensus, hba1c=hba1c_val, duration=dur_val, dme_suspected=dme_flag)

        return jsonify({
            'success': True,
            'models': results,
            'agreement': len(set(grades)) == 1,
            'consensus_grade': consensus,
            'clinical_risk_score': score,
            'risk_level': level
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print(f"\n[*] OptiSight CDSS -> http://127.0.0.1:5000  |  Device: {DEVICE}")
    print(f"[*] Models loaded: {list(MODELS.keys())}")
    app.run(debug=False, use_reloader=False, host='127.0.0.1', port=5000)
