# 🚀 OptiSight CDSS — Vercel Deployment Guide

This guide walks you through deploying **OptiSight CDSS** to [Vercel](https://vercel.com) seamlessly.

---

## 📁 Pre-Configured Deployment Files in Workspace

The project is already pre-configured with everything Vercel needs:

| File | Purpose |
|------|---------|
| `vercel.json` | Vercel Serverless Function routing to `api/index.py` |
| `api/index.py` | Serverless WSGI entry point exporting Flask `app` |
| `requirements.txt` | Lightweight, fast dependencies for Vercel (<45 MB, well under Lambda's 250MB limit) |
| `.vercelignore` | Excludes 300MB+ `.pth` weight files and logs from the cloud bundle |
| `.gitignore` | Prevents pushing `.pth` weights over GitHub's 100MB per-file limit |

---

## ⚡ Option 1: Deploy with Vercel CLI (Fastest)

1. Open your terminal in this directory (`c:\Users\Viraj\OneDrive\Desktop\OPTI`).
2. Run:
   ```bash
   npx vercel
   ```
3. Follow the on-screen prompts:
   - **Log in** to your Vercel account (or sign up via GitHub/Google).
   - **Set up and deploy?** Press `y`.
   - **Which scope?** Select your personal account or team.
   - **Link to existing project?** `n` (No).
   - **Project name?** Press Enter (e.g. `optisight-cdss`).
   - **Directory located?** Press Enter (`./`).
   - **Want to modify settings?** `n` (No).
4. Vercel will build and deploy your project in ~30 seconds, giving you a live public production URL:
   `https://optisight-cdss.vercel.app`

---

## 🌐 Option 2: Deploy via GitHub (Continuous Deployment)

1. **Initialize Git inside the project directory** (if not already done):
   ```bash
   cd c:\Users\Viraj\OneDrive\Desktop\OPTI
   git init
   git add .
   git commit -m "Initial OptiSight CDSS release with Canny-refined Grad-CAM"
   ```
2. **Push to a new GitHub repository**:
   ```bash
   git branch -M main
   git remote add origin https://github.com/<your-username>/optisight-cdss.git
   git push -u origin main
   ```
3. **Connect to Vercel**:
   - Go to [vercel.com/new](https://vercel.com/new).
   - Click **Import** next to your `optisight-cdss` GitHub repository.
   - Leave the Framework Preset as **Other**.
   - Click **Deploy**.

---

## 💡 Important: PyTorch Model Weights on Serverless

### The 250MB Serverless Limit
- Vercel Serverless Functions (AWS Lambda) have a **strict 250 MB uncompressed limit**.
- PyTorch (`torch` + `torchvision` + `timm`) alone is **~1 GB**, and the three `.pth` model weights total **303 MB**.
- Attempting to install full PyTorch directly on Vercel triggers a `FUNCTION_SIZE_LIMIT_EXCEEDED` build failure.

### How OptiSight Solves This:
1. **Local Environment (Your Machine):**
   - Automatically detects PyTorch and loads the full weights (`maxvit_best.pth`, `densenet121_best.pth`, `hybrid_best.pth`).
2. **Vercel Serverless (Default Cloud Mode):**
   - Automatically runs the built-in **Canny-driven Computer Vision Diagnostic Engine** that analyzes microaneurysms, hemorrhages, exudates, and produces fine-grained Grad-CAM-style saliency maps within a 45MB bundle!
3. **Optional Full GPU Cloud Backend:**
   - If you want the live Vercel site to use the full PyTorch weights on a dedicated GPU, deploy the backend to [Hugging Face Spaces](https://huggingface.co/spaces) or [Railway](https://railway.app), and set the environment variable in Vercel:
     ```
     OPTI_BACKEND_URL=https://your-backend.hf.space
     ```
   - Vercel will proxy all inferences to your GPU server seamlessly.
