import sys, os

# Add parent directory to python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from optisight_app import app

# Vercel Serverless Function entry point
app = app
