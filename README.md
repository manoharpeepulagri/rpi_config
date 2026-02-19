# RPi Config Standalone (Vercel)

This folder contains a standalone deployment for the RPi Config UI.

## Files
- `api/index.py` - FastAPI serverless entrypoint (Vercel)
- `templates/rpi_config.html` - UI template (copied from main app)
- `static/styles/theme.css` - theme CSS
- `static/assets/Peepul_Agri_Final_Logo.png` - logo asset
- `requirements.txt` - Python dependencies
- `.env.example` - environment variables template
- `vercel.json` - Vercel routing/build config

## Deploy
1. Create a new Vercel project and set Root Directory to `rpi_config_vercel`.
2. Add environment variables from `.env.example` in Vercel Project Settings.
3. Deploy.

## Required Environment Variables
- `S3_BUCKET_NAME`
- `AWS_REGION` (default `ap-south-1`)
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_SESSION_TOKEN` (optional)
- `RPI_MAX_FILES` (optional, default `300`)

## Notes
- This standalone app exposes `/ops/rpi/*` endpoints directly.
- Action auto-reset is not enforced in-process in serverless mode.
