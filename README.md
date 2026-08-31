# Nifty Stock Research Dashboard

## Run in Streamlit Community Cloud (recommended)
1. Create a public GitHub repository.
2. Upload `app.py` and `requirements.txt` from this package to the repository root.
3. Go to https://share.streamlit.io, sign in with GitHub, choose **Create app**, select the repository, branch, and `app.py`, then deploy.

No Google Colab, ngrok, TA-Lib, or local server is required.

## Local run
```bash
pip install -r requirements.txt
streamlit run app.py
```

The dashboard uses Yahoo Finance data and caches price requests for 15 minutes. The scanner downloads data for the Nifty universe, so it can take a little time on a first run.
