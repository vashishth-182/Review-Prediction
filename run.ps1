$ErrorActionPreference = "Stop"

python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt

if (-not (Test-Path "artifacts\model.h5")) {
  .\.venv\Scripts\python train_model.py
}

.\.venv\Scripts\python -m streamlit run app.py

