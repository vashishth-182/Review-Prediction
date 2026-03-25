## Review Sentiment Predictor

Streamlit app that predicts review sentiment using an LSTM trained on the IMDB dataset.

### Run (minimum)

```powershell
cd C:\Users\Vashishth\Desktop\Review-Prediction
.\run.ps1
```

Open `http://localhost:8501`.

### Manual run (if you prefer)

```powershell
cd C:\Users\Vashishth\Desktop\Review-Prediction
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
.\.venv\Scripts\python train_model.py
.\.venv\Scripts\python -m streamlit run app.py
```

### Evaluate model quickly

```powershell
.\.venv\Scripts\python evaluate.py
```

### Troubleshooting

- If you see “Missing artifacts”, run `python train_model.py` (it creates `artifacts/model.h5` + `artifacts/meta.json`).  
- If port 8501 is busy, stop the other Streamlit process or run `streamlit run app.py --server.port 8502`.


