deployed link:https://eman-bike-demand-app.streamlit.app/

# Bike Rental Demand Prediction System

Machine Learning Fundamentals (AIC354) – Assignment 3, Task 2.
Predicts the number of bikes rented in a given hour from calendar and weather information (UCI Bike Sharing Dataset, `hour.csv`, 17,379 labelled hourly records). Five regression algorithms were compared with 10-fold cross-validation in Google Colab; the best one (lowest mean RMSE) is served by a Streamlit web app.

**Live app:** `PLACEHOLDER – paste your verified https://….streamlit.app link here after deployment`

## Files

| File | Purpose |
|---|---|
| `bike_rental_kfold_full_dataset_10fold.ipynb` | Full Colab notebook (EDA, pre-processing, 10-fold CV, comparison, model saving) |
| `app.py` | Streamlit web application |
| `model/bike_demand_model.joblib` | Fitted pipeline (pre-processing + best model) |
| `model/model_metadata.json` | Feature list, valid ranges, CV results, library versions |
| `requirements.txt` | Libraries (scikit-learn version is **pinned** to the one that created the model) |
| `.streamlit/config.toml` | Theme |
| `data/hour.csv`, `day.csv`, `Readme.txt` | Dataset (cite Fanaee-T & Gama, 2013) |

## 1. Test locally first

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows   (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
streamlit run app.py
```
Open http://localhost:8501, change the inputs and check that the prediction and the 24-hour chart update.

## 2. Upload to GitHub

1. Create a new **public** repository on github.com (e.g. `bike-rental-demand`).
2. Upload everything in this folder (drag-and-drop in the browser works) **or**:
   ```bash
   git init
   git add .
   git commit -m "Bike rental demand prediction app"
   git branch -M main
   git remote add origin https://github.com/<your-username>/bike-rental-demand.git
   git push -u origin main
   ```
3. Check that `model/bike_demand_model.joblib` is visible on GitHub. Files over 100 MB are rejected by GitHub (see troubleshooting).

## 3. Deploy on Streamlit Community Cloud (free)

1. Go to https://share.streamlit.io and sign in with GitHub.
2. **Create app → Deploy a public app from GitHub**.
3. Repository: `<your-username>/bike-rental-demand`, branch `main`, main file path `app.py`.
4. Open **Advanced settings** and select **Python 3.12 or 3.13** (the pinned scikit-learn version needs a recent Python).
5. Click **Deploy** and wait for the build to finish (a few minutes).
6. Open the `https://<something>.streamlit.app` link, try several inputs (e.g. 8:00 on a working day vs 3:00 at night), then copy that **tested** URL to:
   - the very first cell of the notebook (`DEPLOYMENT_URL`),
   - this README.

## 4. Using the model you trained in Colab

The notebook saves `model/bike_demand_model.joblib`, `model/model_metadata.json` and `requirements.txt` (Step 10.3). Download these three files from Colab (Files panel, or set `DOWNLOAD_FILES = True`) and **replace** the ones in this folder. The new `requirements.txt` pins the scikit-learn version used in Colab, which must match on the server.

## Troubleshooting

| Problem | Fix |
|---|---|
| `InconsistentVersionWarning` or "model could not be loaded" | scikit-learn on the server differs from the one used for training. Use the `requirements.txt` generated together with the model file. |
| Build fails: "No matching distribution for scikit-learn==…" | Choose a newer Python (3.12/3.13) in Advanced settings, or retrain in Colab and use the `requirements.txt` it writes. |
| GitHub refuses the model (>100 MB) | Re-save with `joblib.dump(model, path, compress=9)`, or use Git LFS. |
| App says "Model files not found" | Keep the folder layout above: `model/` next to `app.py`. |
| App is "sleeping" | Free apps sleep after inactivity; open the link and click *Wake up*. Do this before the demo/marking. |

## Honest notes

- The cross-validation numbers in the notebook come from the run that produced the saved model; re-running in Colab can change the last decimals if library versions differ.
- No deployment link has been created by the notebook authors' tooling – the link above is a placeholder until you deploy.
