# MCA201P – Lab Sheet 06: Mini Projects on Supervised Learning Applications
MCA III Semester (Session 2026–2027)

```
project1_student_performance/   student_performance.ipynb (executed) + .py, student_performance.csv, generate_dataset.py, model/, outputs/
project2_house_price/           house_price.ipynb (executed) + .py, house_prices.csv, generate_dataset.py, model/, outputs/
REPORT.md                       results, interpretation, viva questions
requirements.txt
```
Run (Python 3.11+): `pip install -r requirements.txt`, then open a notebook in Jupyter/VS Code and "Run All",
or run `python student_performance.py` / `python house_price.py` inside each project folder.

Datasets are synthetic (fixed seed, created by `generate_dataset.py`) with deliberate missing values and duplicates so the
preprocessing step is meaningful. To use a real dataset (e.g. Kaggle "Student Performance" or "House Price"), replace the CSV
and keep the column names (or edit the `features` / `num_cols` lists).
Trained models are saved with Joblib in each `model/` folder.
