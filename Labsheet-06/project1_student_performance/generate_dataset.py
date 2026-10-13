"""Generates a synthetic student dataset -> student_performance.csv"""
import numpy as np, pandas as pd
rng = np.random.default_rng(11)
n = 500
attendance = rng.normal(78, 12, n).clip(35, 100)
study_hours = rng.gamma(4, 1.1, n).clip(0.5, 12)               # hours per day
internal = (0.3 * attendance + 2 * study_hours + rng.normal(5, 5, n)).clip(5, 50)   # out of 50
assignment = (0.25 * attendance + 1.5 * study_hours + rng.normal(3, 4, n)).clip(0, 40) # out of 40
prev_sem = (0.45 * attendance + 3.5 * study_hours + rng.normal(15, 8, n)).clip(30, 98)  # percentage
raw = 0.25 * attendance + 0.5 * internal + 0.45 * assignment + 0.2 * prev_sem + 1.2 * study_hours + rng.normal(0, 4, n)
final = (58 + 16 * (raw - raw.mean()) / raw.std()).clip(0, 100)
df = pd.DataFrame({
    "student_id": range(1, n + 1),
    "attendance_pct": attendance.round(1), "internal_marks": internal.round(1),
    "assignment_marks": assignment.round(1), "study_hours_per_day": study_hours.round(1),
    "previous_sem_percentage": prev_sem.round(1), "final_score": final.round(1)})
# data-quality issues for the preprocessing step
for col in ["attendance_pct", "assignment_marks", "study_hours_per_day"]:
    df.loc[rng.choice(n, 8, replace=False), col] = np.nan
df = pd.concat([df, df.sample(5, random_state=2)], ignore_index=True)
df.to_csv("student_performance.csv", index=False); print(df.shape)
