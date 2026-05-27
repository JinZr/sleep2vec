import pandas as pd

bp_path = "/home/notebook/data/personal/S9063410/pwv+bp_data_multilight/bp_index_mask.csv"
pwv_path = "/home/notebook/data/personal/S9063410/pwv+bp_data_multilight/pwv_index_mask.csv"
out_path = "/home/notebook/data/personal/S9063410/pwv+bp_data_multilight/pwv+bp_index_mask.csv"

bp_df = pd.read_csv(bp_path)
pwv_df = pd.read_csv(pwv_path)

# 前面这些基础字段保持在最前面
base_cols = ["path", "dataset", "session_id", "patient_id", "duration", "age", "sex", "weight", "height", "bmi"]

# 两个表中所有 mask 列的并集
bp_mask_cols = [c for c in bp_df.columns if c not in base_cols]
pwv_mask_cols = [c for c in pwv_df.columns if c not in base_cols]

# 保持顺序：先 bp 里的 mask，再补 pwv 中新增的 mask
mask_cols = bp_mask_cols + [c for c in pwv_mask_cols if c not in bp_mask_cols]

final_cols = base_cols + mask_cols

# 对缺失的列补 0
for col in final_cols:
    if col not in bp_df.columns:
        bp_df[col] = 0
    if col not in pwv_df.columns:
        pwv_df[col] = 0

# 按统一列顺序排列
bp_df = bp_df[final_cols]
pwv_df = pwv_df[final_cols]

# 合并两份 csv 的所有行
merged_df = pd.concat([bp_df, pwv_df], axis=0, ignore_index=True)

# mask 列统一转成 int，避免有些 0 被存成 0.0
for col in mask_cols:
    merged_df[col] = merged_df[col].fillna(0).astype(int)

merged_df.to_csv(out_path, index=False)

print(f"Saved merged csv to: {out_path}")
print(f"Rows: {len(merged_df)}")
print(f"Columns: {len(merged_df.columns)}")
