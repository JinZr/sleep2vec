import os
import csv
from datetime import datetime
import pandas as pd
from pathlib import Path
import numpy as np
import json

metadata_path = "/home/notebook/data/personal/S9063410/bp_data_one_channel/加入研究信息整合名单260422.csv"

device_list = [
    "OPPO_Watch4_Pro",
    "OPPO_Watch_X2_mini",
    "OPPO_Watch_X2",
    "OPPO_Watch_X3",
]

output_path = "/home/notebook/data/personal/S9063410/bp_data_one_channel/index.csv"
fieldnames = [
    "path",
    "dataset",
    "session_id",
    "patient_id",
    "duration",
    "age",
    "sex",
    "weight",
    "height",
    "bmi",
]

def append_row_to_csv(csv_path: str | Path, row: dict, fieldnames: list[str]) -> None:
    csv_path = Path(csv_path)
    file_exists = csv_path.exists()

    # 把 None 转成空字符串，避免写成 "None"
    clean_row = {
        k: ("" if row.get(k) is None else row.get(k))
        for k in fieldnames
    }

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(clean_row)

metadata = pd.read_csv(metadata_path)
metadata_formatted = {}
for i, row in metadata.iterrows():
    ssoid = row["ssoid"]
    # 体重转为单位kg
    if pd.isna(row["weight"]):
        weight = None
    elif not isinstance(row["weight"], float):
        weight = None
    elif row["weight"] < 40000 or row["weight"] > 200000:
        weight = None
    else:
        weight = row["weight"] / 1000
    

    # 身高转为单位cm
    if pd.isna(row["height"]):
        height = None
    elif not isinstance(row["height"], float):
        height = None
    elif row["height"] < 1400 or row["height"] > 2100:
        height = None
    else:
        height = row["height"] / 10
    
    if weight is not None and height is not None:
        bmi = weight / (height / 100) ** 2
    else:
        bmi = None
    # sex转为male或female或None
    if row["sex"] == "M":
        sex = "male"
    elif row["sex"] == "F":
        sex = "female"
    else:
        sex = None

    # 生日转成time格式
    if pd.isna(row["birthday_value"]):
        birthday_value = None
    else:
        birthday_value = datetime.strptime(row["birthday_value"], "%Y-%m-%d")
        if birthday_value.year > 2010 or birthday_value.year < 1940:
            birthday_value = None

    metadata_formatted[str(ssoid)] = {
        "weight": weight,     # 体重转为单位kg
        "height": height,       # 身高转为单位cm
        "bmi": bmi,
        "sex": sex,
        "birthday": birthday_value
    }


src_dir = "/home/notebook/data/personal/S9063410/bp_data_one_channel"
full_info_ssoids = []
for device in device_list:
    device_dir = os.path.join(src_dir, device)
    for ssoid in os.listdir(device_dir):
        if ssoid not in metadata_formatted:
            with open("fail_list.txt", 'a+') as f:
                f.write(f"{ssoid} of {device} not in 加入研究信息整合名单260422.csv")
            continue
        print(device, ssoid)
        ssoid_dir = os.path.join(device_dir, ssoid)
        for file in os.listdir(ssoid_dir):

            npz_file = os.path.join(ssoid_dir, file)
            duration = np.load(npz_file, allow_pickle=True)["duration"]
            # 如果 duration 是 numpy 标量，转成普通 Python 数值
            if hasattr(duration, "item"):
                duration = duration.item()
            # 计算数据采集时用户年龄
            birthday = metadata_formatted[ssoid]["birthday"]
            if birthday is not None:
                collect_date = file.split("-")[1][:8]
                collect_date = datetime.strptime(collect_date, "%Y%m%d")
                age = round((collect_date - birthday).days / 365.2425, 1)
            else:
                age = None
            sex = metadata_formatted[ssoid]["sex"]
            weight = metadata_formatted[ssoid]["weight"]
            height = metadata_formatted[ssoid]["height"]
            bmi = metadata_formatted[ssoid]["bmi"]
            if weight == 60 and height == 170 and birthday is None:
                row = {
                    "path": npz_file,
                    "dataset": device,
                    "session_id": file.split('.')[0],
                    "patient_id": ssoid,
                    "duration": duration,
                    "age": age,
                    "sex": None,
                    "weight": None,
                    "height": None,
                    "bmi": None
                }
            else:
                row = {
                    "path": npz_file,
                    "dataset": device,
                    "session_id": file.split('.')[0],
                    "patient_id": ssoid,
                    "duration": duration,
                    "age": age,
                    "sex": sex,
                    "weight": weight,
                    "height": height,
                    "bmi": bmi
                }
                if age is not None and sex is not None and bmi is not None:
                    full_info_ssoids.append(ssoid)
            append_row_to_csv(output_path, row, fieldnames)

full_info_ssoids = set(full_info_ssoids)
with open("full_info_ssoids.json", 'w') as f:
    json.dump(full_info_ssoids, f, ensure_ascii=False, indent=4)
