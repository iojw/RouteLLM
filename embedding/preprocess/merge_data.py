import json
import pandas as pd
import os

SEED = 1
PROGRAMMING_LABEL = "Programming"
MATH_LABEL = "Math"
NONE_LABEL = "None"

pwd = os.path.dirname(os.path.realpath(__file__))


glaive_data = json.load(open(pwd + "/glaive.json", "r"))
glaive_df = pd.DataFrame({"Prompt": glaive_data, "Label": [PROGRAMMING_LABEL] * len(glaive_data)})
gsm_data = json.load(open(pwd + "/gsm.json", "r"))
gsm_df = pd.DataFrame({"Prompt": gsm_data, "Label": [MATH_LABEL] * len(gsm_data)})
orca_data = json.load(open(pwd + "/orca.json", "r"))
orca_df = pd.DataFrame({"Prompt": orca_data, "Label": [NONE_LABEL] * len(orca_data)})

merge_df = pd.concat([glaive_df, gsm_df, orca_df])
merge_df.to_csv("merged_data.csv", index=False)
