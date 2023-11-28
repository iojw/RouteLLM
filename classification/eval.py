import classifiers
import pandas as pd
from sklearn.metrics import f1_score
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

SOURCE2CAT = {
    "humaneval": "coding",
    "gsm8k": "math",
    "mmlu": "others",
}

data = pd.read_csv("data_v1.csv")

# only use first 20 rows for testing
# data = data.iloc[:20]

# List of classifiers to use
CLASSIFIERS = [
    #classifiers.RandomClassifier(),
    #classifiers.NgramClassifier(),
    classifiers.LLMClassifier(model="gpt-3.5-turbo"),
    # classifiers.LLMClassifier(model="vicuna-7b-v1.5", api_base="FILLME"),
]

for classifier_cls in CLASSIFIERS:
    classifier = classifier_cls
    print(f"Classifier: {classifier.model}")

    with ThreadPoolExecutor(8) as executor:
        future = []
        index_list = []
        for index, row in data.iterrows():
            future.append(executor.submit(classifier.classify_prompt, row["prompt"]))
            index_list.append(index)
        for idx, f in tqdm(enumerate(future)):
            data.loc[index_list[idx], "prediction"] = f.result()

    # from source to label
    y_1 = [SOURCE2CAT[source] for source in data["source"].tolist()]
    y_2 = data["prediction"].tolist()
    print(y_1)
    print(y_2)
    print(f"F1 score: {f1_score(y_1, y_2, average='micro')}")
