import classifiers
import pandas as pd
from sklearn.metrics import f1_score

data = pd.read_csv("classification/data.csv")

for classifier_cls in classifiers.CLASSIFIERS:
    classifier = classifier_cls()
    print(f"Classifier: {classifier_cls.__name__}")
    data["prediction"] = data["prompt"].apply(classifier.is_code_prompt)
    print(f"F1 score: {f1_score(data['is_code_prompt'], data['prediction'])}")
