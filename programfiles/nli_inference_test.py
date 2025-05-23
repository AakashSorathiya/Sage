# This file contains the NLI inference step of Sage and evaluation of 3 sets of heuristics.

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd

labeled_reviews = pd.read_csv('../datafiles/labeled_reviews.csv')
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
nli_model = 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'

domain_specific_hypotheses = [
    "Mental health data is linked across different services.",
    "Online activities across various mental health apps can be connected.",
    "Personal information about users' mental health is collected from external sources.",
    "Anonymized mental health data is used to re-identify the user.",
    "Unique patterns in a user’s psychological data lead to personal identification.",
    "User cannot deny having performed certain actions within the app.",
    "User is concerned about the permanent storage of their mental health history.",
    "User is concerned about others detecting their use of sensitive mental health services.",
    "Users’ participation in mental health apps is discovered from anonymized usage data.",
    "Users’ device communication patterns reveal private information about their mental health conditions.",
    "Mental health data intercepted during transmission.",
    "Mental health app exposes a private aspect of the user’s life.",
    "Private mental health information is accessed by unauthorized parties.",
    "User is not aware of how and why their mental health data is being collected, processed, stored, and shared.",
    "User is concerned about the processing and storage of mental health data against privacy regulations or policies.",
    "Mental health data is being exploited for other purposes.",
    "Mental health data is shared with third parties.",
    "The user is facing a privacy issue.",
    "The user is concerned about protecting their data.",
    "A data anonymity topic is discussed.",
    "A data privacy topic is discussed."
]

def apply_heuristics_set1(entailment_scores):
    labels = []
    for score_l in entailment_scores:
        if score_l is not None and len(score_l)>0:
            sorted_scores = sorted(score_l, reverse=True)
            if sorted_scores[0]>=0.9 or sorted_scores[2]>=0.8 or sorted_scores[4]>=0.75:
                label = 'maybe-privacy'
            else:
                label = 'maybe-not-privacy'
        else:
            label = 'maybe-not-privacy'
        labels.append(label)
    return labels

def apply_heuristics_set2(entailment_scores):
    labels = []
    for score_l in entailment_scores:
        if score_l is not None and len(score_l)>0:
            sorted_scores = sorted(score_l, reverse=True)
            if sorted_scores[0]>=0.85 or sorted_scores[2]>=0.75 or sorted_scores[4]>=0.7:
                label = 'maybe-privacy'
            else:
                label = 'maybe-not-privacy'
        else:
            label = 'maybe-not-privacy'
        labels.append(label)
    return labels

def apply_heuristics_set3(entailment_scores):
    labels = []
    for score_l in entailment_scores:
        if score_l is not None and len(score_l)>0:
            sorted_scores = sorted(score_l, reverse=True)
            if sorted_scores[0]>=0.8 or sorted_scores[2]>=0.7 or sorted_scores[4]>=0.65:
                label = 'maybe-privacy'
            else:
                label = 'maybe-not-privacy'
        else:
            label = 'maybe-not-privacy'
        labels.append(label)
    return labels

def execute_nli(hypotheses, reviews, model, tokenizer):
    entailment_scores = []
    for idx in range(0, len(reviews)):
        review = reviews[idx]
        scores = []
        if review and isinstance(review, str):
            try:
                for hpt in hypotheses:
                    input = tokenizer(review, hpt, truncation=True, return_tensors="pt")
                    output = model(input["input_ids"].to(device))
                    prediction = torch.softmax(output["logits"][0], -1).tolist()
                    scores.append(prediction[0])
            except Exception:
                print(f'some error occured for index: {idx}')
        entailment_scores.append(scores)
    return entailment_scores

def calculate_metrics(data):
    P, R, F1 = 0, 0, 0
    TP, TN, FP, FN = 0, 0, 0, 0
    for key in data[data['Privacy-related']==1]['relevancy_label'].value_counts().keys():
        if key=='maybe-privacy':
            TP += data[data['Privacy-related']==1]['relevancy_label'].value_counts().get(key)
        else:
            FN += data[data['Privacy-related']==1]['relevancy_label'].value_counts().get(key)
    for key in data[data['Privacy-related']==0]['relevancy_label'].value_counts().keys():
        if key=='maybe-privacy':
            FP += data[data['Privacy-related']==1]['relevancy_label'].value_counts().get(key)
        else:
            TN += data[data['Privacy-related']==1]['relevancy_label'].value_counts().get(key)

    if TP>0:
        P = TP/(TP+FP)
        R = TP/(TP+FN)
        F1 = (2*P*R) / (P+R)

    print(TP, TN, FP, FN)
    return {'P': P, 'R': R, 'F1': F1}

def nli_inference():
    tokenizer = AutoTokenizer.from_pretrained(nli_model)
    model = AutoModelForSequenceClassification.from_pretrained(nli_model)
    model.to(device)
    reviews = labeled_reviews['clean_content'].to_list()
    entailment_scores = execute_nli(domain_specific_hypotheses, reviews, model, tokenizer)
    return pd.concat([labeled_reviews, pd.DataFrame({'entailment_scores': entailment_scores})], axis=1)

def apply_heuristics(data):
    set1_annotations = apply_heuristics_set1(data['entailment_scores'].to_list())
    set1_metrics = calculate_metrics(pd.concat([data, pd.DataFrame({'relevancy_label': set1_annotations})], axis=1))

    set2_annotations = apply_heuristics_set2(data['entailment_scores'].to_list())
    set2_metrics = calculate_metrics(pd.concat([data, pd.DataFrame({'relevancy_label': set2_annotations})], axis=1))

    set3_annotations = apply_heuristics_set3(data['entailment_scores'].to_list())
    set3_metrics = calculate_metrics(pd.concat([data, pd.DataFrame({'relevancy_label': set3_annotations})], axis=1))

    return {'set1': set1_metrics, 'set2': set2_metrics, 'set3': set3_metrics}

entailment_scores = nli_inference()
evaluation_results = apply_heuristics(entailment_scores)

print(evaluation_results) # based on this metrics we select the best set of heuristics for further steps.