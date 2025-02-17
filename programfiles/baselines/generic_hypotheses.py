# This file contains the implementation of baseline generic hypotheses with Sage.

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import openai
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix
import pandas as pd
from collections import Counter
import math

labeled_reviews = pd.read_csv('../datafiles/labeled_reviews.csv')
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
nli_model = 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'
gpt_model = 'gpt-4o-mini'

openai.api_key = '<openai-api-key>'

generic_hypotheses = [
    "The user is facing a data surveillance issue.",
    "The user is forced to provide information.",
    "Personal user information is collected from other sources.",
    "The user is concerned about protecting their personal data.",
    "A data anonymity topic is discussed.",
    "The user is concerned about the purposes of personal data access.",
    "The user wants to correct their personal information.",
    "A breach of data confidentiality is discussed.",
    "Personal data disclosure is discussed.",
    "The app exposes a private aspect of the user life.",
    "User's data has been made accessible to public.",
    "A data blackmailing issue is discussed.",
    "User data is being exploited for other purposes.",
    "False data is presented about the user.",
    "Unwanted intrusion to personal info is discussed.",
    "Intrusion by the government to the user's life is discussed.",
    "Opting out from personal data collection is discussed.",
    "More access than needed is required.",
    "The reason for data access is not provided.",
    "Too much personal data is collected.",
    "The data is being used for unexpected purposes.",
    "Data sharing with third parties is discussed.",
    "User choice for personal data collection is discussed.",
    "User did not allow access to their personal data.",
    "A data privacy topic is discussed.",
    "Protecting user's personal data is discussed.",
    "This is about a privacy feature.",
    "The user is facing a privacy issue.",
    "The user likes that data privacy is provided.",
    "The user wants privacy.",
    "The app has privacy features."
]

def apply_heuristics(entailment_scores):
    labels = []
    for score_l in entailment_scores:
        if score_l is not None and len(score_l)>0:
            sorted_scores = sorted(score_l, reverse=True)
            if sorted_scores[0]<0.4:
                label='maybe-not-privacy'
            elif sorted_scores[0]>0.8 or sorted_scores[2]>0.7 or sorted_scores[4]>0.6 or sorted_scores[6]>0.5:
                label = 'maybe-privacy'
            else:
                label = 'undetermined'
        else:
            label = 'undetermined'
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

def nli_inference():
    tokenizer = AutoTokenizer.from_pretrained(nli_model)
    model = AutoModelForSequenceClassification.from_pretrained(nli_model)
    model.to(device)
    reviews = labeled_reviews['clean_content'].to_list()
    entailment_scores = execute_nli(generic_hypotheses, reviews, model, tokenizer)
    nli_annotations = apply_heuristics(entailment_scores)
    nli_annotated_data = pd.concat([labeled_reviews, pd.DataFrame({'entailment_scores': entailment_scores, 'relevancy_label': nli_annotations})], axis=1)
    return nli_annotated_data

def call_gpt_api(review):
    messages = [
        {
            'role': 'system',
            'content': f'''You are a scholarly researcher and your task is to annotate the data. You will receive a list of app review and you have to annotate each review with a yes or no label based on the privacy hypothesis provided below. 
If the review satisfies any of the hypothesis then annotate it with a yes label otherwise annotate it with a no label.
Please remember the answer should be just one word, yes or no, don't add any extra text. If you don't know the answer just say undetermined but do not add any extra explaination.
Privacy Hypotheses:
 - {'\n - '.join(generic_hypotheses)}
            '''
        },
        {
            'role': 'user',
            'content': f'''App Review: {review}
Does this app review satisfies any of the hypothesis? Respond with yes or no'''
        }
    ]
    try:
        response = openai.chat.completions.create(
            model=gpt_model,
            messages=messages,
            max_tokens=1,
            temperature=0.2,
            logprobs=True
        )
        pred_label = response.choices[0].message.content.lower()
        pred_prob = math.exp(response.choices[0].logprobs.content[0].logprob)
        return pred_label, pred_prob
    except Exception as e:
        print(f"Error processing review: {review}\n{e}")
        return "error"
    
def infer_gpt(data):
    reviews = data['clean_content'].to_list()
    gpt_response = []
    for review in reviews:
        five_trials=[]
        for _ in range(5):
            label, prob = infer_gpt(review)
            five_trials.append(label)
        trial_responses = Counter(five_trials)
        majority = trial_responses.most_common()[0][0]
        if majority=='yes':
            gpt_response.append(1)
        else:
            gpt_response.append(0)

    true_labels = data['Privacy-related'].to_list()
    metrics = classification_report(true_labels, gpt_response, output_dict=True)
    kappa_score = cohen_kappa_score(true_labels, gpt_response)
    TN, FP, FN, TP = confusion_matrix(true_labels, gpt_response, labels=[0, 1]).ravel()
    print(f'TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}')
    return {'metrics': metrics, 'kappa': kappa_score}

def generic_hypotheses():
    nli_annotated_dataset = nli_inference()
    relevant_reviews = nli_annotated_dataset[nli_annotated_dataset['relevancy_label']=='maybe-privacy']
    results = infer_gpt(relevant_reviews)
    print(results) # Report results.

generic_hypotheses()