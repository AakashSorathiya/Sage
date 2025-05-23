# This file contains the complete Sage implementation on unlabeled dataset using domain-specific hypotheses and GPT model.

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import openai
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix
import pandas as pd
from collections import Counter
import math

unlabeled_reviews = pd.read_csv('../datafiles/unlabeled_reviews.csv')
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
nli_model = 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'
gpt_model = 'gpt-4o-mini'

openai.api_key = '<openai-api-key>'

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

def apply_heuristics(entailment_scores):
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
    reviews = unlabeled_reviews['clean_content'].to_list()
    entailment_scores = execute_nli(domain_specific_hypotheses, reviews, model, tokenizer)
    nli_annotations = apply_heuristics(entailment_scores)
    nli_annotated_data = pd.concat([unlabeled_reviews, pd.DataFrame({'entailment_scores': entailment_scores, 'relevancy_label': nli_annotations})], axis=1)
    return nli_annotated_data

def call_gpt_api(review):
    messages = [
        {
            'role': 'system',
            'content': f'''You are provided with an app review of a mental health mobile application in this format:
App Review: """content of the app review"""
You have to identify whether it is discussing any privacy concern or not.
Output should be just a yes or no label, where yes indicates that the review is related to privacy and no indicates that it is not related to privacy.
            '''
        },
        {
            'role': 'user',
            'content': f'''App Review: """{review}"""'''
        }
    ]
    try:
        response = openai.chat.completions.create(
            model=gpt_model,
            messages=messages,
            max_tokens=1,
            temperature=0,
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

    sage_classified_reviews = pd.concat([data, pd.DataFrame({'llm_response': gpt_response})], axis=1)
    return sage_classified_reviews

def sage():
    nli_annotated_dataset = nli_inference()
    relevant_reviews = nli_annotated_dataset[nli_annotated_dataset['relevancy_label']=='maybe-privacy']
    relevant_reviews = relevant_reviews.reset_index(drop=True)

    sage_classified_reviews = infer_gpt(relevant_reviews)
    sage_classified_reviews = sage_classified_reviews[sage_classified_reviews['llm_response']=='yes']
    sage_classified_reviews = sage_classified_reviews.reset_index(drop=True)
    return sage_classified_reviews

sage_classified_reviews = sage()
sage_classified_reviews.to_csv('../datafiles/sage_classified_reviews.csv')