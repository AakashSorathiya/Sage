# This file contains the implementation of baseline standalone GPT model.

import torch
import openai
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix
import pandas as pd
from collections import Counter
import math

labeled_reviews = pd.read_csv('../datafiles/labeled_reviews.csv')
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
gpt_model = 'gpt-4o-mini'

openai.api_key = '<openai-api-key>'

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

    true_labels = data['Privacy-related'].to_list()
    metrics = classification_report(true_labels, gpt_response, output_dict=True)
    kappa_score = cohen_kappa_score(true_labels, gpt_response)
    TN, FP, FN, TP = confusion_matrix(true_labels, gpt_response, labels=[0, 1]).ravel()
    print(f'TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}')
    return {'metrics': metrics, 'kappa': kappa_score}

def standalone_gpt():
    results = infer_gpt(labeled_reviews)
    print(results) # Report Sage results.

standalone_gpt()