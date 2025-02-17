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

domain_specific_hypotheses = [
    "User data being linked across different services.",
    "Online user activities from various platforms can be connected.",
    "Anonymized user data could be used to reveal their identity.",
    "Unique digital user data could lead to personal identification.",
    "User is unable to deny their online actions.",
    "User is concerned about the permanent storage of their digital transactions.",
    "User is concerned about others detecting their use of sensitive online services.",
    "User presence on certain platforms could be discovered from anonymized data.",
    "User device's communication patterns reveal private information.",
    "User personal data intercepted during transmission.",
    "Unauthorized access to user's private information.",
    "The user is not aware of how and why their data is being collected, processed, stored, and shared.",
    "The user is concerned about the processing or storing of their personal data against regulations or privacy policies.",
    "The user is facing a privacy issue.",
    "Personal user information is collected from other sources.",
    "The user is concerned about protecting their personal data.",
    "A data anonymity topic is discussed.",
    "The app exposes a private aspect of the user life.",
    "User data is being exploited for other purposes.",
    "Data sharing with third parties is discussed.",
    "A data privacy topic is discussed.",
]

def call_gpt_api(review):
    messages = [
        {
            'role': 'system',
            'content': f'''You are a scholarly researcher and your task is to annotate the data. You will receive a list of app review and you have to annotate each review with a yes or no label based on the privacy hypothesis provided below. 
If the review satisfies any of the hypothesis then annotate it with a yes label otherwise annotate it with a no label.
Please remember the answer should be just one word, yes or no, don't add any extra text. If you don't know the answer just say undetermined but do not add any extra explaination.
Privacy Hypotheses:
 - {'\n - '.join(domain_specific_hypotheses)}
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

def standalone_gpt():
    results = infer_gpt(labeled_reviews)
    print(results) # Report Sage results.

standalone_gpt()