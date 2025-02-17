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