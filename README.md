# Sage
A Context-based Hybrid Approach to Mining Privacy Concerns in Mental Health App Reviews

### Introduction

This repository includes all the materials (code, input data, evaluation results, output dataset) of our approach to extract privacy-related app reviews in mental health domain. These materials involve quantitative and qualitative analysis of Sage.

### File Descriptions

There are three directories in total, and the content of each directory is described as follows:

`datafiles`: the input data used for this study.

- `labeled_reviews.csv` is the ground truth dataset containing privacy or not-privacy labeled app reviews from mental health domain.
- `unlabeled_reviews.csv` is the unlabeled dataset containing 1 and 2 star rated app reviews from mental health domain. We use this dataset extract new privacy related app reviews.
- `sage_classified_reviews.csv` is the dataset extracted using our proposed Sage approach.

`docs`: additional supporting documents for our study.

- `evaluation_results.md` contains the results of our manual inspection process followed to create new gold-standard dataset of privacy-related app reviews.
- `manual_label_guide.md` contains the instructions followed by the annotators for manual inspection.

`programfiles`: the source code of our study.

- `nli_inference_test.py` contains the code for NLI inference and evaluation. We perform the NLI inference using `DeBERTa-V3` base model and evaluate 3 different sets of heuristics to select the best one without any bias.
- `sage_evaluation.py` contains the code for complete Sage implementation and evaluation using domain-specific hypotheses and GPT model.
- `sage_unlabeled_data.py` contains the code to extract new privacy reviews (qualitative evaluation) from the unlabeled dataset using Sage.
- `data_preprocess.py` contains the data pre-processing code used to clean the data before extracting privacy reviews.
- `generic_hypotheses` contains the implementation of baseline generic hypotheses with Sage. 
- `bert.py` contains the implemetation of baseline `BERT` classifier.
- `svm.py` contains the implemetation of baseline `SVM` classifier.
- `t5.py` contains the implemetation of baseline `T5` classifier.
- `standalone_gpt.py` contains the implemetation of baseline `Standalone GPT` model.

`new_privacy_reviews.csv` is the manually curated dataset containing 1,008 privacy-related app reviews, after using Sage on a dataset of 42,271 app reviews.

`supplementary_material.pdf` provides supporting document for the paper.