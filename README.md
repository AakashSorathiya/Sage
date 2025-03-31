# SAGE
SAGE: A Context-Aware Approach for Mining Privacy Requirements Relevant Reviews from Mental Health Apps

### Introduction

This repository includes all the materials (code, input data, evaluation results, output dataset) of our approach to extracting privacy requirements-relevant reviews from mental health apps. These materials involve quantitative and qualitative analysis of SAGE.

### File Descriptions

There are three directories in total, and the content of each directory is described as follows:

`datafiles`: the input data used for this study.

- `labeled_reviews.csv` is the ground truth dataset containing privacy or not-privacy labeled app reviews from the mental health domain.
- `unlabeled_reviews.csv` is the unlabeled dataset containing 1 and 2-star rated app reviews from the mental health domain. We use this dataset to extract new privacy requirements-relevant reviews.
- `sage_classified_reviews.csv` is the dataset extracted using our proposed SAGE approach.

`docs`: additional supporting documents for our study.

- `evaluation_results.md` contains the results of our manual inspection process followed to create new gold-standard dataset of privacy requirements-relevant reviews.
- `manual_label_guide.md` contains the instructions followed by the annotators for manual inspection.

`programfiles`: the source code of our study.

- `nli_inference_test.py` contains the code for NLI inference and evaluation. We perform the NLI inference using the `DeBERTa-V3` base model and evaluate 3 different sets of heuristics to select the best one without any bias.
- `sage_evaluation.py` contains the code for complete SAGE implementation and evaluation using domain-specific hypotheses and the GPT model.
- `sage_unlabeled_data.py` contains the code to extract new privacy reviews (qualitative evaluation) from the unlabeled dataset using SAGE.
- `data_preprocess.py` contains the data pre-processing code used to clean the data before extracting privacy reviews.
- `generic_hypotheses` contains the implementation of baseline generic hypotheses with SAGE.
- `bert.py` contains the implementation of baseline `BERT` classifier.
- `svm.py` contains the implementation of baseline `SVM` classifier.
- `t5.py` contains the implementation of the baseline `T5` classifier.
- `standalone_gpt.py` contains the implementation of the baseline `Standalone GPT` model.

`new_privacy_reviews.csv` is the manually curated dataset containing 1,008 privacy-related app reviews after using SAGE on a dataset of 42,271 app reviews.

`supplementary_material.pdf` provides supporting material for the paper.