fairness_tradeoff.model

To do: 

**Main Steps**
- Load in packages and the datasets
- Model
    - GPT2
        - Prepare the dataset and build ``TextDataset``
        - Load the pre-trained GPT-2 model and tokenizer
        - Start fine-tuning with ``Trainer``
        - Generate text
    - RNN
        - Load in previously built ``TextDataset``
        - Construct the simple RNN structure
        - Generate text
- Evaluate the model
    - BLEU
    - Grammar Check
    - 
1. Datasets:
    - [x] COMPAS
    - [ ] ADULTS
    - [ ] LAW
    - [ ] HMDA

2. Base Estimators:
    - [x] Logistics Regression
    - [x] Random Forest
    - [x] Gradient Boosting
    - [x] Support Vector Machine
    - [x] Gaussian Naive Bayes
    - [ ] TabTransformer

- [ ] Bias Mitigation:
    - [ ] **Pre-processing**
        - [x] Reweighing
        - [x] Learning Fair Representation

    - [ ] **In-processing**
        - [x] Adversarial Debiasing
        - [x] Exponentiated Gradient Reduction

    - [ ] **Post-processing**
        - [x] Reject Option Classifier
        - [x] Calibrated Equalized Odds

- [ ] Code check:
    - LFR-in same for all base estimators
    - EGR random prediction as stated in LearnFair (https://github.com/fairlearn/fairlearn/blob/main/fairlearn/reductions/_exponentiated_gradient/exponentiated_gradient.py)?
