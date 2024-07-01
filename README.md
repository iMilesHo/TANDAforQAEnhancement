# TANDA Approach for QA System Enhancement

### Objectives:

Explore TANDA (Transfer And Adapt) methodology to improve Question-Answering (QA) systems using pre-trained Transformer models, focusing on sequential fine-tuning techniques.

1. Introduction and Theory (15%)
   •Summarize the TANDA methodology’s principles, focusing on its novelty and the
   rationale behind sequential fine-tuning.
   •Discuss how Transformers’ architecture benefits the TANDA approach.
2. Preparation and Dataset Understanding (15%)
   •Select the ASNQ dataset for the transfer learning step and either WikiQA or TREC-
   QA for domain-specific adaptation.
   •Conduct an exploratory data analysis (EDA) on your chosen datasets to understand
   their structure, content, and challenges.
3. Model Implementation (30%)
   •Implement a baseline Transformer model for comparison.
   •Apply the TANDA two-step fine-tuning process:
   –First, fine-tune on the ASNQ dataset.
   –Then, adapt the model to your chosen domain-specific dataset.
   •Document the modifications made to the original Transformer architecture, if any,
   and justify your choices.
4. Experimental Setup and Evaluation (20%)
   •Describe your experimental setup, including training details (e.g., hyperparameters,
   loss functions, optimization strategy).
   •Evaluate both the baseline and TANDA-enhanced models using appropriate metrics
   (accuracy, F1 score, etc.).
   •Analyze the performance impact of the TANDA approach compared to the baseline
   model.
5. Discussion and Conclusion (20%)
   •Discuss any challenges encountered during implementation and how they were ad-
   dressed.
   •Reflect on the effectiveness of the TANDA approach for QA systems and potential
   areas for improvement.
   •Propose future research directions or applications of the TANDA methodology in
