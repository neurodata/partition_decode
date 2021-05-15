This is a developed from starting kit for Predicting Generalization in Deep Learning challenge at NeurIPS 2020.

Prerequisites:
- Python 3.6.6
- Tensorflow 2.2
- pandas
- pyyaml
- scikit-learn

***Main code for our algorithm can be viewed in sample_code_submission/internal_rep***

Usage:

- to run the code properly (double check you are running the correct version of python):

  `python ingestion_program/ingestion.py sample_data sample_result_submission ingestion_program sample_code_submission`

- if you wish to test on the larger public data, download the public data and run:

  `python ingestion_program/ingestion.py **path/to/public/inptu_data** sample_result_submission ingestion_program sample_code_submission`

- if you wish to compute the score of your submission locally, you can run the scoring program:

  `python scoring_program/score.py **path/to/public/reference_data** **path/to/prediction** **path/to/output**`

The `sample_code_submission` directory contains other baselines that you may use too. 