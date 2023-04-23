# LeWiDi
This repository contains the code used in our team's submission for the SemEval 2023 shared task on Learning with Disagreements. The datasets used can be found in the `data_post-competition` directory.

The code is organised into the following directories:
- `models` contains the main models used on the three datasets.
- `scoring` has some notebooks for getting the resultant cross entropy and micro F1 scores and analysing the errors produced by the models.
- `utils` has some basic utility functions needed in addition to the dataloader.

To run training and get the predictions, `python train_armis.py`, `python train_brexit.py` or `python train_md.py` can be used depending on the dataset we want to get predictions for. By default, soft loss is not used. Hence, we must first update `use_soft_loss = True` to use soft loss.

The predicted output logits will be stored in a `predictions` directory as CSV files. The helper functions in the `scoring` directory can be used to get the final TSV files in the format used for the CodaLab submission.
