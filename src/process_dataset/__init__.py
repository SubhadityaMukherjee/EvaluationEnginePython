from src.process_dataset.module import load_dataset, generate_folds
from src.process_dataset.splitting import crossvalidation_splits, learning_curve_splits, leave_one_out_splits, \
    train_on_test_splits, holdout_splits, holdout_ordered_splits
