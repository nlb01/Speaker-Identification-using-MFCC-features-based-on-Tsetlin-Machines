import logging
import argparse

import numpy as np
import pandas as pd

from tmu.models.classification.vanilla_classifier import TMClassifier
from tmu.tools import BenchmarkTimer
from tmu.util.cuda_profiler import CudaProfiler
from datetime import datetime
import json

_LOGGER = logging.getLogger(__name__)

# s Values to explore when varying s
s_values = [3 , 4 , 6 , 8 , 12  , 15 , 18]
# T Values to explore when varying T
T_values = [5 , 10 , 14 , 20 , 25]
# num_clauses Values to explore when varying number of clauses
caluses_per_class = [20 , 40 , 80 , 150 , 200 , 500]


# Dictionary holding arguments used with TM, detailed time used for its training and testing, and its accuracy after each epoch
def metrics(args):
    return dict(
        accuracy=[],
        train_time=[],
        test_time=[],
        args=vars(args)
)
# Prints information about the dataset + first sample in the dataset
def print_dataset_info(dataset):
    print(60 * "=")
    print(type(dataset))
    print(type(dataset['x_train']))
    print(type(dataset['x_train'][0]))
    print(dataset['x_train'].shape)
    print(dataset['x_train'][0])
    classes = [0 , 0 , 0 ,0 ,0 ,0 ,0 ,0 ,0 ,0]
    for i in range(len(dataset["x_train"])):
        classes[dataset["y_train"][i]] = classes[dataset["y_train"][i]] + 1
    print(classes)
    print(60 * "=")


# Build dataset object from the CSV dataset file
def get_dataset():
    # Location of dataset CSV file read as pandas dataframe
    df = pd.read_csv('Datasets/mfcc_booleanized.csv')

    # 2️⃣ Separate the two groups
    train_set = df[df['split'] == 'train']
    test_set = df[df['split'] == 'test']

    # Drop 'label' and 'split' for features
    X_train = train_set.drop(['label', 'split'], axis=1).values
    # Keep only the label column for target
    y_train = train_set['label'].values


    # Drop 'label' and 'split' for features
    X_test = test_set.drop(['label', 'split'], axis=1).values
    # Keep only the label column for target
    y_test = test_set['label'].values


    dataset = {
        'x_train': X_train,
        'x_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }


    for key in dataset:
        dataset[key] = np.array(dataset[key], dtype=np.uint32)
    
    print_dataset_info(dataset)

    return dataset

# Perpare separate datasets when varying speakers
# num_speakers argument is used to determine how many unique labels will be included in the resulting dataset
# Returns new dataset
def prepare_dataset_speakers(num_speakers):
    # df = pd.read_csv('Datasets/mfcc_booleanized.csv')
    df = pd.read_csv('Datasets/mfcc_booleanized.csv')
    selected_speaker = list(range(num_speakers))

    # Training masks used to separate included speakers data from non-included speakers data
    speaker_mask = (df['split'] == 'train') & (df['label'].isin(selected_speaker))
    # Testing masks used to separate included speakers data from non-included speakers data
    test_mask_speaker = (df['split'] == 'test') & (df['label'].isin(selected_speaker))
    
    # 2️⃣ Separate the two groups
    speaker_df = df[speaker_mask]
    speaker_df_test = df[test_mask_speaker]


    # Drop 'label' and 'split' for features
    X_train = speaker_df.drop(['label', 'split'], axis=1).values
    # Keep only the label column for target
    y_train = speaker_df['label'].values

    # Drop 'label' and 'split' for features
    X_test = speaker_df_test.drop(['label', 'split'], axis=1).values
    # Keep only the label column for target
    y_test = speaker_df_test['label'].values

    dataset = {
        'x_train': X_train,
        'x_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }

    for key in dataset:
        dataset[key] = np.array(dataset[key], dtype=np.uint32)

    print_dataset_info(dataset)

    return dataset


# Main function Training a single TM on the provided dataset
# Returns experiment results
def main(args , dataset):
    experiment_results = metrics(args)
    # single TM trained for speaker identification
    tm = TMClassifier(
        type_iii_feedback=False,
        number_of_clauses=args.num_clauses,
        T=args.T,
        s=args.s,
        max_included_literals=args.max_included_literals,
        platform=args.platform,
        weighted_clauses=args.weighted_clauses,
        seed=42,
    )

    _LOGGER.info(f"Running {TMClassifier} for {args.epochs}")

    # Run training/testing loop for the TM 
    for epoch in range(args.epochs):
        benchmark_total = BenchmarkTimer(logger=None, text="Epoch Time")
        with benchmark_total:
            benchmark1 = BenchmarkTimer(logger=None, text="Training Time")
            with benchmark1:
                res = tm.fit(
                    dataset["x_train"].astype(np.uint32),
                    dataset["y_train"].astype(np.uint32),
                    metrics=["update_p"],
                )

            experiment_results["train_time"].append(benchmark1.elapsed())

            benchmark2 = BenchmarkTimer(logger=None, text="Testing Time")
            with benchmark2:
                result = 100 * (tm.predict(dataset["x_test"]) == dataset["y_test"]).mean()
                experiment_results["accuracy"].append(result)
            experiment_results["test_time"].append(benchmark2.elapsed())

            _LOGGER.info(f"Epoch: {epoch + 1}, Accuracy: {result:.2f}, Training Time: {benchmark1.elapsed():.2f}s, "
                         f"Testing Time: {benchmark2.elapsed():.2f}s")

        if args.platform == "CUDA":
            CudaProfiler().print_timings(benchmark=benchmark_total)

    return experiment_results


# Default Arguments
def default_args(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_clauses", default=400, type=int)
    parser.add_argument("--T", default=10, type=int)
    parser.add_argument("--s", default=6.0, type=float)
    parser.add_argument("--max_included_literals", default=12, type=int)
    parser.add_argument("--platform", default="CPU", type=str, choices=["CPU", "CPU_sparse", "CUDA"])
    parser.add_argument("--weighted_clauses", default=True, type=bool)
    parser.add_argument("--epochs", default=60, type=int)
    args = parser.parse_args()
    for key, value in kwargs.items():
        if key in args.__dict__:
            setattr(args, key, value)
    return args


# Save all results to a JSON file
def save_results_to_json(results , filename):
    # Append to file with timestamp in JSON format
    with open(filename, 'a') as f:
        entry = {
            'Time': str(datetime.now()),
            'results': results,
        }
        f.write(json.dumps(entry, indent=4) + '\n')


# Runs the main loop and saves the results to JSON File
# The main loop execution depends on variable_parameter
# Allows performance of 4 separate experiments:
#     0 - Varying s experiment
#     1 - Varying T experiment
#     2 - Varying number of clauses experiment
#     3 - Varying number of speakers experiment
if __name__ == "__main__":
    #0 for varying s , 1 for varying T , 2 for varying clauses per class , 3 for varying number of speakers
    variable_parameter = 3

    match variable_parameter:
        case 0:
            dataset = get_dataset()
            for i in range(len(s_values)):
                results = main(default_args(s=s_values[i]) , dataset)
                _LOGGER.info(results)
                save_results_to_json(results , "variable_s.json")
        case 1:
            dataset = get_dataset()
            for i in range(len(T_values)):
                results = main(default_args(T=T_values[i]) , dataset)
                _LOGGER.info(results)
                save_results_to_json(results , "variable_T.json")
        case 2:
            dataset = get_dataset()
            for i in range(len(caluses_per_class)):
                results = main(default_args(num_clauses=caluses_per_class[i] * 10) , dataset)
                _LOGGER.info(results)
                save_results_to_json(results , "variable_clauses.json")
        case 3:
            for i in range(2 , 11):
                dataset = prepare_dataset_speakers(i)
                results = main(default_args(num_clauses=40 * i) , dataset)
                _LOGGER.info(results)
                save_results_to_json(results , "variable_speakers.json")