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

# Starting Speakers when varying speakers
speakers = [0, 1]
# s Values to explore when varying s
s_values = [3 , 4 , 6 , 8 , 12  , 15 , 18]
# T Values to explore when varying T
T_values = [5 , 10 , 14 , 20 , 25]
# clauses_per_class Values to explore when varying number of clauses
caluses_per_class = [20 , 40 , 80 , 150 , 200 , 500]

# Dictionary holding arguments used with TM, detailed time used for its training and testing, and its accuracy after each epoch
def metrics(args):
    return dict(
        accuracy=[],
        train_time=[],
        test_time=[],
        args=vars(args)
    )


# Trains a single TM for speaker verification task
# The speaker selected for verification is given as integer value in selected_speaker argument
# args specifies the architecture of the TM (T, s and num_clauses)
# Returns results of the Trained TM, as well as the TM object itself for further use in the Combined architecture with the Voting Module
def train_tm(args , selected_speaker):
    experiment_results = metrics(args)

    # Dataset CSV file location - Holds boths training and test data 
    df = pd.read_csv('Datasets/mfcc_booleanized_8.csv')

    # Training masks used to separate speaker data from non-speaker data, later labeled as 1 for speaker and 0 for non-spaker
    speaker_mask = (df['split'] == 'train') & (df['label'].isin([selected_speaker]))
    non_speaker_mask = (df['split'] == 'train') & (~df['label'].isin([selected_speaker]))
    
    # 2️⃣ Separate the two groups
    speaker_df = df[speaker_mask].copy()
    non_speaker_df = df[non_speaker_mask].copy()

    # 3️⃣ Determine how many samples to take from the non-speakers
    n_samples = len(speaker_df)

    # If other speakers have fewer total samples, sample with replacement = True if needed
    non_speaker_sampled = non_speaker_df.sample(n=n_samples, random_state=42, replace=False)

    # 4️⃣ Replace labels
    speaker_df['label'] = 1
    non_speaker_sampled['label'] = 0

    # 5️⃣ Combine and shuffle
    balanced_train_df = pd.concat([speaker_df, non_speaker_sampled], ignore_index=True)
    balanced_train_df = balanced_train_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Drop 'label' and 'split' for features
    X_train = balanced_train_df.drop(['label', 'split'], axis=1).values

    # Keep only the label column for target
    y_train = balanced_train_df['label'].values


    # Testing masks used to separate speaker data from non-speaker data, later labeled as 1 for speaker and 0 for non-spaker
    test_mask_speaker = (df['split'] == 'test') & (df['label'].isin([selected_speaker]))
    test_mask_non_speaker = (df['split'] == 'test') & (~df['label'].isin([selected_speaker]))

    # 2️⃣ Separate the two groups
    speaker_df_test = df[test_mask_speaker].copy()
    non_speaker_df_test = df[test_mask_non_speaker].copy()

    # 3️⃣ Determine how many samples to take from the non-speakers
    n_samples = len(speaker_df_test)

    # If other speakers have fewer total samples, sample with replacement = True if needed
    non_speaker_df_test_sampled = non_speaker_df_test.sample(n=n_samples, random_state=42, replace=False)

    # 4️⃣ Replace labels
    speaker_df_test['label'] = 1
    non_speaker_df_test_sampled['label'] = 0

    # 5️⃣ Combine and shuffle
    balanced_test_df = pd.concat([speaker_df_test, non_speaker_df_test_sampled], ignore_index=True)
    balanced_test_df = balanced_test_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Drop 'label' and 'split' for features
    X_test = balanced_test_df.drop(['label', 'split'], axis=1).values

    # Keep only the label column for target
    y_test = balanced_test_df['label'].values


    # Combine everything in dataset object to feed the TM
    dataset = {
        'x_train': X_train,
        'x_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }


    for key in dataset:
        dataset[key] = np.array(dataset[key], dtype=np.uint32)
    

    # Print Dataset information + Sample
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

    # Initialize the TMClassifier
    tm = TMClassifier(
        type_iii_feedback=True,
        number_of_clauses=args.num_clauses,
        T=args.T,
        s=args.s,
        max_included_literals=args.max_included_literals,
        platform=args.platform,
        weighted_clauses=args.weighted_clauses,
        seed=256,
    )

    # Run training/testing loop for the TM
    _LOGGER.info(f"Running {TMClassifier} for {args.epochs}")
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

            # print(res)
            benchmark2 = BenchmarkTimer(logger=None, text="Testing Time")
            with benchmark2:
                result = 100 * (tm.predict(dataset["x_test"]) == dataset["y_test"]).mean()
                experiment_results["accuracy"].append(result)
            experiment_results["test_time"].append(benchmark2.elapsed())

            _LOGGER.info(f"Epoch: {epoch + 1}, Accuracy: {result:.2f}, Training Time: {benchmark1.elapsed():.2f}s, "
                         f"Testing Time: {benchmark2.elapsed():.2f}s")

        if args.platform == "CUDA":
            CudaProfiler().print_timings(benchmark=benchmark_total)

    return experiment_results , tm

# Main function training all 10 speaker verification TMs - Using the additional Test-Set to test the combined TM architecture
# Returns the results of training and testing individual TMs + Correctly classified test samples by the whole system + Total number of samples
def main():
    # Holds results of training each individual TM
    results = {}
    # Holds the individual TM objects
    tms = []

    # Additional Combined TM testing dataset
    df = pd.read_csv('Datasets/mfcc_booleanized_testing_8.csv')
    # Holds subsets of the Additional TM dataset (one subset for each of the speakers)
    speaker_datasets = {}

    # Main loop training individual TMs by calling train_tm + saving results and TMs + creating the sub-test-datasets for each speaker
    for i in range(0, 10):
        speaker_mask = (df['label'].isin([i]))
        exp_results , tm = train_tm(default_args() , i)
        entry = {
            'timestamp': str(datetime.now()),
            'results': exp_results
        }

        tms.append(tm)
        results[f"speaker - {i}"] = entry

        speaker_df = df[speaker_mask].copy()
        speaker_datasets[f"speaker_{i}"] = speaker_df.drop(["label"] , axis=1).values
    
    correct = 0
    tested = 0
    # input batch_size
    batch_size = 196
    i = 0

    # Voting module 
    for speaker_name, data in speaker_datasets.items():
         # Loop through data in steps of 'batch_size'
        for start in range(0, len(data), batch_size):
            chunk = data[start:start + batch_size]
            # Perform speaker verification for each batch on each of the individual TMs
            prediction = predict_for_audio_sample(chunk , tms)
            tested += 1
            if(prediction == i):
                correct += 1

        i +=1

    return results , correct , tested


# Voting Module - Predicts the class of a sample of size batch_size = 196 over all TMs and class with the majority of votes
def predict_for_audio_sample(sample , tms):
    predictions = [0,0,0,0,0,0,0,0,0,0]
    for i in range(196):
        for j in range(10):
            prediction = tms[j].predict(sample[i])
            print(30 * "=")
            print(f"Prediction is {prediction}")
            print(30 * "=")
            predictions[j] += prediction[0]
        
    print(f"predictions - {predictions}")
    return np.argmax(predictions)


# Default Arguments
def default_args(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_clauses", default=80, type=int)
    parser.add_argument("--T", default=10, type=int)
    parser.add_argument("--s", default=6.0, type=float)
    parser.add_argument("--max_included_literals", default=36, type=int)
    parser.add_argument("--platform", default="CPU", type=str, choices=["CPU", "CPU_sparse", "CUDA"])
    parser.add_argument("--weighted_clauses", default=True, type=bool)
    parser.add_argument("--epochs", default=100, type=int)
    args = parser.parse_args()
    for key, value in kwargs.items():
        if key in args.__dict__:
            setattr(args, key, value)
    return args

# Runs the main loop and saves the results to JSON File
if __name__ == "__main__":
    results , correct , tested = main()
    _LOGGER.info(results)
    
    # Append to file with timestamp in JSON format
    with open('arch_combine_tms_8_bins.json', 'a') as f:
        entry = {
            'End Time': str(datetime.now()),
            'results': results,
            'correct_predictions' : correct,
            'total' : tested,
            'total_accuracy' : correct/tested
        }
        f.write(json.dumps(entry, indent=4) + '\n')