import sys
import os
import logging
import argparse
import numpy as np

from datasets import linear, lotka_volterra, lorenz96, swat, nonlinear, msds, wadi, batadal
from args import linear_args, lotka_volterra_args, lorenz96_args, swat_args, msds_args, nonlinear_args, wadi_args, batadal_args
from models import aerca
from utils import utils
import warnings
warnings.filterwarnings("ignore")

def main(argv):
    """
    Main function to run the AERCA model on a specified dataset.

    The script supports multiple datasets. It selects the appropriate dataset class,
    argument parser, and log file based on the provided --dataset_name argument.
    If preprocessing_data is set to 1, the dataset is generated and saved; otherwise,
    the existing data is loaded.

    After setting the random seed and loading the data, the AERCA model is instantiated
    with common parameters and then trained and tested accordingly.

    Args:
        argv (list): List of command-line arguments.
    """
    # Preliminary parsing: retrieve the dataset name.
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        '--dataset_name',
        type=str,
        default='linear',
        help='Name of the dataset to run. Options: linear, lotka_volterra, lorenz96, msds, swat, nonlinear'
    )
    pre_args, remaining_args = pre_parser.parse_known_args(argv[1:])
    dataset_name = pre_args.dataset_name.lower()
    print("Selected dataset:", dataset_name)
    # Map dataset names to their configuration: argument parser, dataset class, log file, and slicing flag.
    dataset_mapping = {
        "linear": {
            "args": linear_args.create_arg_parser,
            "dataset_class": linear.Linear,
            "log_file": "linear.log",
            "use_slice": True
        },
        "lotka_volterra": {
            "args": lotka_volterra_args.create_arg_parser,
            "dataset_class": lotka_volterra.LotkaVolterra,
            "log_file": "lotka_volterra.log",
            "use_slice": True
        },
        "lorenz96": {
            "args": lorenz96_args.create_arg_parser,
            "dataset_class": lorenz96.Lorenz96,
            "log_file": "lorenz96.log",
            "use_slice": True
        },
        "msds": {
            "args": msds_args.create_arg_parser,
            "dataset_class": msds.MSDS,
            "log_file": "msds.log",
            "use_slice": False
        },
        "swat": {
            "args": swat_args.create_arg_parser,
            "dataset_class": swat.SWaT,
            "log_file": "swat.log",
            "use_slice": False
        },
        "nonlinear": {
            "args": nonlinear_args.create_arg_parser,
            "dataset_class": nonlinear.Nonlinear,
            "log_file": "nonlinear.log",
            "use_slice": True
        },
        "wadi": {
            "args": wadi_args.create_arg_parser,
            "dataset_class": wadi.WADI,
            "log_file": "wadi.log",
            "use_slice": False
        },
        "batadal": {
            "args": batadal_args.create_arg_parser,
            "dataset_class": batadal.BATADAL,
            "log_file": "batadal.log",
            "use_slice": False
        }
    }

    # Ensure the specified dataset is recognized.
    if dataset_name not in dataset_mapping:
        print("Dataset '{}' not recognized. Available options are: {}"
              .format(dataset_name, list(dataset_mapping.keys())))
        sys.exit(1)

    mapping = dataset_mapping[dataset_name]

    # Set up logging: create a logs directory relative to the current working directory if it doesn't exist.
    logging_dir = os.path.join(os.getcwd(), "logs")
    if not os.path.exists(logging_dir):
        os.makedirs(logging_dir)
    log_file_path = os.path.join(logging_dir, mapping["log_file"])
    logging.basicConfig(
        filename=log_file_path,
        filemode='w',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Parse the remaining command-line arguments using the dataset-specific argument parser.
    parser = mapping["args"]()
    args, unknown = parser.parse_known_args(remaining_args)
    options = vars(args)

    # Set the random seed for reproducibility.
    utils.set_seed(options['seed'])
    print('Set seed: {}, lr: {}'.format(options['seed'], options["lr"]), 'window_size:', options['window_size'], 'encoder_alpha:', options['encoder_alpha'], 'decoder_alpha:', options['decoder_alpha'], 'encoder_gamma:', options['encoder_gamma'], 'decoder_gamma:', options['decoder_gamma'], 'encoder_lambda:', options['encoder_lambda'], 'decoder_lambda:', options['decoder_lambda'], 'beta:', options['beta'])
    print(f"shrinkage :{options['shrinkage']}")
    # Instantiate the dataset class and generate or load data based on the preprocessing flag.
    data_class = mapping["dataset_class"](options)
    if options['preprocessing_data'] == 1:
        print('Preprocessing data: generating and saving new data...')
        data_class.generate_example()
        data_class.save_data()
        orth_transformer = data_class.load_data()
        import hashlib
        sample_data = np.array(data_class.data_dict['x_n_list'][0])
        data_hash = hashlib.md5(sample_data.tobytes()).hexdigest()
        with open(os.path.join(logging_dir, 'data_hash.txt'), 'w') as f:
            f.write(f"Data Hash for first sample: {data_hash}\n")
    else:
        print('Loading existing data...')
        orth_transformer = data_class.load_data()
        import hashlib
        sample_data = np.array(data_class.data_dict['x_n_list'][0])
        data_hash = hashlib.md5(sample_data.tobytes()).hexdigest()
        with open(os.path.join(logging_dir, 'data_hash.txt'), 'a') as f:
            f.write(f"Data Hash for loading sample: {data_hash}\n")

    # Instantiate the iTransformer model using the common set of parameters.
    
    options['orth_transformer'] = orth_transformer
    aerca_model = aerca.AERCA(
        num_vars=options['num_vars'],
        hidden_layer_size=options['hidden_layer_size'],
        num_hidden_layers=options['num_hidden_layers'],
        device=options['device'],
        window_size=options['window_size'],
        stride=options['stride'],
        encoder_gamma=options['encoder_gamma'],
        decoder_gamma=options['decoder_gamma'],
        encoder_lambda=options['encoder_lambda'],
        decoder_lambda=options['decoder_lambda'],
        beta=options['beta'],
        lr=options['lr'],
        epochs=options['epochs'],
        recon_threshold=options['recon_threshold'],
        data_name=options['dataset_name'],
        causal_quantile=options['causal_quantile'],
        root_cause_threshold_encoder=options['root_cause_threshold_encoder'],
        root_cause_threshold_decoder=options['root_cause_threshold_decoder'],
        risk=options['risk'],
        initial_level=options['initial_level'],
        num_candidates=options['num_candidates'],
        graph_structure=data_class.data_dict['graph_structure'] if "graph_structure" in data_class.data_dict else None,
        options=options
    )

    # Training phase: train the model if enabled.
    if options['training_aerca']:
        # Use slicing for training data if required by the dataset configuration.
        if mapping["use_slice"]:
            training_data = data_class.data_dict['x_n_list'][:options['training_size']]
        else:
            training_data = data_class.data_dict['x_n_list']
        print('Start training AERCA model...')
        
        aerca_model._training(training_data)
        print('Done training')

        """
        # Convert each sequence into sliding windows of shape (window_size+1, num_vars)
        batched_training_data = []
        for seq in training_data:
            seq = np.array(seq)  # (seq_len, num_vars)
            # sliding windows along time axis
            for start in range(0, len(seq) - options['window_size']):
                window = seq[start:start + options['window_size'] + 1]
                batched_training_data.append(window)

        # Now batched_training_data is a list of (window_size+1, num_vars) arrays
        print(f"Number of training windows: {len(batched_training_data)}")
        aerca_model._training_batches(batched_training_data, batch_size=1000)
        print('Done training')
        """
        xs_val = training_data[int(0.8 * len(training_data)):]
        aerca_model._get_recon_threshold(xs_val)
        aerca_model._get_root_cause_threshold_encoder(xs_val)
        aerca_model._get_root_cause_threshold_decoder(xs_val)
        if "include_logs_and_traces" in options and options["include_logs_and_traces"]:
            aerca_model._get_root_cause_threshold_encoder_multi_modality(xs_val)
    aerca_model.example_normal_window = data_class.data_dict['x_n_list'][0]
    # Testing phase for causal discovery (applies only if slicing is used).
    if mapping["use_slice"]:
        test_causal = data_class.data_dict['x_n_list'][options['training_size']:]
        print('Start testing AERCA model for causal discovery...')
        #aerca_model._testing_causal_discover(test_causal, data_class.data_dict['causal_struct'])
        print('Done testing for causal discovery')

    # Testing phase for root cause analysis.
    if mapping["use_slice"]:
        test_x_ab = data_class.data_dict['x_ab_list'][options['training_size']:]
        test_label = data_class.data_dict['label_list'][options['training_size']:]
    else:
        test_x_ab = data_class.data_dict['x_ab_list']
        test_label = data_class.data_dict['label_list']
    print('Start testing AERCA model for root cause analysis...')
    # dataset is aiops
    if "include_logs_and_traces" in options and options["include_logs_and_traces"]:
        aerca_model._testing_root_cause_multi_modality(test_x_ab, test_label)
    else:
        if options['dataset_name'] == 'aiops':
            aerca_model._testing_root_cause(test_x_ab, test_label)
            if options['coeff_architecture'] not in ['BARO', 'torai', 'rcd']:
                if "include_logs_and_traces" in options and options["include_logs_and_traces"]:
                    aerca_model._testing_root_cause_services_metrics(test_x_ab, test_label)
        else:
            aerca_model._testing_root_cause(test_x_ab, test_label)
    print('Done testing for root cause analysis')


    # =========================================================
    # CASE STUDY PIPELINE ANALYSIS
    # =========================================================
    if options.get("run_case_study", False):

        print("Running RCA case study analysis...")

        # Example anomalous sample
        case_idx = None

        for idx, lbl in enumerate(test_label):

            lbl_arr = np.asarray(lbl)

            # supports shapes like:
            # (T, P), (1, P), or (P,)
            if np.any(lbl_arr == 1):
                case_idx = idx
                break

        if case_idx is None:
            raise ValueError("No anomalous sample found in test labels.")

        print(f"Selected case study sample: {case_idx}")

        case_x = test_x_ab[case_idx]
        case_label = test_label[case_idx]

        case_result = aerca_model.case_study_rca_pipeline(
            case_x,
            case_label,
            alpha=0.5,
            use_attention_fusion=False
        )

        print("\n===== CASE STUDY RESULTS =====")
        print("True Root Causes:", case_result["true_idx"])
        print("Top-10 Ranking:", case_result["ranking"][:10])

        for k, v in case_result["metrics"].items():
            print(f"{k}: {v:.5f}")

        # Optional saving
        if options.get("save_case_study", True):

            case_save_dir = os.path.join(
                aerca_model.save_dir,
                "case_study"
            )

            os.makedirs(case_save_dir, exist_ok=True)

            np.save(
                os.path.join(case_save_dir, "residual.npy"),
                case_result["residual"]
            )

            np.save(
                os.path.join(case_save_dir, "z_scores.npy"),
                case_result["z_scores"]
            )

            np.save(
                os.path.join(case_save_dir, "z_pot.npy"),
                case_result["z_pot"]
            )

            np.save(
                os.path.join(case_save_dir, "ranking.npy"),
                case_result["ranking"]
            )

            np.save(
                os.path.join(case_save_dir, "labels.npy"),
                case_result["labels"]
            )
            print(f"Case study artifacts saved to: {case_save_dir}")

    print('done')



if __name__ == '__main__':
    main(sys.argv)
