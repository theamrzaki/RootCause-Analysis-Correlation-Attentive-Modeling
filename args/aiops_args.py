import argparse
import os

def create_arg_parser():
    """
    Creates and returns the argument parser for the SWaT dataset.

    Returns:
        argparse.ArgumentParser: The argument parser for the SMD dataset.
    """
    parser = argparse.ArgumentParser(description='SMD')

    # Dataset arguments
    parser.add_argument('--preprocessing_data', type=int, default=1, help='Flag for preprocessing data (default: 1)')
    parser.add_argument('--data_dir', type=str, default='/home/db2003/Desktop/Amr/Tests/Medicine/dataset/aiops22-pre/初赛评分数据', help='Data directory (default: ./datasets/smd)')
    parser.add_argument('--num_vars', type=int, default=30, help='Number of variables (default: 20)')
    parser.add_argument('--causal_quantile', type=float, default=0.70, help='Causal quantile (default: 0.70)')
    parser.add_argument('--shuffle', type=int, default=1, help='Flag for shuffling data (default: 1)')
    parser.add_argument('--main_model', type=str, default='aerca', help='Main model to use (default: aerca)')
    parser.add_argument('--include_logs_and_traces', type=int, default=0, help='Flag for including logs and traces (default: 0)')
    parser.add_argument('--num_log_features', type=int, default=0, help='Number of log features (default: 10)')
    parser.add_argument('--num_trace_features', type=int, default=0, help='Number of trace features (default: 10)')

    # Meta arguments
    parser.add_argument('--seed', type=int, default=4, help='Random seed (default: 4)')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (default: cuda)')
    parser.add_argument('--dataset_name', type=str, default='aiops', help='Dataset name (default: aiops)')

    # AERCA arguments
    parser.add_argument('--window_size', type=int, default=1, help='Window size (default: 1)')
    parser.add_argument('--stride', type=int, default=1, help='Stride (default: 1)')
    parser.add_argument('--encoder_alpha', type=float, default=1.0, help='Encoder alpha (default: 0.5)')
    parser.add_argument('--decoder_alpha', type=float, default=1.0, help='Decoder alpha (default: 0.5)')
    parser.add_argument('--encoder_gamma', type=float, default=0.1, help='Encoder gamma (default: 0.5)')
    parser.add_argument('--decoder_gamma', type=float, default=0.1, help='Decoder gamma (default: 0.5)')
    parser.add_argument('--encoder_lambda', type=float, default=0.5, help='Encoder lambda (default: 0.5)')
    parser.add_argument('--decoder_lambda', type=float, default=0.5, help='Decoder lambda (default: 0.5)')
    parser.add_argument('--beta', type=float, default=0.1, help='Beta (default: 0.5)')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate (default: 0.000001)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs (default: 5000)')
    parser.add_argument('--hidden_layer_size', type=int, default=128, help='Hidden layer size (default: 1000)')
    parser.add_argument('--num_hidden_layers', type=int, default=2, help='Number of hidden layers (default: 8)')
    parser.add_argument('--recon_threshold', type=float, default=0.90, help='Reconstruction threshold (default: 0.95)')
    parser.add_argument('--root_cause_threshold_encoder', type=float, default=0.99, help='Root cause threshold for encoder (default: 0.99)')
    parser.add_argument('--root_cause_threshold_decoder', type=float, default=0.99, help='Root cause threshold for decoder (default: 0.99)')
    parser.add_argument('--training_aerca', type=int, default=1, help='Flag for training AERCA (default: 1)')
    parser.add_argument('--initial_z_score', type=float, default=3.0, help='Initial Z-score (default: 3.0)')
    parser.add_argument('--risk', type=float, default=1e-5, help='Risk (default: 1e-5)')
    parser.add_argument('--initial_level', type=float, default=0.9, help='Initial level (default: 0.00)')
    parser.add_argument('--num_candidates', type=int, default=100, help='Number of candidates (default: 100)')
    parser.add_argument('--early_stopping', type=int, default=0, help='Flag for early stopping (default: 0)')
    parser.add_argument('--mean_std_recon_loss', type=int, default=0, help='Patience for early stopping (default: 50)')
    parser.add_argument('--AMOC_Loss', type=int, default=0, help='Minimum delta for early stopping (default: 1e-4)')

    # Dual KL arguments
    parser.add_argument('--correlated_KL', type=int, default=0, help='Flag for correlated KL (default: 1)')
    parser.add_argument('--lambda_indep', type=float, default=1.0, help='Lambda for independence (default: 1.0)')
    parser.add_argument('--lambda_corr', type=float, default=1.0, help='Lambda for correlated (default: 1.0)')
    parser.add_argument('--shrinkage', type=float, default=0.5, help='Shrinkage factor (default: 0.07)') 
    
    # Architecture arguments
    parser.add_argument('--coeff_architecture', type=str, default='deep_mlp', help='Coefficient architecture options (deep_mlp, gnn_attention) (default: deep_mlp)')
    parser.add_argument('--temporal_mixer', type=int, default=0, help='Flag for using temporal mixer (default: 0)')
    
    # Attention arguments
    parser.add_argument('--global_attention_over_all_lag', type=str)
    parser.add_argument('--local_attention_per_lag', type=int, default=0, help='Flag for using local attention per lag (default: 0)')

    # Results arguments
    parser.add_argument('--results_csv', type=str, default='results.csv', help='Path to the results CSV file (default: results.csv)')
    
    # cross attention arguments
    parser.add_argument('--time_freq_representation', type=str, default='normal', help='Time-frequency representation options (normal, mag_phase, learnable_filter) (default: normal)')
    parser.add_argument('--combine_method', type=str, default='gated', help='Method to combine time and frequency coefficients (gated, attention, average) (default: gated)')

    return parser

if __name__ == "__main__":
    try:
        arg_parser = create_arg_parser()
        args = arg_parser.parse_args()
        print(args)
    except Exception as e:
        print(f"Error parsing arguments: {e}")