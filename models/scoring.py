import numpy as np
import statistics
import math

def scoring(data, data_scaled, anomaly, rca="000", amoc=True, rcr=True, lma=True):    
    if amoc:
        anomaly = amoc_segmentation(data_scaled.to_numpy())

    if anomaly >= len(data_scaled):
        return {col: 0 for col in data_scaled.columns}

    scores = {}
    normal_data = data_scaled.iloc[:anomaly]
    anormal_data = data_scaled.iloc[anomaly:]

    for col in data_scaled.columns:
        normal_period = normal_data[col]
        anormal_period = anormal_data[col]

        if rca == "nsigma":
            center, scale = normal_period.mean(), normal_period.std()
        else:
            center = np.median(normal_period)
            scale = np.quantile(normal_period, 0.75) - np.quantile(normal_period, 0.25)

        zscores = np.abs(anormal_period - center)
        zscores = zscores / scale if scale > 0 else zscores

        scores[col] = np.max(zscores) if rca in ["nsigma", "baro"] else np.median(zscores)

    if rcr:
        scores = relative_correlation_ranking(data, data_scaled, scores)
        
    if lma:
        scores = leading_metric_alignment(scores)

    return scores


def amoc_segmentation(data, jump=1):    
    n = data.shape[0]

    S1 = np.cumsum(data, axis=0)
    S2 = np.cumsum(data ** 2, axis=0)

    split_indices = np.arange(jump, n, jump)

    left_counts = split_indices[:, None]
    right_counts = (n - split_indices)[:, None]

    left_means = S1[split_indices - 1] / left_counts
    right_means = (S1[-1] - S1[split_indices - 1]) / right_counts

    left_vars = (S2[split_indices - 1] / left_counts) - (left_means ** 2)
    right_vars = ((S2[-1] - S2[split_indices - 1]) / right_counts) - (right_means ** 2)

    left_costs = np.sum(left_vars, axis=1) * left_counts.flatten()
    right_costs = np.sum(right_vars, axis=1) * right_counts.flatten()
    total_costs = left_costs + right_costs

    best_idx = np.argmin(total_costs)
    best_cp = split_indices[best_idx]

    return best_cp


def relative_correlation_ranking(data, data_scaled, scores):
    scores = {key: value for key, value in scores.items() if not math.isnan(value)}
    
    ranks = sorted(scores, key=scores.get, reverse=True)
    values = list(scores.values())
    mean, std_dev = statistics.mean(values), statistics.stdev(values)

    top_scored_metrics = [x for x in ranks if abs(scores[x] - mean) > 3*std_dev]

    if len(top_scored_metrics) < 3:
        top_scored_metrics = ranks[:3]

    top_scored_metrics_mean = data_scaled[top_scored_metrics].mean(axis=1)

    df_range = data.max() - data.min()
    relative_diff_score = df_range / data.max().replace(0, np.nan)
    correlation_score = data.corrwith(top_scored_metrics_mean).abs()

    return (relative_diff_score * correlation_score).dropna().to_dict()
    

def leading_metric_alignment(scores):
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    ranks = [x[0] for x in sorted_scores]
    
    values = np.array(list(scores.values()))
    mean, std_dev = np.mean(values), np.std(values)
    
    if std_dev > 0:
        index_score = {}
        services = to_services(ranks)
        
        for sr in ranks:
            service = sr.split("_")[0]
            index = services.index(service)
            if index not in index_score:
                index_score[index] = scores[sr]
            if "lat" not in sr and "err" not in sr and abs(scores[sr] - mean) > std_dev:
                index_score[index] -= 1e-6
                scores[sr] = max(index_score[index], scores[sr])

    return scores


import os
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

def prepare_data(datapath):
    data = pd.read_csv(datapath)

    parent_path = os.path.dirname(datapath)

    with open(f"{parent_path}/inject_time.txt", 'r') as file:
        inject_time = file.read().strip()
        try:
            inject_time = int(inject_time)
        except ValueError:
            print("The inject.txt does not contain a valid integer.")

    inject_time = int(inject_time - data['time'][0])

    data = data.ffill()
    data = data.fillna(0)

    columns = data.columns[data.nunique() > 1]
    columns = [x for x in columns if "time" not in x]
    
    data = data[columns]

    x = data.values 
    min_max_scaler = MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    data_scaled = pd.DataFrame(x_scaled, columns=data.columns, index=data.index)

    return data, data_scaled, inject_time


def to_services(ranks):
    _service_ranks = [r.split("_")[0] for r in ranks]
    service_ranks = []
    for s in _service_ranks:
        if s not in service_ranks:
            service_ranks.append(s)
    return service_ranks
