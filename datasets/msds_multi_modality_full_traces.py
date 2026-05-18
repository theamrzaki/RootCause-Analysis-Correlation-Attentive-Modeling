import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import RobustScaler
import torch
import hashlib
import json

#make layers import work
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from layers.vlinear_arch import OrthTransform_multi_modal

class MSDS_multi_modality_full_traces:
    def __init__(self, options):
        self.opt=options
        self.data_dir=options["data_dir"]
        self.window=options.get("window_size",10)
        self.step=options.get("step",1)
        self.log_len=options.get("log_len",256)
        self.use_orth=options.get("use_orth",True)
        self.device=options.get("device","cpu")
        self.data_dict={}


        
    ###########################################################################
    # LOAD
    ###########################################################################

    def load_raw(self):
        print("Loading MSDS...")
        self.metric_df=pd.read_csv(os.path.join(self.data_dir,"metric.csv"))
        self.log_df=pd.read_csv(os.path.join(self.data_dir,"log.csv"))
        self.trace_df=pd.read_csv(os.path.join(self.data_dir,"trace.csv"))
        self.labels=pickle.load(open(os.path.join(self.data_dir,"label.pkl"),"rb")).astype(np.float32)
        self.infer_pods()
        self.cache_dir=os.path.join(self.data_dir,f"cache_w{self.window}_full_traces")
        self.metric_p=os.path.join(self.cache_dir,"metrics.pkl")
        self.log_p=os.path.join(self.cache_dir,"logs.pkl")
        self.trace_p=os.path.join(self.cache_dir,"traces.pkl")

    def infer_pods(self):
        cols=[c for c in self.metric_df.columns if c!="now"]
        pods=sorted(set([c.split("_")[0] for c in cols]))
        self.pods=pods
        print("Inferred pods:",self.pods)

    ###########################################################################
    # METRICS
    ###########################################################################

    def process_metrics(self):
        df=self.metric_df.sort_values("now")
        timestamps=sorted(df["now"].unique())
        metric_cols=[c for c in df.columns if c!="now"]
        X=[]
        for ts in tqdm(timestamps,desc="metrics"):
            row=df[df["now"]==ts].iloc[0]
            nodes=[]
            for pod in self.pods:
                cols=[c for c in metric_cols if pod in c]
                assert len(cols)>0,f"No metric columns found for pod {pod}"
                vals=row[cols].values.astype(np.float32)
                nodes.append(vals)
            X.append(np.stack(nodes,axis=0))
        self.metric_tensor=np.stack(X,axis=0)
        print("metric tensor:",self.metric_tensor.shape)

    ###########################################################################
    # LOGS
    ###########################################################################

    def process_logs(self):
        df=self.log_df.sort_values("@timestamp")
        timestamps=sorted(df["@timestamp"].unique())
        tids=sorted(df["templateid"].unique())
        tid2idx={t:i for i,t in enumerate(tids)}
        ts2idx={t:i for i,t in enumerate(timestamps)}

        X=np.zeros(
            (len(timestamps),len(self.pods),self.log_len),
            dtype=np.float32
        )

        for _,row in tqdm(df.iterrows(),total=len(df),desc="logs"):
            pod=row["Hostname"]
            if pod not in self.pods:
                continue

            tid=row["templateid"]
            if tid not in tid2idx:
                continue

            tidx=tid2idx[tid]
            if tidx>=self.log_len:
                continue

            ts=row["@timestamp"]
            if ts not in ts2idx:
                continue

            t=ts2idx[ts]
            n=self.pods.index(pod)

            X[t,n,tidx]+=1

        X=X/(X.max(axis=0,keepdims=True)+1e-6)

        self.log_tensor=X

        print("log tensor:",self.log_tensor.shape)

    ###########################################################################
    # TRACES
    ###########################################################################
    def process_traces(self):

        df = self.trace_df.sort_values("end_time")

        timestamps = sorted(df["end_time"].unique())
        trace_types = sorted(df["stats"].unique())

        ts2idx = {t: i for i, t in enumerate(timestamps)}
        type2idx = {t: i for i, t in enumerate(trace_types)}

        T = len(timestamps)
        N = len(self.pods)
        K = len(trace_types)

        edge_X = np.zeros((T, N, N, K), dtype=np.float32)

        for _, row in tqdm(df.iterrows(), total=len(df), desc="traces"):

            src = row["cmbd_id"]
            dst = row["fatherpod"]

            if src not in self.pods or dst not in self.pods:
                continue

            ts = row["end_time"]
            if ts not in ts2idx:
                continue

            tr = row["stats"]
            if tr not in type2idx:
                continue

            t = ts2idx[ts]
            k = type2idx[tr]

            i = self.pods.index(src)
            j = self.pods.index(dst)

            edge_X[t, i, j, k] += row["duration"]

        edge_X = edge_X / (np.mean(edge_X, axis=(1,2), keepdims=True) * 10 + 1e-6)

        self.trace_tensor = edge_X
        print("edge trace tensor:", edge_X.shape)

    ###########################################################################
    # WINDOWING
    ###########################################################################

    def build_windows(self):

        T=min(
            len(self.metric_tensor),
            len(self.log_tensor),
            len(self.trace_tensor),
            len(self.labels)
        )

        Xm=self.metric_tensor[:T]
        Xl=self.log_tensor[:T]
        Xt=self.trace_tensor[:T]
        Y=self.labels[:T]

        x_m=[]
        x_l=[]
        x_t=[]
        y=[]

        for i in tqdm(
            range(0,T-self.window,self.step),
            desc="windowing"
        ):
            end=i+self.window

            x_m.append(Xm[i:end])
            x_l.append(Xl[i:end])
            x_t.append(Xt[i:end])

            y.append(Y[end-1])

        x_m=np.array(x_m,dtype=np.float32)
        x_l=np.array(x_l,dtype=np.float32)
        x_t=np.array(x_t,dtype=np.float32)
        y=np.array(y,dtype=np.float32)

        #######################################################################
        # SCALING
        #######################################################################

        B,W,N,Fm=x_m.shape

        scaler=RobustScaler()

        flat=x_m.reshape(-1,Fm)

        scaler.fit(flat)

        flat=scaler.transform(flat)

        x_m=flat.reshape(B,W,N,Fm)

        #######################################################################
        # SAVE
        #######################################################################

        self.data_dict["x_metric"]=x_m
        self.data_dict["x_log"]=x_l
        self.data_dict["x_trace"]=x_t
        self.data_dict["labels"]=y

        print("\nFinal Shapes")
        print("metrics:",x_m.shape)
        print("logs   :",x_l.shape)
        print("traces :",x_t.shape)
        print("labels :",y.shape)

    ###########################################################################
    # SANITY CHECK
    ###########################################################################

    def pipeline_sanity_check(self):

        print("\n--- Starting MSDS Sanity Check ---")

        Xm = self.data_dict["x_metric"]
        Xl = self.data_dict["x_log"]
        Xt = self.data_dict["x_trace"]
        Y  = self.data_dict["labels"]

        print("Metric Shape:", Xm.shape)
        print("Log Shape   :", Xl.shape)
        print("Trace Shape :", Xt.shape)
        print("Label Shape :", Y.shape)

        #######################################################################
        # NAN CHECK
        #######################################################################
        assert not np.isnan(Xm).any(), "NaN in metrics"
        assert not np.isnan(Xl).any(), "NaN in logs"
        assert not np.isnan(Xt).any(), "NaN in traces"

        print("NaN Check: Passed")

        #######################################################################
        # VARIANCE CHECK (METRICS)
        #######################################################################
        metric_var = np.var(Xm, axis=(0,1,2))
        dead_metric = np.where(metric_var < 1e-12)[0]

        if len(dead_metric) > 0:
            print("CRITICAL: Dead metric dims:", dead_metric)
        else:
            print("Metric Variance Check: Passed")

        #######################################################################
        # VARIANCE CHECK (LOGS)
        #######################################################################
        log_var = np.var(Xl, axis=(0,1,2))
        dead_log = np.where(log_var < 1e-12)[0]

        if len(dead_log) > 0:
            print("WARNING: Dead log dims:", dead_log[:20])
        else:
            print("Log Variance Check: Passed")

        #######################################################################
        # VARIANCE CHECK (TRACES - EADRO EDGE TENSOR)
        #######################################################################
        # Xt expected: [B, W, N, N, K]
        trace_var = np.var(Xt, axis=(0,1,2,3))  # -> [K]
        dead_trace = np.where(trace_var < 1e-12)[0]

        if len(dead_trace) > 0:
            print("WARNING: Dead trace channels (K):", dead_trace)
        else:
            print("Trace Variance Check: Passed")

        # extra structural sanity (highly recommended)
        edge_sparsity = np.mean(Xt == 0)
        print(f"Trace Sparsity Ratio: {edge_sparsity:.4f}")

        #######################################################################
        # LABEL COVERAGE
        #######################################################################
        num_anom = np.sum(Y)

        print("Total anomalous labels:", int(num_anom))

        if num_anom == 0:
            print("CRITICAL: No anomalies found")
        else:
            print("Anomaly Coverage Check: Passed")

        #######################################################################
        # RANGE CHECK
        #######################################################################
        print(
            "Metric Stats -> "
            f"min={Xm.min():.4f}, max={Xm.max():.4f}, mean={Xm.mean():.4f}"
        )

        print(
            "Log Stats -> "
            f"min={Xl.min():.4f}, max={Xl.max():.4f}, mean={Xl.mean():.4f}"
        )

        print(
            "Trace Stats -> "
            f"min={Xt.min():.4f}, max={Xt.max():.4f}, mean={Xt.mean():.4f}"
        )

        Y = self.data_dict["labels"]  # (N, 5)
        anomalous_mask = np.any(Y > 0, axis=1)
        anomalous_labels = Y[anomalous_mask]
        print(f"Anomalous samples: {len(anomalous_labels)}")
        print(f"Unique label patterns:\n{np.unique(anomalous_labels, axis=0)}")
        print(f"Per-var anomaly counts: {anomalous_labels.sum(axis=0)}")

        print("--- Sanity Check Passed ---\n")

    ###########################################################################
    # ORTHOGONAL TRANSFORMS
    ###########################################################################

    def apply_orthogonal_transform(self, save_path, device='cpu'):

        os.makedirs(save_path, exist_ok=True)

        self.orth_transformer = OrthTransform_multi_modal(
            dataset_obj=self,
            save_path=save_path,
            time_lag=self.window,
            device=device
        )

        with torch.no_grad():

            xm = torch.from_numpy(self.data_dict['x_metric']).float().to(device)
            xl = torch.from_numpy(self.data_dict['x_log']).float().to(device)
            xt = torch.from_numpy(self.data_dict['x_trace']).float().to(device)

            xm_orth = self.orth_transformer.forward(xm, mode="m")
            xl_orth = self.orth_transformer.forward(xl, mode="l")
            xt_orth = self.orth_transformer.forward(xt, mode="t")

            self.data_dict['x_metric_orth'] = xm_orth.cpu().numpy()
            self.data_dict['x_log_orth'] = xl_orth.cpu().numpy()
            self.data_dict['x_trace_orth'] = xt_orth.cpu().numpy()

        return self.orth_transformer

    ###########################################################################
    # SAVE
    ###########################################################################

    def save(self):
        save_dir = os.path.join(self.data_dir, "processed_multimodal_full_traces")
        os.makedirs(save_dir, exist_ok=True)

        # --------------------------------------------------------
        # 1. Build canonical multimodal tensor (NORMAL ONLY)
        # --------------------------------------------------------
        xm = self.data_dict["x_metric"]
        xl = self.data_dict["x_log"]
        xt = self.data_dict["x_trace"]

        self.data_dict["x_n_list"] = {
            "metric": xm,
            "log": xl,
            "trace": xt
        }

        np.save(
            os.path.join(save_dir, "x_n_list.npy"),
            self.data_dict["x_n_list"],
            allow_pickle=True
        )
        # --------------------------------------------------------
        # 2. Labels (THE true supervision signal)
        # --------------------------------------------------------
        if "labels" in self.data_dict:
            self.data_dict["label_list"] = self.data_dict["labels"]
            np.save(
                os.path.join(save_dir, "label_list.npy"),
                self.data_dict["label_list"]
            )

        # --------------------------------------------------------
        # 3. IMPORTANT: no x_ab_list unless you define anomalies
        # --------------------------------------------------------
        self.data_dict["x_ab_list"] = np.zeros_like(self.data_dict["x_n_list"])
        np.save(
            os.path.join(save_dir, "x_ab_list.npy"),
            self.data_dict["x_ab_list"]
        )

        # --------------------------------------------------------
        # 4. Metadata (CRITICAL for OrthTransform)
        # --------------------------------------------------------
        meta = {
            "md": self.data_dict["x_metric"].shape[-1],
            "ld": self.data_dict["x_log"].shape[-1],
            "td": self.data_dict["x_trace"].shape[-1],
            "window": self.window,
            "step": self.step,
        }

        np.save(os.path.join(save_dir, "meta.npy"), meta)

        print(f"Saved to {save_dir}")


    def save_pickle(self, obj, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(obj, f)

    def check_pickle_cache(self):
        return (
            os.path.exists(self.metric_p) and
            os.path.exists(self.log_p) and
            os.path.exists(self.trace_p) 
        )

    def load_from_pickle(self):
        with open(self.metric_p,"rb") as f:
            self.metric_tensor=pickle.load(f)

        with open(self.log_p,"rb") as f:
            self.log_tensor=pickle.load(f)

        with open(self.trace_p,"rb") as f:
            self.trace_tensor=pickle.load(f)

        print("Loaded cache:",
            self.metric_tensor.shape,
            self.log_tensor.shape,
            self.trace_tensor.shape)

    ###########################################################################
    # PIPELINE
    ###########################################################################
    def load_data(self):
        with open(os.path.join(self.data_dir, "x_n_list.pkl"), "rb") as f:
            self.data_dict["x_n_list"] = pickle.load(f)

        xm = self.data_dict["x_n_list"]["metric"]
        xl = self.data_dict["x_n_list"]["log"]
        xt = self.data_dict["x_n_list"]["trace"]
        self.data_dict["xm"] = xm
        self.data_dict["xl"] = xl
        self.data_dict["xt"] = xt
        self.data_dict['x_ab_list'] = np.load(os.path.join(self.data_dir, 'x_ab_list.npy'))
        self.data_dict['label_list'] = np.load(os.path.join(self.data_dir, 'label_list.npy'))
        orth_matrix_dir = os.path.join(self.data_dir, 'orth_transform_meta')
        #self.pipeline_sanity_check()
        return None#self.apply_orthogonal_transform(save_path=orth_matrix_dir, device='cpu')
    
    def generate(self):

        self.load_raw()

        #check if they were saved to pickle files 
        if self.check_pickle_cache():
            print("Pickle cache found, loading...")
            self.load_from_pickle()
        else:
            print("No pickle cache found, processing raw data...")

            self.process_metrics()
            self.save_pickle(self.metric_tensor, self.metric_p)
    
            self.process_logs()
            self.save_pickle(self.log_tensor, self.log_p)

            self.process_traces()
            self.save_pickle(self.trace_tensor, self.trace_p)


        print("Building windows...")
        self.build_windows()

        print("Performing sanity check...")
        self.pipeline_sanity_check()

        print("Saving results...")
        self.save()

        print("Applying orthogonal transforms...")
        #self.apply_orthogonal_transform(save_path=os.path.join(self.data_dir,"orth_transforms"), device=self.device)


###############################################################################
# MAIN
###############################################################################

if __name__=="__main__":

    options={

        "data_dir":
        "/home/db2003/Desktop/Amr/MicroService_Twin_Original/data/MSDS-pre",

        "window_size":2,

        "step":1,

        "log_len":256,

        "use_orth":True,

        "device":"cpu",

        "pods":[
            "dbservice1",
            "dbservice2",
            "mobservice1",
            "mobservice2",
            "logservice1"
        ]
    }

    dataset=MSDS_multi_modality_full_traces(options)

    dataset.generate()