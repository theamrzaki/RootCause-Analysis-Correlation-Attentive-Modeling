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
from layers.vlinear_arch import OrthTransform

class MSDSMultiModal:
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
        self.cache_id=self.make_hash()
        self.cache_dir=os.path.join(self.data_dir,f"cache_w{self.window}_{self.cache_id}")
        self.metric_p=os.path.join(self.cache_dir,"metrics.pkl")
        self.log_p=os.path.join(self.cache_dir,"logs.pkl")
        self.trace_p=os.path.join(self.cache_dir,"traces.pkl")
        self.window_p=os.path.join(self.cache_dir,"windows.pkl")

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
        df=self.trace_df.sort_values("end_time")

        timestamps=sorted(df["end_time"].unique())
        trace_types=sorted(df["stats"].unique())

        ts2idx={t:i for i,t in enumerate(timestamps)}
        type2idx={t:i for i,t in enumerate(trace_types)}

        X=np.zeros(
            (
                len(timestamps),
                len(self.pods),
                len(self.pods),
                len(trace_types)
            ),
            dtype=np.float32
        )

        for _,row in tqdm(df.iterrows(),total=len(df),desc="traces"):

            src=row["cmbd_id"]
            dst=row["fatherpod"]

            if src not in self.pods or dst not in self.pods:
                continue

            ts=row["end_time"]

            if ts not in ts2idx:
                continue

            tr=row["stats"]

            t=ts2idx[ts]
            i=self.pods.index(src)
            j=self.pods.index(dst)
            k=type2idx[tr]

            X[t,i,j,k]+=row["duration"]

        X=X/(X.mean(axis=0,keepdims=True)*10+1e-6)

        self.trace_tensor=X

        print("trace tensor:",self.trace_tensor.shape)

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

        Xm=self.data_dict["x_metric"]
        Xl=self.data_dict["x_log"]
        Xt=self.data_dict["x_trace"]
        Y=self.data_dict["labels"]

        print("Metric Shape:",Xm.shape)
        print("Log Shape   :",Xl.shape)
        print("Trace Shape :",Xt.shape)
        print("Label Shape :",Y.shape)

        #######################################################################
        # NAN CHECK
        #######################################################################

        assert not np.isnan(Xm).any(),"NaN in metrics"
        assert not np.isnan(Xl).any(),"NaN in logs"
        assert not np.isnan(Xt).any(),"NaN in traces"

        print("NaN Check: Passed")

        #######################################################################
        # VARIANCE CHECK
        #######################################################################

        metric_var=np.var(Xm,axis=(0,1,2))
        dead_metric=np.where(metric_var<1e-12)[0]

        if len(dead_metric)>0:
            print("CRITICAL: Dead metric dims:",dead_metric)
        else:
            print("Metric Variance Check: Passed")

        log_var=np.var(Xl,axis=(0,1,2))
        dead_log=np.where(log_var<1e-12)[0]

        if len(dead_log)>0:
            print("WARNING: Dead log dims:",dead_log[:20])
        else:
            print("Log Variance Check: Passed")

        trace_var=np.var(Xt,axis=(0,1,2,3))
        dead_trace=np.where(trace_var<1e-12)[0]

        if len(dead_trace)>0:
            print("WARNING: Dead trace dims:",dead_trace)
        else:
            print("Trace Variance Check: Passed")

        #######################################################################
        # LABEL COVERAGE
        #######################################################################

        num_anom=np.sum(Y)

        print("Total anomalous labels:",int(num_anom))

        if num_anom==0:
            print("CRITICAL: No anomalies found")
        else:
            print("Anomaly Coverage Check: Passed")

        #######################################################################
        # RANGE CHECK
        #######################################################################

        print(
            "Metric Stats -> "
            f"min={Xm.min():.4f}, "
            f"max={Xm.max():.4f}, "
            f"mean={Xm.mean():.4f}"
        )

        print(
            "Log Stats -> "
            f"min={Xl.min():.4f}, "
            f"max={Xl.max():.4f}, "
            f"mean={Xl.mean():.4f}"
        )

        print(
            "Trace Stats -> "
            f"min={Xt.min():.4f}, "
            f"max={Xt.max():.4f}, "
            f"mean={Xt.mean():.4f}"
        )

        print("--- Sanity Check Passed ---\n")

    ###########################################################################
    # ORTHOGONAL TRANSFORMS
    ###########################################################################

    def apply_orthogonal_transform(self):

        if not self.use_orth:
            return

        print("\nApplying modality-specific orthogonal transforms...")

        #######################################################################
        # METRICS
        #######################################################################

        Xm=torch.from_numpy(
            self.data_dict["x_metric"]
        ).float().to(self.device)

        B,W,N,F=Xm.shape

        Xm_flat=Xm.reshape(B*W*N,F)

        Qm=torch.linalg.qr(torch.randn(F,F,device=self.device))[0]

        Xm_orth=(Xm_flat@Qm).reshape(B,W,N,F)

        self.data_dict["x_metric_orth"]=Xm_orth.cpu().numpy()

        #######################################################################
        # LOGS
        #######################################################################

        Xl=torch.from_numpy(
            self.data_dict["x_log"]
        ).float().to(self.device)

        B,W,N,F=Xl.shape

        Xl_flat=Xl.reshape(B*W*N,F)

        Ql=torch.linalg.qr(torch.randn(F,F,device=self.device))[0]

        Xl_orth=(Xl_flat@Ql).reshape(B,W,N,F)

        self.data_dict["x_log_orth"]=Xl_orth.cpu().numpy()

        #######################################################################
        # TRACES
        #######################################################################

        Xt=torch.from_numpy(
            self.data_dict["x_trace"]
        ).float().to(self.device)

        B,W,N1,N2,F=Xt.shape

        Xt_flat=Xt.reshape(B*W*N1*N2,F)

        Qt=torch.linalg.qr(torch.randn(F,F,device=self.device))[0]

        Xt_orth=(Xt_flat@Qt).reshape(B,W,N1,N2,F)

        self.data_dict["x_trace_orth"]=Xt_orth.cpu().numpy()

        print("Orthogonal transforms complete")

    ###########################################################################
    # SAVE
    ###########################################################################

    def save(self):

        save_dir=os.path.join(
            self.data_dir,
            "processed_multimodal"
        )

        os.makedirs(save_dir,exist_ok=True)

        for k,v in self.data_dict.items():
            np.save(os.path.join(save_dir,f"{k}.npy"),v)

        print(f"Saved to {save_dir}")

    def save_pickle(self, obj, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(obj, f)

    def make_hash(self):
        cfg={
            "window":self.window,
            "step":self.step,
            "pods":self.pods,
            "log_len":self.log_len
        }
        return hashlib.md5(json.dumps(cfg,sort_keys=True).encode()).hexdigest()

    def check_pickle_cache(self):
        return (
            os.path.exists(self.metric_p) and
            os.path.exists(self.log_p) and
            os.path.exists(self.trace_p) and
            os.path.exists(self.window_p)
        )

    def load_from_pickle(self):
        with open(self.metric_p,"rb") as f:
            self.metric_tensor=pickle.load(f)

        with open(self.log_p,"rb") as f:
            self.log_tensor=pickle.load(f)

        with open(self.trace_p,"rb") as f:
            self.trace_tensor=pickle.load(f)

        with open(self.window_p,"rb") as f:
            self.data_dict=pickle.load(f)

        print("Loaded cache:",
            self.metric_tensor.shape,
            self.log_tensor.shape,
            self.trace_tensor.shape)

    ###########################################################################
    # PIPELINE
    ###########################################################################

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


        self.build_windows()

        self.pipeline_sanity_check()

        self.apply_orthogonal_transform()

        self.save()

###############################################################################
# MAIN
###############################################################################

if __name__=="__main__":

    options={

        "data_dir":
        "/home/db2003/Desktop/Amr/MicroService_Twin_Original/data/MSDS-pre",

        "window_size":10,

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

    dataset=MSDSMultiModal(options)

    dataset.generate()