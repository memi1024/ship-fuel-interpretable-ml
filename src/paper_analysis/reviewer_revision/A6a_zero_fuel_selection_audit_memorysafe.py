# -*- coding: utf-8 -*-
"""Memory-safe A6a audit for V15 zero-fuel-underway selection."""
from __future__ import annotations
import argparse, json, logging, math
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
try:
    from scipy.stats import ks_2samp
except Exception:
    ks_2samp = None

KEY=["ship_type","pseudo_ship_group_id","trajectory_segment_id","timestamp_utc"]
VARS=["speed_kn","mean_draught_m","trim_m","rel_wind_speed_kn","wave_height_m","wave_period_s","surface_pressure_pa","surface_temperature_c"]
WANT=KEY+["voyage_phase","primary_key_valid","complete_window_flag","speed_kn_raw","speed_kn","fuel_t_10min_raw","fuel_t_10min","fuel_zero_underway_flag","mean_draught_m","trim_m","rel_wind_speed_kn","wave_height_m","wave_period_s","surface_pressure_pa","surface_temperature_c","latitude_deg","longitude_deg","speed_std_3","speed_stable_flag","final_model_ready_flag"]

def num(s):
    x=pd.to_numeric(s,errors="coerce")
    return x.where(np.isfinite(x))

def flag(s): return pd.to_numeric(s,errors="coerce").eq(1)
def rid(df):
    h=pd.util.hash_pandas_object(df[KEY].astype("string").fillna("<NA>"),index=False).astype("uint64")
    return h.map(lambda x:f"{int(x):016x}")
def smd(a,b):
    if len(a)<2 or len(b)<2:return math.nan
    p=math.sqrt((float(np.var(a,ddof=1))+float(np.var(b,ddof=1)))/2)
    return 0.0 if p==0 and np.mean(a)==np.mean(b) else (math.inf if p==0 else float((np.mean(a)-np.mean(b))/p))
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",required=True,type=Path)
    ap.add_argument("--output-dir",required=True,type=Path)
    ap.add_argument("--ship-type",default="container")
    ap.add_argument("--speed-threshold",type=float,default=1.0)
    ap.add_argument("--chunksize",type=int,default=100000)
    a=ap.parse_args(); a.output_dir.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format="%(asctime)s | %(levelname)s | %(message)s",handlers=[logging.StreamHandler(),logging.FileHandler(a.output_dir/"A6a_selection_audit.log",mode="w",encoding="utf-8")],force=True)
    hdr=list(pd.read_csv(a.input,nrows=0,encoding="utf-8-sig").columns)
    req=set(KEY+["speed_kn_raw","fuel_t_10min_raw"]); miss=sorted(req-set(hdr))
    if miss: raise KeyError("Missing required V15 columns: "+", ".join(miss))
    use=[c for c in WANT if c in hdr]
    cnt_ship=defaultdict(lambda:defaultdict(int)); cnt_v=defaultdict(lambda:defaultdict(int)); cnt_m=defaultdict(lambda:defaultdict(int)); cnt_p=defaultdict(lambda:defaultdict(int))
    arr={u:{v:{"r":[],"p":[]} for v in VARS} for u in ["rule_universe","nonfuel_valid_universe"]}
    vst=defaultdict(lambda:defaultdict(lambda:{"rs":0.0,"rn":0,"ps":0.0,"pn":0}))
    total=mismatch=cands=0; first=True; outcand=a.output_dir/"A6a_zero_underway_candidates.csv"
    for k,df in enumerate(pd.read_csv(a.input,usecols=use,chunksize=a.chunksize,encoding="utf-8-sig",low_memory=True),1):
        total+=len(df); df["ship_type"]=df["ship_type"].astype("string").str.lower(); df["a6_row_id"]=rid(df)
        sr=num(df.speed_kn_raw); fr=num(df.fuel_t_10min_raw); elig=sr.gt(a.speed_threshold)&fr.notna()&fr.ge(0); rem=elig&fr.eq(0); pos=elig&fr.gt(0)
        nf=pd.Series(True,index=df.index)
        if "primary_key_valid" in df:nf&=flag(df.primary_key_valid)
        if "complete_window_flag" in df:nf&=flag(df.complete_window_flag)
        if "speed_kn" in df:nf&=num(df.speed_kn).notna()
        nfu=elig&nf
        if "fuel_zero_underway_flag" in df:mismatch+=int((flag(df.fuel_zero_underway_flag)!=rem).sum())
        df["rule_removed_recomputed"]=rem; df["rule_retained_positive"]=pos; df["a6_nonfuel_valid"]=nf
        df["calendar_month"]=pd.to_datetime(df.timestamp_utc,errors="coerce",utc=True).dt.strftime("%Y-%m")
        for uname,um in [("rule_universe",elig),("nonfuel_valid_universe",nfu)]:
            g=df.loc[um].groupby("ship_type",dropna=False).agg(n=("ship_type","size"),r=("rule_removed_recomputed","sum"),p=("rule_retained_positive","sum"))
            for ship,row in g.iterrows():
                z=cnt_ship[(uname,str(ship))]; z["eligible_underway"]+=int(row.n); z["zero_underway_candidates"]+=int(row.r); z["positive_fuel_underway"]+=int(row.p)
        prim=df.ship_type.eq(a.ship_type.lower()); pm=prim&nfu
        for col,store in [("pseudo_ship_group_id",cnt_v),("calendar_month",cnt_m)]+([("voyage_phase",cnt_p)] if "voyage_phase" in df else []):
            g=df.loc[pm].groupby(col,dropna=False).agg(n=(col,"size"),r=("rule_removed_recomputed","sum"),p=("rule_retained_positive","sum"))
            for key,row in g.iterrows():
                z=store[str(key)]; z["eligible_underway"]+=int(row.n); z["zero_underway_candidates"]+=int(row.r); z["positive_fuel_underway"]+=int(row.p)
        for uname,um in [("rule_universe",elig&prim),("nonfuel_valid_universe",nfu&prim)]:
            for v in VARS:
                if v not in df: continue
                x=num(df[v]); aa=x[um&rem&x.notna()].to_numpy(float); bb=x[um&pos&x.notna()].to_numpy(float)
                if len(aa):arr[uname][v]["r"].append(aa)
                if len(bb):arr[uname][v]["p"].append(bb)
        sub=df.loc[pm]
        for vessel,g in sub.groupby("pseudo_ship_group_id",dropna=False):
            for v in VARS:
                if v not in g:continue
                x=num(g[v]); r=x[g.rule_removed_recomputed&x.notna()]; p=x[g.rule_retained_positive&x.notna()]; st=vst[str(vessel)][v]
                if len(r):st["rs"]+=float(r.sum());st["rn"]+=len(r)
                if len(p):st["ps"]+=float(p.sum());st["pn"]+=len(p)
        cand=df.loc[rem].copy(); cands+=len(cand)
        if len(cand):
            cols=["a6_row_id"]+use+["rule_removed_recomputed","rule_retained_positive","a6_nonfuel_valid","calendar_month"]
            cols=list(dict.fromkeys([c for c in cols if c in cand]))
            cand[cols].to_csv(outcand,mode="w" if first else "a",header=first,index=False,encoding="utf-8-sig" if first else "utf-8"); first=False
        if k%10==0: logging.info("chunks=%d rows=%s candidates=%s mismatches=%s",k,f"{total:,}",f"{cands:,}",f"{mismatch:,}")
    def ctable(store,keyname,file):
        rows=[]
        for key,z in sorted(store.items()):
            n=z["eligible_underway"];rows.append({keyname:key,**z,"zero_candidate_pct":100*z["zero_underway_candidates"]/n if n else np.nan})
        pd.DataFrame(rows).to_csv(a.output_dir/file,index=False,encoding="utf-8-sig")
    rows=[]
    for (u,s),z in sorted(cnt_ship.items()):
        n=z["eligible_underway"];rows.append({"universe":u,"ship_type":s,**z,"zero_candidate_pct":100*z["zero_underway_candidates"]/n if n else np.nan})
    pd.DataFrame(rows).to_csv(a.output_dir/"A6a_rule_counts_by_ship_type.csv",index=False,encoding="utf-8-sig")
    ctable(cnt_v,"pseudo_ship_group_id","A6a_counts_by_vessel.csv");ctable(cnt_m,"calendar_month","A6a_counts_by_month.csv")
    if cnt_p:ctable(cnt_p,"voyage_phase","A6a_counts_by_phase.csv")
    rows=[]
    for u in arr:
        for v in VARS:
            aa=np.concatenate(arr[u][v]["r"]) if arr[u][v]["r"] else np.array([]);bb=np.concatenate(arr[u][v]["p"]) if arr[u][v]["p"] else np.array([])
            d={"universe":u,"ship_type":a.ship_type.lower(),"variable":v,"n_removed":len(aa),"n_retained":len(bb),"removed_mean":np.mean(aa) if len(aa) else np.nan,"retained_mean":np.mean(bb) if len(bb) else np.nan,"removed_median":np.median(aa) if len(aa) else np.nan,"retained_median":np.median(bb) if len(bb) else np.nan,"smd_removed_minus_retained":smd(aa,bb),"ks_statistic":np.nan,"ks_pvalue":np.nan}
            if ks_2samp is not None and len(aa) and len(bb): q=ks_2samp(aa,bb);d["ks_statistic"]=float(q.statistic);d["ks_pvalue"]=float(q.pvalue)
            rows.append(d)
    pd.DataFrame(rows).to_csv(a.output_dir/"A6a_removed_vs_retained_continuous.csv",index=False,encoding="utf-8-sig")
    rows=[]
    for v in VARS:
        ds=[]
        for ship,b in vst.items():
            z=b[v]
            if z["rn"] and z["pn"]:ds.append(z["rs"]/z["rn"]-z["ps"]/z["pn"])
        x=np.array(ds,float);rows.append({"ship_type":a.ship_type.lower(),"variable":v,"n_vessels_with_both_groups":len(x),"mean_vessel_difference_removed_minus_retained":np.mean(x) if len(x) else np.nan,"median_vessel_difference_removed_minus_retained":np.median(x) if len(x) else np.nan,"vessels_positive_difference":int((x>0).sum()),"vessels_negative_difference":int((x<0).sum())})
    pd.DataFrame(rows).to_csv(a.output_dir/"A6a_vessel_balanced_effects.csv",index=False,encoding="utf-8-sig")
    (a.output_dir/"A6a_manifest.json").write_text(json.dumps({"task":"A6a_zero_fuel_selection_audit_memorysafe","input":str(a.input.resolve()),"input_size_bytes":a.input.stat().st_size,"rows_scanned":total,"rule":f"fuel_t_10min_raw == 0 AND speed_kn_raw > {a.speed_threshold:g} kn","primary_ship_type":a.ship_type.lower(),"chunksize":a.chunksize,"source_flag_mismatch_count":mismatch,"zero_underway_candidates_all_ship_types":cands,"scipy_ks_available":ks_2samp is not None,"note":"Selection audit only; canonical dataset unchanged."},ensure_ascii=False,indent=2),encoding="utf-8")
    logging.info("A6a COMPLETE | rows=%s | candidates=%s | flag mismatches=%s | output=%s",f"{total:,}",f"{cands:,}",f"{mismatch:,}",a.output_dir)
if __name__=="__main__":main()
