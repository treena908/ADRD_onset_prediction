import sklearn
from collections import Counter
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
# from typing import Union, List, Tuple
# import torch.distributed as dist
# from torch.nn.parallel import DistributedDataParallel as DDP
# from transformers import AutoModelForCausalLM, get_scheduler
# from torch.optim import AdamW
from transformers.modeling_outputs import SequenceClassifierOutput
# from hf_ehr.data.tokenization import CLMBRTokenizer
# from hf_ehr.config import Event, SPLIT_TRAIN_CUTOFF, SPLIT_VAL_CUTOFF, SPLIT_SEED
import json
from sklearn.metrics import roc_auc_score,confusion_matrix, precision_score,f1_score,average_precision_score,balanced_accuracy_score,brier_score_loss,recall_score
from sklearn.calibration import calibration_curve
# from hf_ehr.data.datasets import MEDSDataset
from huggingface_hub import login
from torch.utils.data import Dataset
from tqdm import tqdm
import argparse
from datetime import datetime
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
# import polars as pl
# import meds_reader
# import os
# import torch, gc
# import time
# import csv
# import meds
import torch, numpy as np, random
from collections import defaultdict
import torch
import datetime
# import meds
from xgboost import XGBClassifier
import lightgbm as lgb
from pathlib import Path
import subprocess
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import OneHotEncoder

# unified global unknown-code dictionary
UNKNOWN_CODE_DICT = defaultdict(
    lambda: {
        "freq": 0,
        "pids": set(),
    }
)
BIRTH_CODES = {"MEDS_BIRTH", "SNOMED/184099003", "SNOMED/3950001"}
# param_grid_l2 = [
#     {"C": 0.001, "penalty": "l2", "solver": "liblinear", "class_weight": "balanced","l1_ratio":None},
#
#
#
#
#     {"C": 0.001, "penalty": "l2", "solver": "liblinear", "class_weight": None,"l1_ratio":None},
#
#
#
#
# ]
# param_grid_elasticnet = [
#
# {"C": 0.001, "penalty": "elasticnet", "solver": "saga", "class_weight": "balanced","l1_ratio":0.1},
#     {"C": 0.01,  "penalty": "elasticnet", "solver": "saga", "class_weight": "balanced","l1_ratio":0.1},
#     {"C": 0.001, "penalty": "elasticnet", "solver": "saga", "class_weight": None,"l1_ratio":0.1},
#     {"C": 0.01,  "penalty": "elasticnet", "solver": "saga", "class_weight": None,"l1_ratio":0.1},
# {"C": 0.001, "penalty": "elasticnet", "solver": "saga", "class_weight": "balanced","l1_ratio":0.3},
#     {"C": 0.01,  "penalty": "elasticnet", "solver": "saga", "class_weight": "balanced","l1_ratio":0.3},
#     {"C": 0.001, "penalty": "elasticnet", "solver": "saga", "class_weight": None,"l1_ratio":0.3},
#     {"C": 0.01,  "penalty": "elasticnet", "solver": "saga", "class_weight": None,"l1_ratio":0.3},
#     {"C": 0.001, "penalty": "elasticnet", "solver": "saga", "class_weight": "balanced","l1_ratio":0.5},
#     {"C": 0.01,  "penalty": "elasticnet", "solver": "saga", "class_weight": "balanced","l1_ratio":0.5},
#     {"C": 0.001, "penalty": "elasticnet", "solver": "saga", "class_weight": None,"l1_ratio":0.5},
#     {"C": 0.01,  "penalty": "elasticnet", "solver": "saga", "class_weight": None,"l1_ratio":0.5}]
param_grid_xgboost = [
    {
        "max_depth": 2,
        "learning_rate": 0.03,
        "n_estimators": 300,
        "subsample": 0.8,
        "colsample_bytree": 0.5,
    }
]

param_grid_lightgbm = [
    {
        "num_leaves": 15,
        "learning_rate": 0.03,
        "n_estimators": 300,
        "subsample": 0.8,
        "colsample_bytree": 0.5,
    }
]
# param_grid_elasticnet = [
#
#
#     {"C": 0.01,  "penalty": "elasticnet", "solver": "saga", "class_weight": "balanced","l1_ratio":0.1},
#     {"C": 0.001, "penalty": "elasticnet", "solver": "saga", "class_weight": None,"l1_ratio":0.1},
#     {"C": 0.01,  "penalty": "elasticnet", "solver": "saga", "class_weight": None,"l1_ratio":0.1},
# {"C": 0.001, "penalty": "elasticnet", "solver": "saga", "class_weight": "balanced","l1_ratio":0.3},
#     {"C": 0.01,  "penalty": "elasticnet", "solver": "saga", "class_weight": "balanced","l1_ratio":0.3},
#     {"C": 0.001, "penalty": "elasticnet", "solver": "saga", "class_weight": None,"l1_ratio":0.3},
#     {"C": 0.01,  "penalty": "elasticnet", "solver": "saga", "class_weight": None,"l1_ratio":0.3}
#
# ]
##################subcohort_analysis#########################
import matplotlib.pyplot as plt
from pathlib import Path
def make_reliability_analysis_df(X_test,y_test,model_path,model_family,classifier,cohort_definition,offset_month):

    # ---------------------------------------
    # Representative models
    # ---------------------------------------





    model = joblib.load(model_path)

    probs = model.predict_proba(X_test)[:, 1]

    frac_pos, mean_pred = calibration_curve(
        y_test,
        probs,
        n_bins=10,
        strategy="quantile",
    )

    for i, (mp, fp) in enumerate(zip(mean_pred, frac_pos), start=1):
        calibration_row={

            "model_family": model_family,
            "classifier":classifier,
            "cohort_definition":cohort_definition,
            "offset_month":offset_month,
            "bin": i,
            "mean_predicted_probability": mp,
            "observed_event_rate": fp,
        }
        log_row_dict('./figures/reliability_analysis/calibration_curve_test.csv', calibration_row)
        print('calibration_curve_test saved')


    summary_row={

        "model_family": model_family,
        "classifier": classifier,
        "cohort_definition": cohort_definition,
        "offset_month": offset_month,
        "auroc": roc_auc_score(y_test, probs),
        "brier_score": brier_score_loss(y_test, probs),
    }
    print(summary_row)
    log_row_dict('./figures/reliability_analysis/calibration_summary.csv', summary_row)
    print('calibration_curve_test saved')





# ######################load_trained_model_from_joblib###############
def make_sensitivity_analysis_df(X_test,y_test,model_path,model_name,classifier,cohort_definition,offset_month,cv_3_threshold):
    # ------------------------
    # Load trained model
    # ------------------------

    model = joblib.load(model_path)

    # probabilities
    probs = model.predict_proba(X_test)[:, 1]

    thresholds = np.arange(0.25, 0.61, 0.01)

    # rows = []

    for thr in thresholds:
        pred = (probs >= thr).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_test,
            pred
        ).ravel()
        row={
            "model_name": model_name,
            "classifier": classifier,
            "cohort_definition": cohort_definition,
            "offset_month": offset_month,
            "cv_3_threshold": cv_3_threshold,


            "threshold": thr,

            "recall":
                recall_score(y_test, pred),

            "precision":
                precision_score(
                    y_test,
                    pred,
                    zero_division=0
                ),

            "f1":
                f1_score(
                    y_test,
                    pred,
                    zero_division=0
                ),

            "balanced_accuracy":
                balanced_accuracy_score(
                    y_test,
                    pred
                ),

            "specificity":
                tn / (tn + fp),

            "ppv":
                tp / (tp + fp) if tp + fp > 0 else 0,

            "npv":
                tn / (tn + fn),

            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn
        }

        # rows.append(row["balanced_accuracy"])
        print(row)
        log_row_dict('./figures/sensitivity_analysis/test_threshold_metrics.csv', row)
        print("test_threshold_metrics saved")

    # metric_df = pd.DataFrame(rows)
####################################################################
def plot_subgroup_bars(args,
    subgroup_results,
    out_dir="./subgroup_plots",
    metrics=("auroc", "auprc"),
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []

    for subgroup_col in subgroup_results["subgroup_type"].unique():
        df = (
            subgroup_results[
                subgroup_results["subgroup_type"] == subgroup_col
            ]
            .copy()
            .sort_values("subgroup")
        )

        for metric in metrics:
            plot_df = df.dropna(subset=[metric])

            plt.figure(figsize=(8, 5))
            plt.bar(
                plot_df["subgroup"].astype(str),
                plot_df[metric],
            )
            plt.ylim(0, 1)
            plt.xlabel(subgroup_col)
            plt.ylabel(metric.upper())
            plt.title(f"{metric.upper()} by {subgroup_col}")
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()

            out_path = out_dir / f"{subgroup_col}_{metric}_{args.model_dir}_{args.classifier}.png"
            plt.savefig(out_path, dpi=300)
            plt.close()

            saved_paths.append(str(out_path))
            print("Saved:", out_path)

    return saved_paths
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score


def bootstrap_metric(
    y_true,
    y_prob,
    metric="auroc",
    n_bootstrap=1000,
    ci=95,
    random_state=42,
):
    """
    Bootstrap confidence interval for AUROC or AUPRC.
    """

    rng = np.random.default_rng(random_state)

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    scores = []

    for _ in range(n_bootstrap):

        idx = rng.integers(0, len(y_true), len(y_true))

        yt = y_true[idx]
        yp = y_prob[idx]

        # AUROC/AP requires both classes
        if len(np.unique(yt)) < 2:
            continue

        if metric == "auroc":
            score = roc_auc_score(yt, yp)
        elif metric == "auprc":
            score = average_precision_score(yt, yp)
        else:
            raise ValueError(metric)

        scores.append(score)

    scores = np.asarray(scores)

    alpha = (100 - ci) / 2

    return (
        scores.mean(),
        np.percentile(scores, alpha),
        np.percentile(scores, 100 - alpha),
    )
def evaluate_subcohort(args,
    df,
    subgroup_col,
    min_cases=20,
):
    rows = []

    for grp, g in df.groupby(subgroup_col):

        n = len(g)

        n_cases = int(g["y_true"].sum())

        n_controls = n - n_cases

        if (
            n_cases < min_cases
            or n_controls < min_cases
        ):
            print(f"{n_cases} < {min_cases} for {grp}")
            continue
            # auroc = np.nan
            # auprc = np.nan
            # # recall = np.nan
            # mpp=np.nan
            # case_prevalence=np.nan
            # pp_25 = np.nan
            # pp_50 = np.nan
            # pp_75 = np.nan




        else:
            auroc,auroc_ci_lower,auroc_ci_upper=bootstrap_metric(
                g["y_true"],
                g["y_prob"],
                metric="auroc",
                n_bootstrap=1000,
                ci=95,
                random_state=42,
            )
            auprc, auprc_ci_lower, auprc_ci_upper = bootstrap_metric(
                g["y_true"],
                g["y_prob"],
                metric="auprc",
                n_bootstrap=1000,
                ci=95,
                random_state=42,
            )
            # auroc = sklearn.metrics.roc_auc_score(
            #     g["y_true"],
            #     g["y_prob"],
            # )
            # recall = threshold_metrics(g["y_true"],
            #     g["y_prob"], threshold=0.5)['recall']

            # auprc = average_precision_score(
            #     g["y_true"],
            #     g["y_prob"],
            # )
            mpp=g["y_prob"].mean()
            case_prevalence=n_cases / n
            # prob_summary = g["y_prob"].describe(
            #     percentiles=[0.25, 0.5, 0.75]
            # )
            # pp_25= prob_summary["25%"]
            # pp_50= prob_summary["50%"]
            # pp_75= prob_summary["75%"]



        model_name = define_model_name(args)
        classifier_name = args.classifier if args.lr_penalty == 'l2' else args.classifier + "_" + args.lr_penalty
        # print(f"recall: {recall}, auprc: {auprc}")
        row={
            "subgroup_type": subgroup_col,
            "subgroup": grp,
            "n": n,
            "n_cases": n_cases,
            "n_controls": n_controls,
            "case_rate": n_cases / n,
            "auroc": auroc,"auroc_ci_lower":auroc_ci_lower,"auroc_ci_upper":auroc_ci_upper,
            "auprc": auprc,"auprc_ci_lower":auprc_ci_lower,"auprc_ci_upper":auprc_ci_upper,
            # "recall": recall,
            "model_name": model_name,
            "classifier": classifier_name,
            "cohort_definition": args.cohort_definition,
            "offset_month": args.label_col,
            "mpp": mpp,
            "case_prevalence": case_prevalence,
            # "pp_25": pp_25,
            # "pp_50": pp_50,
            # "pp_75": pp_75
        }
        rows.append(row)
        log_row_dict( './checkpoints/subgroup_results_'  + 'v3.csv', row)
        print("saved_subgroup_results_v3.csv")

    return pd.DataFrame(rows)

def make_subcohort_df(meta_df, X_test, y_test,pid_test, model_path, model_family,
                                                    classifier,
                                                 cohort_definition, offset_month):
    model = joblib.load(model_path)

    probs_test = model.predict_proba(X_test)[:, 1]
    test_df = pd.DataFrame({
        "person_id": pid_test.astype(int),
        "y_true": y_test,
        "y_prob": probs_test,
    })

    sub_meta = meta_df.copy()

    # ------------------------
    # Sex
    # ------------------------

    sub_meta["sex_group"] = sub_meta["gender"].replace({
        "PMI: Skip": np.nan,
        "No matching concept": np.nan,
        "Sex At Birth: Sex At Birth None Of These": np.nan,
    })

    # ------------------------
    # Race
    # ------------------------

    sub_meta["race_group"] = "Other"

    sub_meta.loc[
        sub_meta["race"] == "White",
        "race_group"
    ] = "White"

    sub_meta.loc[
        sub_meta["race"] == "Black or African American",
        "race_group"
    ] = "Black"

    # sub_meta.loc[
    #     sub_meta["race"] == "Asian",
    #     "race_group"
    # ] = "Asian"

    # ------------------------
    # Age group
    # ------------------------

    sub_meta["age_at_index"] = (
        pd.to_datetime(
            sub_meta["final_index_date"],
            utc=True,
        )
        -
        pd.to_datetime(
            sub_meta["birth_datetime"],
            utc=True,
        )
    ).dt.days / 365.25

    sub_meta["age_group"] = pd.cut(
        sub_meta["age_at_index"],
        bins=[0, 65, 75, 85, 200],
        labels=["<65", "65-74", "75-84", "85+"],
        right=False,
    )

    # ------------------------
    # Merge
    # ------------------------

    test_df = test_df.merge(
        sub_meta[
            [
                "person_id",
                "sex_group",
                "race_group",
                "age_group",
                f"history_bin_{offset_month}m",
            ]
        ],
        on="person_id",
        how="left",
    )
    return test_df
def run_subcohort_analysis(args,test_df):
    print("######Running Subcohort Analysis#########")
    results = []

    for subgroup_col in [
        "sex_group",
        "race_group",
        "age_group",
        f"history_bin_{args.label_col}m",
    ]:
        results.append(
            evaluate_subcohort(args,
                test_df,
                subgroup_col,
            )
        )

    subgroup_results = pd.concat(
        results,
        ignore_index=True,
    )
    for c in subgroup_results.columns:
        if subgroup_results[c].dtype == "object":
            subgroup_results[c] = subgroup_results[c].astype(str)
    # save_df_local_gcs(subgroup_results,'subgroup_results_v1','./checkpoints/')
    col=["subgroup","subgroup_type","n_cases","auroc","auprc","case_prevalence","mpp","model_name","offset_month"]
    for subgroup_col in [
        "sex_group",
        "race_group",
        "age_group",
        f"history_bin_{args.label_col}m",
    ]:
        print("\n")
        print("=" * 80)
        print(subgroup_col)

        print(
            subgroup_results[
                subgroup_results["subgroup_type"]
                == subgroup_col
                ][col]
            .sort_values("subgroup")
        )

#########################count_featurizer###################
def clean_sex(x):
    if pd.isna(x):
        return "Unknown"

    x = str(x).strip()

    if x == "Male":
        return "Male"

    if x == "Female":
        return "Female"

    return "Unknown"


def clean_race(x):
    if pd.isna(x):
        return "Unknown"

    x = str(x).strip()

    unknown_vals = {
        "PMI: Skip",
        "I prefer not to answer",
        "None Indicated",
        "None of these",
        "No matching concept",
    }

    if x in unknown_vals:
        return "Unknown"

    return x


def clean_ethnicity(x):
    if pd.isna(x):
        return "Unknown"

    x = str(x).strip()

    if x == "Hispanic or Latino":
        return "Hispanic"

    if x == "Not Hispanic or Latino":
        return "NonHispanic"

    return "Unknown"


class TimeBinnedCountFeaturizer:
    def __init__(
            self,
            time_bins=None,
            pid_col="person_id",
            event_date_col="event_date",
            measurements_col="measurements",
            prediction_time_col="final_index_date",
            birth_col="birth_datetime",
            race_col="race",
            ethnicity_col="ethnicity",
            gender_col="gender",
            offset_month=6,
            use_binary_counts=True,
            add_utilization=True,
            add_demographics=True,demographics_only=False
    ):
        self.time_bins = time_bins or [
            ("0_24h", 0, 1),
            ("1_7d", 1, 7),
            ("8_30d", 8, 30),
            ("31d_any", 31, np.inf),
        ]

        self.pid_col = pid_col
        self.event_date_col = event_date_col
        self.measurements_col = measurements_col
        self.prediction_time_col = prediction_time_col

        self.birth_col = birth_col
        self.race_col = race_col
        self.ethnicity_col = ethnicity_col
        self.gender_col = gender_col
        self.offset_month = offset_month

        self.use_binary_counts = use_binary_counts
        self.add_utilization = add_utilization
        self.add_demographics = add_demographics
        self.demographics_only=demographics_only


        self.vectorizer = DictVectorizer(sparse=True)

    def _time_bin(self, days_before):
        for name, lo, hi in self.time_bins:
            if lo <= days_before <= hi:
                return name
        return None

    def _measurement_feats(self, m):
        feats = {}

        code = m.get("code")
        if code is None:
            return feats

        domain = m.get("domain", "unknown")
        val = m.get("numeric_value")
        low = m.get("low")
        high = m.get("high")

        base = f"{domain}::{code}"

        feats[f"code::{base}"] = 1

        try:
            val = float(val) if val is not None else None
            low = float(low) if low is not None else None
            high = float(high) if high is not None else None
        except Exception:
            val, low, high = None, None, None

        if domain == "measurement" and val is not None:
            feats[f"measurement_present::{code}"] = 1

            if low is not None and val < low:
                feats[f"measurement_low::{code}"] = 1

            if high is not None and val > high:
                feats[f"measurement_high::{code}"] = 1

        return feats

    def _patient_to_dict(self, patient_df,
                         prediction_time,
                         birth_datetime,
                         race,
                         ethnicity,
                         gender):
        feats = {}

        prediction_time = pd.to_datetime(prediction_time, utc=True)
        # print(f"############ before Prediction time########: {prediction_time}\n")
        # prediction_time = (
        #         pd.to_datetime(
        #             prediction_time,
        #             errors="coerce",
        #             utc=True,
        #         )
        #         - pd.DateOffset(months=self.offset_month)
        # )
        # print(f"############ after Prediction time: {prediction_time} at offset {self.offset_month}\n")
        patient_df = patient_df.copy()
        patient_df[self.event_date_col] = pd.to_datetime(
            patient_df[self.event_date_col],
            errors="coerce",
            utc=True,
        )
        #prediction_time is already offset by offset_month
        patient_df = patient_df[
            patient_df[self.event_date_col] <= prediction_time
            ]
        if self.add_demographics:

            birth = pd.to_datetime(
                birth_datetime,
                errors="coerce",
                utc=True,
            )

            if pd.notna(birth) and pd.notna(prediction_time):

                age = (
                              prediction_time - birth
                      ).days / 365.25

                if age >= 0:
                    age_bin = int(age // 5 * 5)
                    feats[f"age_bin::{age_bin}_{age_bin + 4}"] = 1

            demo_values = {
                "gender": clean_sex(gender),
                "race": clean_race(race),
                "ethnicity": clean_ethnicity(ethnicity),
            }

            for prefix, val in demo_values.items():

                if pd.notna(val):

                    val = str(val).strip()

                    if val != "":
                        feats[f"{prefix}::{val}"] = 1
                # QC mode: demographics only
            if self.demographics_only:
                return feats


        for _, row in patient_df.iterrows():
            event_date = row[self.event_date_col]
            days_before = (prediction_time.date() - event_date.date()).days

            bin_name = self._time_bin(days_before)
            if bin_name is None:
                continue

            measurements = row[self.measurements_col]
            if measurements is None:
                continue

            n_events = 0
            unique_codes = set()
            domain_counts = {}

            for m in measurements:
                code = m.get("code")
                domain = m.get("domain", "unknown")

                if code is None:
                    continue

                n_events += 1
                unique_codes.add(code)
                domain_counts[domain] = domain_counts.get(domain, 0) + 1

                m_feats = self._measurement_feats(m)

                for k, v in m_feats.items():
                    name = f"{bin_name}::{k}"
                    if self.use_binary_counts:
                        feats[name] = 1
                    else:
                        feats[name] = feats.get(name, 0) + v

            if self.add_utilization:
                feats[f"{bin_name}::util::n_events"] = feats.get(
                    f"{bin_name}::util::n_events", 0
                ) + n_events

                feats[f"{bin_name}::util::n_unique_codes"] = feats.get(
                    f"{bin_name}::util::n_unique_codes", 0
                ) + len(unique_codes)

                for domain, cnt in domain_counts.items():
                    feats[f"{bin_name}::util::n_{domain}_events"] = feats.get(
                        f"{bin_name}::util::n_{domain}_events", 0
                    ) + cnt

        return feats

    def _build_dicts(self, patient_day_df, meta_df, pid_order):
        patient_day_df = patient_day_df.copy()
        meta_df = meta_df.copy()

        patient_day_df[self.pid_col] = patient_day_df[self.pid_col].astype(int)
        meta_df[self.pid_col] = meta_df[self.pid_col].astype(int)

        # print("############ before offseting #########")
        # print(pd.to_datetime(meta_df[self.prediction_time_col], utc=True).describe())

        # offset prediction_time
        meta_df[self.prediction_time_col] = (
                pd.to_datetime(meta_df[self.prediction_time_col], utc=True)
                - pd.DateOffset(months=self.offset_month)
        )

        # print("############ after offseting #########")
        # print(meta_df[self.prediction_time_col].describe())

        # CREATE LOOKUP AFTER OFFSET
        meta_lookup = (
            meta_df[
                [
                    self.pid_col,
                    self.prediction_time_col,
                    self.birth_col,
                    self.race_col,
                    self.ethnicity_col,
                    self.gender_col,
                ]
            ]
            .drop_duplicates(self.pid_col)
            .set_index(self.pid_col)
        )

        grouped = {
            int(pid): g
            for pid, g in patient_day_df.groupby(self.pid_col)
        }

        dicts = []
        pids = []

        for pid in np.asarray(pid_order).astype(int):
            if pid not in meta_lookup.index:
                raise ValueError(f"Missing meta row for pid={pid}")

            g = grouped.get(pid)

            if g is None or len(g) == 0:
                dicts.append({})
            else:
                meta_row = meta_lookup.loc[pid]

                feats = self._patient_to_dict(
                    patient_df=g,
                    prediction_time=meta_row[self.prediction_time_col],
                    birth_datetime=meta_row[self.birth_col],
                    race=meta_row[self.race_col],
                    ethnicity=meta_row[self.ethnicity_col],
                    gender=meta_row[self.gender_col],
                )

                dicts.append(feats)

            pids.append(pid)

        return dicts, np.array(pids, dtype=int)

    def fit(self, patient_day_df, meta_df, pid_order):
        dicts, pids = self._build_dicts(patient_day_df, meta_df, pid_order)
        self.vectorizer.fit(dicts)
        self.feature_names_ = self.vectorizer.get_feature_names_out()
        return self

    def transform(self, patient_day_df, meta_df, pid_order):
        dicts, pids = self._build_dicts(patient_day_df, meta_df, pid_order)
        X = self.vectorizer.transform(dicts)
        return X, pids

    def fit_transform(self, patient_day_df, meta_df, pid_order):
        dicts, pids = self._build_dicts(patient_day_df, meta_df, pid_order)
        X = self.vectorizer.fit_transform(dicts)
        self.feature_names_ = self.vectorizer.get_feature_names_out()
        return X, pids

    def qc_before_featurization(
            self,
            patient_day_df,
            meta_df,
            pid_order,
            label_col=None,
            max_examples=10,
    ):
        """
        QC before fit_transform/transform.
        Checks PID overlap, date validity, prediction time, event timing,
        and possible leakage before featurization.
        """

        patient_day_df = patient_day_df.copy()
        meta_df = meta_df.copy()

        pid_order = np.asarray(pid_order).astype(int)

        patient_day_df[self.pid_col] = patient_day_df[self.pid_col].astype(int)
        meta_df[self.pid_col] = meta_df[self.pid_col].astype(int)

        patient_day_df[self.event_date_col] = pd.to_datetime(
            patient_day_df[self.event_date_col],
            errors="coerce",
            utc=True,
        )

        meta_df[self.prediction_time_col] = pd.to_datetime(
            meta_df[self.prediction_time_col],
            errors="coerce",
            utc=True,
        )

        print("\n===== BEFORE FEATURIZATION QC =====")

        print("Requested pids:", len(pid_order))
        print("Unique requested pids:", len(np.unique(pid_order)))

        meta_pids = set(meta_df[self.pid_col].unique())
        event_pids = set(patient_day_df[self.pid_col].unique())

        missing_meta = [pid for pid in pid_order if pid not in meta_pids]
        missing_events = [pid for pid in pid_order if pid not in event_pids]

        print("Missing in meta_df:", len(missing_meta))
        print("Missing in patient_day_df:", len(missing_events))

        if missing_meta:
            print("Example missing meta pids:", missing_meta[:max_examples])

        if missing_events:
            print("Example no-event pids:", missing_events[:max_examples])

        dup_meta = meta_df[self.pid_col].duplicated().sum()
        print("Duplicate person_id rows in meta_df:", dup_meta)

        print("\nDate ranges:")
        print("event_date min:", patient_day_df[self.event_date_col].min())
        print("event_date max:", patient_day_df[self.event_date_col].max())
        print("prediction_time min:", meta_df[self.prediction_time_col].min())
        print("prediction_time max:", meta_df[self.prediction_time_col].max())

        # Check future events relative to prediction time
        pred_lookup = (
            meta_df[[self.pid_col, self.prediction_time_col]]
            .drop_duplicates(self.pid_col)
            .set_index(self.pid_col)[self.prediction_time_col]
        )

        tmp = patient_day_df[
            patient_day_df[self.pid_col].isin(pid_order)
        ].copy()

        tmp["prediction_time"] = tmp[self.pid_col].map(pred_lookup)

        tmp["after_prediction"] = (
                tmp[self.event_date_col].notna()
                & tmp["prediction_time"].notna()
                & (tmp[self.event_date_col] > tmp["prediction_time"])
        )

        print("\nRows after prediction_time:")
        print(tmp["after_prediction"].sum())

        print("Patients with any event after prediction_time:")
        print(
            tmp.loc[tmp["after_prediction"], self.pid_col]
            .nunique()
        )

        # Events before prediction
        tmp["before_or_on_prediction"] = (
                tmp[self.event_date_col].notna()
                & tmp["prediction_time"].notna()
                & (tmp[self.event_date_col] <= tmp["prediction_time"])
        )

        pre_counts = (
            tmp[tmp["before_or_on_prediction"]]
            .groupby(self.pid_col)
            .size()
        )

        print("\nPre-prediction patient-day row counts:")
        print(pre_counts.describe())

        zero_pre_pids = [
            pid for pid in pid_order
            if pre_counts.get(pid, 0) == 0
        ]

        print("Patients with zero pre-prediction patient-day rows:", len(zero_pre_pids))
        print("Examples:", zero_pre_pids[:max_examples])

        # Optional label distribution for no-event patients
        if label_col is not None and label_col in meta_df.columns:
            label_lookup = (
                meta_df[[self.pid_col, label_col]]
                .drop_duplicates(self.pid_col)
                .set_index(self.pid_col)[label_col]
            )

            zero_labels = pd.Series(
                [label_lookup.get(pid, np.nan) for pid in zero_pre_pids]
            )

            print("\nLabels among zero-pre-event patients:")
            print(zero_labels.value_counts(dropna=False))

        return {
            "missing_meta_pids": missing_meta,
            "missing_event_pids": missing_events,
            "zero_pre_event_pids": zero_pre_pids,
        }

    def qc_after_featurization(
            self,
            X,
            pids,
            feature_names=None,
            y=None,
            top_n=50,
    ):
        """
        QC after sparse feature generation.
        Checks sparsity, zero-feature rows, rare/common features,
        top features, and obvious leakage keywords.
        """

        if feature_names is None:
            feature_names = self.feature_names_

        feature_names = np.asarray(feature_names)

        print("\n===== AFTER FEATURIZATION QC =====")

        print("X shape:", X.shape)
        print("n pids:", len(pids))
        print("n feature_names:", len(feature_names))

        assert X.shape[0] == len(pids)
        assert X.shape[1] == len(feature_names)

        nnz = X.nnz
        density = nnz / (X.shape[0] * X.shape[1])

        print("NNZ:", nnz)
        print("Density:", density)
        print("Sparsity:", 1 - density)

        row_nnz = np.diff(X.indptr)

        print("\nFeatures per patient:")
        print(pd.Series(row_nnz).describe())

        zero_mask = row_nnz == 0
        zero_pids = np.asarray(pids)[zero_mask]

        print("\nPatients with zero generated features:", len(zero_pids))
        print("Percent zero-feature patients:", 100 * zero_mask.mean())
        print("Example zero-feature pids:", zero_pids[:20])

        if y is not None:
            y = np.asarray(y)
            print("\nLabel distribution overall:")
            print(pd.Series(y).value_counts(dropna=False))

            print("\nLabel distribution among zero-feature patients:")
            print(pd.Series(y[zero_mask]).value_counts(dropna=False))

        # Feature prevalence
        feat_nnz = (X > 0).sum(axis=0).A1
        feat_prev = feat_nnz / X.shape[0]

        feat_qc = pd.DataFrame({
            "feature": feature_names,
            "n_patients": feat_nnz,
            "prevalence": feat_prev,
        })

        print("\nFeature prevalence summary:")
        print(feat_qc["prevalence"].describe())

        print("\nTop common features:")
        print(
            feat_qc.sort_values("prevalence", ascending=False)
            .head(top_n)
        )

        print("\nRare feature counts:")
        print("n_features present in <2 patients:", int((feat_nnz < 2).sum()))
        print("n_features present in <5 patients:", int((feat_nnz < 5).sum()))
        print("n_features present in <10 patients:", int((feat_nnz < 10).sum()))
        print("n_features present in <1% patients:", int((feat_prev < 0.01).sum()))

        # Leakage keyword scan
        leakage_keywords = [
            "dementia",
            "alzheimer",
            "alzheim",
            "mci",
            "mild cognitive",
            "cognitive impairment",
            "memory loss",
            "donepezil",
            "memantine",
            "rivastigmine",
            "galantamine",
            "aducanumab",
            "lecanemab",
        ]

        leak_mask = np.array([
            any(k in f.lower() for k in leakage_keywords)
            for f in feature_names
        ])

        leak_features = feat_qc.loc[leak_mask].sort_values(
            "prevalence",
            ascending=False,
        )

        print("\nPotential leakage features by keyword:")
        print(leak_features.head(100))

        # Domain-wise counts
        domain_prefixes = [
            "condition",
            "drug",
            "procedure",
            "measurement",
            "observation",
            "visit",
            "device",
            "util",
            "age_bin",
            "sex",
            "race",
            "ethnicity",
        ]

        print("\nFeature counts by rough domain:")
        for d in domain_prefixes:
            n = sum(d in f for f in feature_names)
            print(f"{d}: {n}")

        return {
            "zero_feature_pids": zero_pids,
            "feature_qc": feat_qc,
            "leak_features": leak_features,
        }
import re

def get_max_length(model_name):
    if "clmbr-t-base" in model_name:
        return 496
    elif "count_featurizer" in model_name or "baseline_utils" in model_name:
        return 0
    match = re.search(r"-(\d+)-clmbr$", model_name)
    if not match:
        raise ValueError(f"Could not extract max length from {model_name}")
    return int(match.group(1))

# print(get_max_length("StanfordShahLab/llama-base-2048-clmbr"))      # 2048
# print(get_max_length("StanfordShahLab/mamba-tiny-16384-clmbr"))     # 16384
# print(get_max_length("StanfordShahLab/gpt-base-512-clmbr"))         # 512
def make_model_config():
    import itertools

    param_list = []

    model_names = [
        "count_featurizer",
        # "baseline_utils",
    ]

    # model_names = ["StanfordShahLab/clmbr-t-base","StanfordShahLab/mamba-tiny-16384-clmbr",
    #     "StanfordShahLab/llama-base-2048-clmbr","StanfordShahLab/llama-base-512-clmbr",
    #     "StanfordShahLab/gpt-base-2048-clmbr","StanfordShahLab/gpt-base-512-clmbr"
    #
    # ]
    # model_names=["count_featurizer","baseline_utils"]
    # model_names = [
    # # "StanfordShahLab/clmbr-t-base",
    # #                "StanfordShahLab/mamba-tiny-16384-clmbr",
    #     "StanfordShahLab/llama-base-2048-clmbr",
    #     # "StanfordShahLab/llama-base-512-clmbr",
    #     #            "StanfordShahLab/gpt-base-512-clmbr",
    #     "StanfordShahLab/gpt-base-2048-clmbr"]

    # classifiers = [
    #     "lr",
    #     "xgboost",
    #     "lightgbm",
    # ]
    classifiers = [
        # "xgboost",
        "lightgbm"
        # "lr"

    ]


    offset_months = [ 12,24,36]

    cohort_defs = [
        "icd_or_drug",
        # "cd4",

        # "icd_confirmed_first",
        # "drug_only",

    ]

    for model_name, offset_month, cd, classifier in itertools.product(
        model_names,
        offset_months,
        cohort_defs,
        classifiers,
    ):

        if classifier == "lr":

            for lr_penalty in ["l2"]:

                param_list.append({
                    "model_name": model_name,
                    "offset_month": offset_month,
                    "cd": cd,
                    "classifier": classifier,
                    "lr_penalty": lr_penalty,
                    "max_length":get_max_length(model_name)

                })

        else:

            param_list.append({
                "model_name": model_name,
                "offset_month": offset_month,
                "cd": cd,
                "classifier": classifier,
                "lr_penalty": "",
                "max_length": get_max_length(model_name)
            })

    print(f"Total configs: {len(param_list)}")
    print(param_list[:5])
    return param_list
def make_model(args,params,y_train=None):
    if args.classifier=='lr' and args.lr_penalty=='elasticnet':
        model = Pipeline([
            ("scaler",  MaxAbsScaler()),

            ("clf", LogisticRegression(
                penalty="elasticnet",
                solver="saga",

                C=params.get("C", 1.0),
                l1_ratio=params.get("l1_ratio", None),
                class_weight=params.get("class_weight", "balanced"),
                max_iter=params.get("max_iter", 5000),
                random_state=42,
            ))
        ])
    elif args.classifier=='lr' and args.lr_penalty=='l2':
        # with_mean = False
        if "count_featurizer" in args.model_dir or "baseline_utils" in args.model_dir:
            scaler = MaxAbsScaler()
        else:
            scaler = StandardScaler()
        model=Pipeline([
        ("scaler", scaler),
        ("clf", LogisticRegression(
            penalty=params.get("penalty", "l2"),
            C=params.get("C", 1.0),
            solver=params.get("solver", "liblinear"),
            class_weight=params.get("class_weight", "balanced"),
            max_iter=params.get("max_iter", 5000),
            random_state=42,
        ))
    ])
    elif args.classifier=='xgboost':
        scale_pos_weight = None

        if y_train is not None:
            n_pos = np.sum(y_train == 1)
            n_neg = np.sum(y_train == 0)
            scale_pos_weight = n_neg / max(1, n_pos)

        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            tree_method="hist",
            random_state=42,
            n_jobs=-1,
            scale_pos_weight=scale_pos_weight,
            **params
        )
    elif args.classifier=='lightgbm':
        scale_pos_weight = None

        if y_train is not None:
            n_pos = np.sum(y_train == 1)
            n_neg = np.sum(y_train == 0)
            scale_pos_weight = n_neg / max(1, n_pos)

        model= lgb.LGBMClassifier(
            objective="binary",
            metric="auc",
            random_state=42,
            n_jobs=-1,
            scale_pos_weight=scale_pos_weight,
            verbose=-1,
            **params
        )
    return model
def define_model_name(args):
    if args.baseline_utils:
        # if 'all' not in args.exclude:
        #
        #     model_name = 'baseline_utils'+"+all_lifestyle-" + args.exclude
        if  len(args.include)>0:
            if 'all' in args.include:
                model_name = 'baseline_utils_+' + "+all_lifestyle-"
            else:

                model_name = 'baseline_utils_+'+ args.include
        else:
            model_name = 'baseline_utils'
    elif  len(args.include)>0 and 'value' not in args.embedding_type: #clmbr-t-base wo value embed
        if 'all' in args.include:
            model_name = args.model_dir + "+all_lifestyle-"
        else:
            model_name = args.model_dir + "_"+args.include
    elif 'w_value_dedup' in args.embedding_type: #clmbr-t-base w value embed
        if  len(args.include)>0 and 'all' in args.include:
            model_name = args.model_dir + '-w-value_dedup' + "+all_lifestyle-"
        elif len(args.include)>0 :
            model_name = args.model_dir + '-w-value_dedup' + "_"+args.include
        else:
            model_name = args.model_dir + '-w-value_dedup'
    elif 'value' in args.embedding_type: #clmbr-t-base w value embed
        if  len(args.include)>0 and 'all' in args.include:
            model_name = args.model_dir + '-w-value' + "+all_lifestyle-"
        elif len(args.include)>0 :
            model_name = args.model_dir + '-w-value' + "_"+args.include
        else:
            model_name = args.model_dir + '-w-value'

    else :

        model_name = args.model_dir



    # elif 'all' not in args.exclude:
    #     model_name = args.model_dir + "+all_lifestyle-"+args.exclude
    # elif 'all' in args.exclude:
    #     model_name = args.model_dir

    return model_name

##########################count_featurizer###################
# =========================
# CHUNK 21: GCS CACHE HELPERS
# =========================
def clean_date_col(df):
    date_cols = [
        c for c in df.columns
        if "date" in c.lower() or "time" in c.lower()
    ]

    for c in date_cols:
        df[c] = pd.to_datetime(
            df[c],
            errors="coerce",
            utc=True,
        )
    return df

def qc_embed(name):
    print("#####################QC for file :####################", name)

    data = np.load(name)

    X = data["embeddings"]
    y = data["labels"]
    pids = data["pids"]

    print("===== BASIC =====")
    print("shape:", X.shape)
    print("dtype:", X.dtype)

    print("\n===== NAN / INF =====")
    print("nan count:", np.isnan(X).sum())
    print("inf count:", np.isinf(X).sum())

    print("\n===== GLOBAL STATS =====")
    print("mean:", X.mean())
    print("std:", X.std())
    print("min:", X.min())
    print("max:", X.max())

    print("\n===== VECTOR NORM =====")

    norms = np.linalg.norm(X, axis=1)

    print("norm mean:", norms.mean())
    print("norm std:", norms.std())
    print("norm min:", norms.min())
    print("norm max:", norms.max())

    print("\n===== ZERO VECTORS =====")

    zero_mask = norms == 0

    print("n zero vectors:", zero_mask.sum())

    if zero_mask.sum() > 0:
        print("zero vector pids:", pids[zero_mask][:20])

    print("\n===== DUPLICATE VECTORS =====")

    rounded = np.round(X, 5)

    dup_count = (
        pd.DataFrame(rounded)
        .duplicated()
        .sum()
    )

    print("duplicate vectors:", dup_count)

    print("\n===== CLASS DISTRIBUTION =====")
    print(pd.Series(y).value_counts())

    print("\n===== CLASS-WISE EMBEDDING STATS =====")

    for cls in np.unique(y):
        cls_X = X[y == cls]

        print(f"\nclass={cls}")
        print("n:", len(cls_X))
        print("mean:", cls_X.mean())
        print("std:", cls_X.std())

        cls_norms = np.linalg.norm(cls_X, axis=1)

        print("norm mean:", cls_norms.mean())
        print("norm std:", cls_norms.std())
def save_fig_local_gcs(fig, basename, local_dir, dpi=300):
    WORKSPACE_BUCKET = "gs://rw-migration-aou-rw-0ab327cd"
    GCS_OUT_DIR = f"{WORKSPACE_BUCKET}/ADRD_onset/checkpoints/"

    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    local_png = local_dir / f"{basename}"

    fig.savefig(
        local_png,
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
    )

    subprocess.run(
        ["gsutil", "-m", "cp", str(local_png), GCS_OUT_DIR + "/"],
        check=True,
    )

    print(f"Saved {local_png}")
def save_df_local_gcs(df, basename, local_dir):
    WORKSPACE_BUCKET = "gs://rw-migration-aou-rw-0ab327cd"
    # WORKSPACE_CDR = "wb-silky-artichoke-2408.C2024Q3R9"
    # WORKSPACE_BUCKET = os.environ["WORKSPACE_BUCKET"]
    GCS_OUT_DIR = f"{WORKSPACE_BUCKET}/ADRD_onset/checkpoints/"


    if isinstance(df, pd.DataFrame):
        df=clean_date_col(df)


    local_dir = Path(local_dir)  # 🔥 fix
    local_dir.mkdir(parents=True, exist_ok=True)
    if ".npz" in basename:
        local_parquet = local_dir / basename
        if 'embeddings' in basename:
            pids = np.array([x["pids"] for x in df])
            labels = np.array([x["labels"] for x in df])
            all_embs = np.stack([x["embeddings"] for x in df])
            np.savez_compressed(
                local_parquet,
                embeddings=all_embs,
                pids=pids,
                labels=labels)
        elif ".csv" in basename:
            local_parquet = local_dir / basename
            df.to_csv(local_parquet, index=False)

        else:

            raise ValueError("For .npz saving, basename should contain 'embedding'.")
    elif ".csv" in basename:
        local_parquet = local_dir / f"{basename}"

        df.to_csv(local_parquet, index=False)
    else:
        local_parquet = local_dir / f"{basename}.parquet"

        df.to_parquet(local_parquet, index=False)

    subprocess.run(["gsutil", "-m", "cp", str(local_parquet), GCS_OUT_DIR + "/"], check=True)
    print('saved')
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def log_row(path, row, header=None):
    # create parent directory if needed
    try:
        dir_path = os.path.dirname(path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        file_exists = os.path.exists(path)

        with open(path, "a", newline="") as f:
            writer = csv.writer(f)

            # write header only once
            if header is not None and (not file_exists or os.stat(path).st_size == 0):
                writer.writerow(header)

            writer.writerow(row)
            print(f"row saved in {path}")
    except Exception as e:
        print(e)
import os
import csv
import pandas as pd
import numpy as np

def log_row_dict(path, row_dict):

    os.makedirs(os.path.dirname(path), exist_ok=True)

    file_exists = os.path.exists(path) and os.path.getsize(path) > 0

    # -------------------------
    # New file
    # -------------------------
    if not file_exists:

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=list(row_dict.keys())
            )

            writer.writeheader()
            writer.writerow(row_dict)

        return

    # -------------------------
    # Existing file
    # -------------------------
    with open(path, "r", newline="") as f:
        existing_header = next(csv.reader(f))

    # new columns
    new_cols = [
        c for c in row_dict.keys()
        if c not in existing_header
    ]

    # -------------------------
    # Expand columns safely
    # -------------------------
    if new_cols:

        df = pd.read_csv(path)

        for col in new_cols:
            df[col] = np.nan

        expanded_header = existing_header + new_cols

        df = df.reindex(columns=expanded_header)

        df.to_csv(path, index=False)

        existing_header = expanded_header

    # -------------------------
    # Append aligned row
    # -------------------------
    with open(path, "a", newline="") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=existing_header,
        )

        writer.writerow({
            col: row_dict.get(col, np.nan)
            for col in existing_header
        })

    print(f"row saved in {path}")
def keep_tokenizer_supported_events(events,pid, tokenizer):
    kept = []


    for e in events:
        code = getattr(e, "code", None)

        if code in BIRTH_CODES:
            kept.append(e)
        elif code in tokenizer.code_lookup:
            kept.append(e)
            # collect unknown codes

    return kept

def meds_event_to_femr_event(e):
    """
    Convert one meds_reader.Event object to FEMR/MEDS dict format.
    """
    code=e.code
    # if code in {"MEDS_BIRTH", "SNOMED/184099003", "SNOMED/3950001"}:
    #     code = meds.birth_code

    measurement = {
        "code": code,
    }

    # Add optional fields only if present
    if hasattr(e, "numeric_value") and e.numeric_value is not None:
        measurement["numeric_value"] = e.numeric_value

    if hasattr(e, "text_value") and e.text_value is not None:
        measurement["text_value"] = e.text_value

    return {
        "time": e.start,
        "measurements": [measurement],
    }


def meds_subject_events_to_femr_patient(events, pid):
    """
    Convert list[meds_reader.Event] to FEMR patient dict.
    """

    """
        Convert wrapped Event list to FEMR patient dict.
        """

    femr_events = [
        meds_event_to_femr_event(e)
        for e in events
        if getattr(e, "code", None) is not None
           and getattr(e, "start", None) is not None
    ]

    return {
        "patient_id": int(pid),
        "events": femr_events,
    }
#keep tokenizable event, then truncate
def truncate_tokenizable_events_keep_birth(events,pid, tokenizer, max_length):
    events = keep_tokenizer_supported_events(events,pid, tokenizer)

    birth_events = [e for e in events if e.code in BIRTH_CODES]
    non_birth_events = [e for e in events if e.code not in BIRTH_CODES]

    if max_length is None:
        return birth_events + non_birth_events

    keep_n = max_length - len(birth_events)
    keep_n = max(0, keep_n)

    return birth_events + non_birth_events[-keep_n:]
#collate batch for femr tokenizer for clmbr-t-base model
def collate_batch_femr(samples, batch_processor, max_length,tokenizer):
    assert len(samples) == 1

    s = samples[0]

    raw_n_events = len(s["text"])

    patient_events = truncate_tokenizable_events_keep_birth(
        s["text"],s["pid"],
        tokenizer,
        max_length=max_length,
    )

    truncated_n_events = len(patient_events)

    patient = meds_subject_events_to_femr_patient(
        patient_events,
        s["pid"],
    )

    femr_n_events = len(patient["events"])

    raw_batch = batch_processor.convert_patient(
        patient,
        tensor_type="pt",
    )

    batch = batch_processor.collate([raw_batch])

    batch["labels"] = torch.tensor([s["label"]], dtype=torch.float)
    batch["pid"] = [s["pid"]]

    # debug metadata
    batch["raw_n_events"] = [raw_n_events]
    batch["truncated_n_events"] = [truncated_n_events]
    batch["femr_n_events"] = [femr_n_events]
    # debug info
    batch["patient_events"] = patient_events
    batch["patient_dict"] = patient

    return batch
class SNPParquetDataset(Dataset):
    def __init__(self, parquet_path, label_col=None):
        df = pd.read_parquet(parquet_path)
        df=df[['geno']]
        label=[1,0,1,1,0,0,1,0,1,0]
        # self.y = torch.tensor(df[label_col].values, dtype=torch.float32)
        self.y = torch.tensor(label, dtype=torch.float32)
        # self.X = torch.tensor(
        #     df.drop(columns=[label_col]).values,
        #     dtype=torch.float32
        # )
        X_np = np.stack(df['geno'].values)  # shape: (num_samples, 1000)

        # Convert everything to float32 (or int if you know it's integer)
        X_np = X_np.astype(np.float32)
        self.X = torch.tensor(
            X_np
        )

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
def log_message(log_file, msg):
    """Log message to file and print to console."""
    timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    full_msg = f"{timestamp} {msg}"
    with open(log_file, "a") as f:
        f.write(full_msg + "\n")
    print(full_msg)


# import meds_reader
# class MultiPathSubjectDatabase:
#     def __init__(self, db_paths,allowed_ids):
#         self.dbs = [meds_reader.SubjectDatabase(p) for p in db_paths]
#         self.allowed_ids = allowed_ids
#
#     def map_with_data(self, fn, *args, **kwargs):
#         def wrapped(rows):
#             for subject, data in rows:
#                 if subject.subject_id in self.allowed_ids:
#                     yield subject, data
#
#         for db in self.dbs:
#             yield from db.map_with_data(wrapped, *args, **kwargs)
# class FilteredSubjectDatabase:
#     def __init__(self, unified_db, allowed_subject_ids):
#         self.db = unified_db
#         self.allowed_subject_ids=allowed_subject_ids
#         self.allowed_indices = [
#             i for i, sid in enumerate(unified_db.subject_ids())
#             if sid in allowed_subject_ids
#         ]
#
#     def __len__(self):
#         return len(self.allowed_indices)
#
#     def __getitem__(self, idx):
#         return self.db[self.allowed_indices[idx]]
#
#     def map_with_data(self, fn, *args, **kwargs):
#         def wrapped(rows):
#             for subject, data in rows:
#                 if subject.subject_id in self.allowed_subject_ids:
#                     yield subject, data
#         return self.db.map_with_data(wrapped, *args, **kwargs)


# class BaseDataset(Dataset):
#     pass
# #copied from utils_ddp as per current split file
# class TokenizedDatasetByPID(torch.utils.data.Dataset):
#     def __init__(self,data_dir, token_file,split='train', split_index=None,seed=42):
#         data = torch.load(token_file, map_location="cpu")
#
#         pid_to_idx = {pid.item(): i for i, pid in enumerate(data["pids"])}
#
#         self.split=split
#         self.split_index=split_index
#         self.seed=seed
#         pid_list = []
#
#         if self.split_index is not None:
#
#
#
#
#             fullpath = data_dir +'cv_3/fold_' + str(self.split_index) + "/full.parquet"
#             splits = pd.read_parquet(fullpath, engine='pyarrow')
#         else:
#             fullpath = data_dir + 'train_test_70_30.parquet'
#             splits = pd.read_parquet(fullpath, engine='pyarrow')
#
#             # print('loading data from ..')
#             # print(fullpath)
#
#         if self.split == "train":
#
#             pid_list.extend(splits[splits["split"].isin(["train"])]["patient_id"].to_numpy())
#             # print(self.split)
#             # print(len(pids))
#
#         elif self.split == "val":
#             pid_list.extend(splits[splits["split"].isin(["test"])]["patient_id"].to_numpy())
#             # print(self.split)
#             # print(len(pids))
#     # print('loaded %d patients in %s split'%(len(pid_list),self.split))
#         indices=[]
#         for pid in pid_list:
#             try:
#                 indices.append(pid_to_idx[pid])
#             except:
#                 print('no pid')
#                 print(pid)
#         self.paths = data_dir
#         try:
#             if len(indices)>0:
#                     self.input_ids =data["input_ids"][indices]
#                     self.attention_mask =data["attention_mask"][indices]
#                     self.labels = [data["labels"][i].item() for i in indices]
#                     self.pids = [data["pids"][i].item() for i in indices]
#             else:
#                 print('indices empty')
#                 exit()
#         except Exception as e:
#             print(e)
#             # print(self.labels)
#         # print(len(self.labels))
#         # print(self.pids)
#         # print(len(self.pids))
#
#     def get_n_patients(self) -> int:
#         return len(self.pids)
#
#     def get_pids(self) -> np.ndarray:
#         return np.array(self.pids)
#
#     def get_labels(self) -> np.ndarray:
#         return np.array(self.labels)
#     def __len__(self):
#         return len(self.labels)
#
#     def __getitem__(self, idx):
#         # ---------------- SAFE GUARDS ----------------
#         if self.input_ids[idx] is None or len(self.input_ids[idx]) == 0:
#             raise ValueError(f"Empty input_ids at index {idx}")
#
#         if sum(self.attention_mask[idx]) == 0:
#             raise ValueError(f"All padding sample at index {idx}")
#         return {
#             "input_ids": self.input_ids[idx],
#             "attention_mask": self.attention_mask[idx],
#             "labels": self.labels[idx],
#             "pid": self.pids[idx],
#         }
#
#
#
# class MEDSDataset(BaseDataset):
#     """
#     Dataset that returns patients in one or more MEDS datasets (after conversion to MEDSReader extracts).
#     dataset[idx] = a specific patient.
#     """
#
#     def __init__(self,args,
#                  paths_to_meds_reader_extract: Union[str, List[str]],
#                  split: str = 'train',
#                  split_index: str=None,
#
#
#                  is_debug: bool = False,
#                  seed: int = None):
#
#
#         if isinstance(paths_to_meds_reader_extract, str):
#             paths_to_meds_reader_extract = [paths_to_meds_reader_extract]
#
#         for path in paths_to_meds_reader_extract:
#             assert os.path.exists(path), f"{path} is not a valid path"
#
#         assert split in ['train', 'val', 'test','all'], f"{split} not in ['train','val','test', 'all']"
#
#         self.paths = paths_to_meds_reader_extract
#         self.dbs = [meds_reader.SubjectDatabase(path, num_threads=1) for path in self.paths]
#         self.split = split
#         self.split_index = split_index
#         self.is_debug = is_debug
#         self.seed = seed
#         # print('seed %d' % (self.seed))
#
#         # Store metadata
#         self.metadata = {
#             'cls': 'MEDSDataset',
#             'paths_to_meds_reader_extract': self.paths,
#             'split': split,
#             'is_debug': is_debug,
#             'seed': seed,
#         }
#
#         # Collect splits from all databases
#         self.pid_map = []  # list of (db_index, pid)
#         self.label_map = []
#         self.label = None
#         for db_idx, path in enumerate(self.paths):
#
#             if 'control' in path:
#                 self.label = 0
#             else:
#                 self.label = 1
#
#             # pids=None
#
#             if self.split_index is None:
#                 fullpath =args.train_test_path +args.cohort_definition+ "/train_test_70_30.parquet"
#                 splits = pd.read_parquet(fullpath, engine='pyarrow')
#                 pids = splits[
#
#                     (splits["label"] == self.label)
#                 ]["patient_id"].to_numpy()
#
#                 if split == "train":
#                     pids = splits[
#                         (splits["split"] == "train") &
#                         (splits["label"] == self.label)
#                         ]["patient_id"].to_numpy()
#
#                 elif split == "val" or split == "test":
#                     pids = splits[
#                         (splits["split"] == "test") &
#                         (splits["label"] == self.label)
#                         ]["patient_id"].to_numpy()
#                 elif split == "all":
#                     pids = splits[
#
#                         (splits["label"] == self.label)
#                         ]["patient_id"].to_numpy()
#
#             else:
#
#                 fullpath = args.train_test_path + args.cohort_definition + '/cv_3/fold_' + str(
#                     split_index) + "/full.parquet"
#
#                 splits = pd.read_parquet(fullpath, engine='pyarrow')
#                 if self.split == "train":
#                     print('train')
#
#                     pids = splits[
#                         (splits["split"] == "train") &
#                         (splits["label"] == self.label)
#                         ]["patient_id"].to_numpy()
#
#                 elif self.split == "val":
#                     # print('val')
#                     pids = splits[
#                         (splits["split"] == "test") &
#                         (splits["label"] == self.label)
#                         ]["patient_id"].to_numpy()
#                 elif split == "all":
#                     pids = splits[
#                         (splits["split"] == "test") | (splits["split"] == "train") &
#                         (splits["label"] == self.label)
#                         ]["patient_id"].to_numpy()
#
#             for pid in pids:
#                     self.pid_map.append((db_idx, pid))
#                     self.label_map.append((pid, self.label))
#
#         # Debug mode → shrink dataset
#         if is_debug:
#             self.pid_map = self.pid_map[:1000]
#
#     def get_n_patients(self) -> int:
#         return len(self.pid_map)
#
#     def get_pids(self) -> np.ndarray:
#         return np.array([pid for _, pid in self.pid_map])
#
#     def get_labels(self) -> np.ndarray:
#         return np.array([label for _, label in self.label_map])
#
#     def __len__(self) -> int:
#         return len(self.pid_map)
#
#
#     def __getitem__(self, idx: int) -> Tuple[int, List["Event"]]:
#         # Case 1: slice (e.g. dataset[0:5] or dataset[:])
#
#
#         if isinstance(idx, slice):
#             indices = range(*idx.indices(len(self)))
#             return [self[i] for i in indices]
#
#         # Case 2: integer index
#         db_idx, pid = self.pid_map[idx]
#         pids,label=self.label_map[idx]
#         db = self.dbs[db_idx]
#         # print("############logging############")
#         # print(f"actual pid: {pid}")
#         # print(f"actual label : {label}, db_index : {db_idx}")
#         # db_ids = set(db._all_subject_ids)
#         # print(f"db ids from subjectdb")
#         #
#         # print(db_ids)
#         # print("pid value:", pid, "type:", type(pid))
#         # print("db sample type:", type(next(iter(db._all_subject_ids))))
#         #
#         # if pid in db_ids:
#         #     print("Exists")
#         # else:
#         #     print("Missing:", pid)
#
#
#         try:
#             events = [
#                 Event(
#                     code=e.code,
#                     value=getattr(e, "numeric_value", None) or getattr(e, "text_value", None),
#                     unit=getattr(e, "unit", None),
#                     start=e.time,
#                     end=getattr(e, "end", None),
#                     omop_table=getattr(e, "omop_table", None),
#                 )
#                 for e in db[pid].events
#             ]
#         except Exception as e:
#             print(e)
#
#             print(f"actual pid: {pid}")
#             print(f"actual label : {label}, db_index : {db_idx}")
#         return (pid, label,events)
#
# ###wrapper class for meds format patient rep
# class MEDSDATASET(Dataset):
#     def __init__(self, patients,labels_df,label_col, tokenizer,max_length,demo):
#         self.patients = patients         # list of MEDS patient sequences
#         # self.labels = labels             # list or tensor of labels
#         self.tokenizer = tokenizer
#         self.max_length = max_length
#         self.labels_df=pd.read_parquet(labels_df,engine="pyarrow")
#         self.offset_year = label_col
#         self.patient_list=[] #list of list; list of even list
#         self.labels=[]
#         self.seq_len=[]
#         self.pids = []
#         self.demo=demo
#         self.subjects_2_event=set()
#         self.subjects_1_event = set()
#
#         # print('label_df size')
#         # print(self.labels_df.columns)
#         # print(len(self.labels_df))
#         for idx in range(len(self.patients)):
#             pid, label, events = self.patients[idx]   # <-- gives (patient_id, [Event])
#                       # keep raw list[Event]
#             self.labels.append(label)              # <-- placeholder; you must define task labels (e.g., mortality)
#             if label == 1:
#                 row = self.labels_df[self.labels_df["case_id"] == pid]
#                 prediction_time = (
#                         np.datetime64(row["cohort_index_date"].iloc[0])
#                         - np.timedelta64(int(365.25 * self.offset_year), 'D')
#                 )
#
#
#             else:
#
#                 row = self.labels_df[self.labels_df["control_id"] == pid]
#                 prediction_time = (
#                         np.datetime64(row["control_pseudo_index_date"].iloc[0])
#                         - np.timedelta64(int(365.25 * self.offset_year), 'D')
#                 )
#
#             # Get patient timeline and truncate
#             # print(prediction_time)
#             if self.demo:  # add demographic variable code from the event code if max_length allow
#                 # print('adding demo')
#
#                 event_list = [
#                     e for e in events
#                         if( e.code in BIRTH_CODES or ( e.start is not None and e.start <= prediction_time))
#                 ]
#             else:
#                 code_demo = ['LOINC/LA3-6', 'LOINC/LA2-8', 'LOINC/LA18959-9', 'Ethnicity', 'SNOMED/184099003',
#                              'SNOMED/419620001',
#                              'SNOMED/419620001',
#                              'SNOMED/90049009', 'SNOMED/276507005', 'Race', 'AoU_Custom/AoUDRC_NoneIndicated',
#                              'PPI/PMI_Skip', 'AoU_General/GenderIdentity_GeneralizedDiffGender', 'LOINC/LA29631-1',
#                              'PPI/PMI_PreferNotToAnswer', 'LOINC/LA14327-3',
#                              'AoU_General/WhatRaceEthnicity_GeneralizedMultPopulations', 'Ethnicity/Not Hispanic',
#                              'Ethnicity/Hispanic', 'MEDS_BIRTH', 'PCORNet/Generic-NI']
#                 event_list = [
#                     e for e in events
#                     if e.start is not None and e.start <= prediction_time and e.code not in code_demo
#                 ]
#
#             self.patient_list.append(event_list)
#             self.seq_len.append(len(event_list))
#             self.pids.append(pid)
#             if len(event_list)==2:
#                 print("###########debug 2 event list##############")
#                 print(f"demo {self.demo}")
#                 print(event_list)
#                 print(pid)
#                 self.subjects_2_event.add((pid,label))
#             if len(event_list)<2:
#                 print("###########debug 1 event list##############")
#                 print(f"demo {self.demo}")
#                 print(events)
#                 print(event_list)
#                 print(prediction_time)
#                 print(pid)
#                 self.subjects_1_event.add((pid,label))
#         print("##################################################")
#         print(f"# subjects with 2 event {len(self.subjects_2_event)}")
#         print(self.subjects_2_event)
#         print(f"# subjects with 1 event {len(self.subjects_1_event)}")
#         print(self.subjects_1_event)
#     def __len__(self):
#         return len(self.patients)
#     def get_pids(self):
#         return self.pids
#     def seq_len_stat(self):
#
#         print('seq_len stat')
#         print(len(self.get_pids()))
#         series = pd.Series(self.seq_len)
#         print(series.describe())
#         # """
#         #     lengths: list or 1D-array of token lengths for each sequence in your dataset.
#         #     seq_lens: iterable of sequence lengths to evaluate.
#         #     Returns a dict with stats for each seq_len.
#         #     """
#         # lengths = np.asarray(lengths, dtype=int)
#         # out = {}
#         # N = len(lengths)
#         # for L in seq_lens:
#         #     truncated_mask = lengths > L
#         #     truncated_count = int(truncated_mask.sum())
#         #     truncated_fraction = truncated_count / N
#         #     # How many tokens are lost (sum of (len - L) only where len > L)
#         #     tokens_lost = int((lengths[truncated_mask] - L).sum()) if truncated_count > 0 else 0
#         #     out[L] = {
#         #         "count": N,
#         #         "truncated_count": truncated_count,
#         #         "truncated_fraction": truncated_fraction,
#         #         "tokens_lost": tokens_lost,
#         #         "median_len": int(np.median(lengths)),
#         #         "75pct_len": int(np.percentile(lengths, 75)),
#         #         "90pct_len": int(np.percentile(lengths, 90)),
#         #         "max_len": int(lengths.max())
#         #     }
#         #     if verbose:
#         #         print(f"--- seq_len = {L} ---")
#         #         print(f"dataset size: {N}")
#         #         print(f"truncated_count: {truncated_count} ({truncated_fraction:.2%})")
#         #         print(f"tokens_lost_total: {tokens_lost}")
#         #         print(
#         #             f"median: {out[L]['median_len']}, 75p: {out[L]['75pct_len']}, 90p: {out[L]['90pct_len']}, max: {out[L]['max_len']}")
#         #         print()
#         # return out
#         # print(series.describe())
#
#     def __getitem__(self, idx):
#         patient = self.patient_list[idx]     # list[Event] (MEDS)
#         label = self.labels[idx]
#         pid = self.pids[idx]
#
#         # Tokenize MEDS patient into GPT-CLMBR tokens
#         # encoding = self.tokenizer(
#         #     [patient],
#         #     padding="max_length",
#         #     truncation=True,
#         #     max_length=self.max_length,
#         #     return_tensors="pt"
#         # )
#         # encoding = self.tokenizer(
#         #     patient,
#         #     padding="max_length",
#         #     truncation=True,
#         #     max_length=self.max_length,
#         #     return_tensors="pt"
#         # )
#         # vocabulary size
#         # vocab_size = self.tokenizer.vocab_size
#         # print("CLMBR tokenizer vocab size:", vocab_size)
#         # Return dict for DataLoader
#         # data is not pretokenized and when called embedding extraction
#         patient = self.patient_list[idx]  # list[Event] (MEDS)
#         label = self.labels[idx]
#         pid = self.pids[idx]
#
#         return {"text": patient, "label": label, "pid": pid}
#         # return {
#         #     "input_ids": encoding["input_ids"].squeeze(0),  # shape (max_length,)
#         #     "attention_mask": encoding["attention_mask"].squeeze(0),
#         #     "labels": torch.tensor(label, dtype=torch.long),
#         #     "pids": torch.tensor(pid, dtype=torch.long)
#         # }
# # ======================
# # 3. Model with Classification Head
# # ======================
#
# class GPTForClassification(nn.Module):
#     def __init__(self,args, model_dir, num_classes,class_weights=None,explain_mode=False):
#         super().__init__()
#         self.args = args
#         self.class_weights = class_weights
#         self.base_model =  AutoModelForCausalLM.from_pretrained(model_dir)
#         # Enable gradient checkpointing here
#         if self.args.tune_last_layer:
#             self.base_model.gradient_checkpointing_enable()
#         else:
#             self.base_model.gradient_checkpointing_disable()
#         self.explain_mode = explain_mode
#         # if 'StanfordShahLab/' in model_dir:
#         #     model_dir=args.save_dir+model_dir.split('/')[1]+"/"
#         if not os.path.exists(model_dir):
#             # torch.save(model.state_dict(), save_path)
#             os.makedirs(model_dir, exist_ok=True)
#             self.base_model.save_pretrained(model_dir)
#             print(f"✅ Model saved to {model_dir}")
#         else:
#             print(f"⚠️ Model already exists at {model_dir}, skipping save.")
#         # self.base_model.save_pretrained('./clmbr/llama-base-4096-clmbr/')
#         self.dropout = nn.Dropout(p=self.args.dropout)
#         self.classifier = nn.Linear(self.base_model.config.hidden_size, num_classes)
#         # -----------------------
#         # FREEZE ALL
#         # -----------------------
#         if not self.args.tune_last_layer:
#             if not args.finetune_full:
#                 for param in self.base_model.parameters():
#                     param.requires_grad = False
#         else:
#
#             for param in self.base_model.parameters():
#                 param.requires_grad = False
#             self._unfreeze_last_layer()
#
#             # -----------------------
#             # UNFREEZE LAST LAYER
#             # -----------------------
#             # self._unfreeze_last_layer()
#
#             # -----------------------
#             # UNFREEZE CLASSIFIER
#             # -----------------------
#         if not args.embedding:
#             for param in self.classifier.parameters():
#                 param.requires_grad = True
#         # -----------------------
#
#         # HANDLE DIFFERENT BACKBONES
#         # -----------------------
#
#     def get_last_layer(self):
#         try:
#             # BERT / RoBERTa style
#             last_layer = self.base_model.encoder.layer[-1]
#             return last_layer
#
#         except AttributeError:
#             try:
#                 # GPT / LLaMA style
#                 last_layer = self.base_model.model.layers[-1]
#                 return last_layer
#
#             except AttributeError:
#                 try:
#                     # generic fallback
#                     last_layer = self.base_model.layers[-1]
#                     return last_layer
#
#                 except AttributeError:
#                     raise ValueError("Could not locate last layer in base_model")
#
#     def _unfreeze_last_layer(self):
#
#         last_layer = self.get_last_layer()
#         for param in last_layer.parameters():
#             # print('##############finetuning clmbr last layer#########')
#             param.requires_grad = True
#     def get_input_embeddings(self):
#         return self.base_model.model.embed_tokens
#     def set_class_weights(self, class_weights):
#         self.class_weights = class_weights
#     def forward(self, input_ids=None, attention_mask=None, labels=None,pid=None,inputs_embeds=None,output_hidden_states=False):
#         # outputs = self.base_model(
#         #     input_ids=input_ids,
#         #     attention_mask=attention_mask,
#         #     output_hidden_states=True,
#         #     output_attentions=True #added to later analyze attention score based explainability
#         # )
#
#         outputs = self.base_model(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             output_hidden_states=output_hidden_states,inputs_embeds=inputs_embeds,
#
#         )
#
#
#         last_hidden_state = outputs.hidden_states[-1]
#         # better than CLS for EHR sequence models
#         if  "masked_mean_pooling" in self.args.pooling: #remove pad tokens
#             mask = attention_mask.unsqueeze(-1).float()
#             pooled_output = (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
#
#         elif "last_pooling" in self.args.pooling:
#             seq_lengths = attention_mask.sum(dim=1) - 1
#             pooled_output = last_hidden_state[torch.arange(self.args.train_batch_size), seq_lengths]
#
#         elif "first_pooling" in self.args.pooling:
#             pooled_output = last_hidden_state[:, 0, :]  # first token rep. or cls, not suitable for causal lm
#         if self.args.embedding:
#             return pooled_output
#         pooled_output = self.dropout(pooled_output)  # 🔥 HERE
#         logits = self.classifier(pooled_output)
#
#         loss = None
#         if labels is not None:
#             if self.args.weight_balance and self.class_weights is not None:
#                 # print("################### class balance #################")
#                 device = input_ids.device
#                 if 'crossentropy' in self.args.loss:
#                     loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(device),reduction="mean")
#             else:
#                 # print("################### no class balance #################")
#                 if 'crossentropy' in self.args.loss:
#                     loss_fct = nn.CrossEntropyLoss(reduction="mean") #works fine
#             loss = loss_fct(logits.view(-1, self.classifier.out_features), labels.view(-1))
#             # if loss is not None and loss.dim() != 0: worked previously, chatgpt says unnecessary
#             #     print('here')
#             #     loss = loss.mean()
#         if self.explain_mode:
#             return logits
#         else:
#             return SequenceClassifierOutput(
#                 loss=loss,
#                 logits=logits
#             )
# class EmbeddingWrapper(torch.nn.Module):
#     def __init__(self, model):
#         super().__init__()
#         self.model = model
#         self.emb_layer = model.get_input_embeddings()
#
#     def forward(self, embeddings, attention_mask=None):
#         out = self.model(inputs_embeds=embeddings, attention_mask=attention_mask)
#         try:
#             return out.logits
#         except :
#             return out
# def ids_to_embeds(model, input_ids):
#     emb_layer = model.get_input_embeddings()
#     return emb_layer(input_ids)
# #collate batch for hf_tokenizer
# def collate_batch(samples, tokenizer,max_length,padding):
#     #called if data is not pretokenized
#     """
#     Collator that tokenizes, truncates (keep last tokens),
#     and left-pads so recent tokens are preserved at the end.
#     """
#     print(samples[0])
#     print(type(samples[0]))
#     raise SystemExit
#     texts = [s["text"] for s in samples]
#     labels = [s["label"] for s in samples]
#     pids = [s["pid"] for s in samples]
#
#     # tokenize without truncation/padding
#     encodings = tokenizer(
#         texts,
#         add_special_tokens=True,
#         padding=False,
#         truncation=False,
#         return_attention_mask=False,
#         return_tensors=None
#     )
#
#     batch_input_ids = []
#     batch_attention_mask = []
#
#     for ids in encodings["input_ids"]:
#         if len(ids) > max_length:
#             if padding == "left":
#                 # keep last tokens
#                 ids = ids[-max_length:]
#             else:
#                 # keep first tokens
#                 ids = ids[:max_length]
#
#         pad_len = max_length - len(ids)
#
#         if padding == "left":
#             # left-pad
#             padded_ids = [tokenizer.pad_token_id] * pad_len + ids
#             attn_mask = [0] * pad_len + [1] * len(ids)
#         else:
#             # right-pad
#             padded_ids = ids + [tokenizer.pad_token_id] * pad_len
#             attn_mask = [1] * len(ids) + [0] * pad_len
#
#         batch_input_ids.append(padded_ids)
#         batch_attention_mask.append(attn_mask)
#
#     batch = {
#         "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
#         "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
#         "labels": torch.tensor(labels, dtype=torch.long),
#         "pids": torch.tensor(pids, dtype=torch.long)
#
#     }
#     return batch
# def load_dataset(args,path='./case/medsdataset/',train=False,split_index=None,seed=None):
#     dataset_train=None
#     dataset_val=None
#     dataset_test=None
#
#     path_to_extract = path
#     if train:
#         if split_index is None:
#             if args.embedding:
#                 dataset_train = MEDSDataset(args, path_to_extract, split='all', is_debug=False)
#                 pids = dataset_train.get_pids().tolist()
#                 print(f"Loaded n={len(pids)} all patients using extract at: `{path_to_extract}`")
#
#             else:
#                 dataset_train = MEDSDataset(args,path_to_extract, split='train', is_debug=False)
#                 dataset_val = MEDSDataset(args,path_to_extract, split='val', is_debug=False)
#                 pids = dataset_train.get_pids().tolist()
#                 print(f"Loaded n={len(pids)} train patients using extract at: `{path_to_extract}`")
#                 print(f"Loaded n={len(dataset_val.get_pids().tolist())} val  patients using extract at: `{path_to_extract}`")
#         else:
#             print('seed %d'%(seed))
#             dataset_train = MEDSDataset(args,path_to_extract, split='train',split_index=split_index,seed=seed, is_debug=False)
#             dataset_val = MEDSDataset(args,path_to_extract, split='val',split_index=split_index,seed=seed, is_debug=False)
#             pids = dataset_train.get_pids().tolist()
#             print(f"Loaded n={len(pids)}  patients from `{split_index}` seed `{seed}` train fold using extract at : `{path_to_extract}`")
#             pids = dataset_val.get_pids().tolist()
#             print(f"Loaded n={len(pids)}  patients from `{split_index}` seed `{seed}` val fold using extract at: `{path_to_extract}`")
#
#
#     else:
#         dataset_test = MEDSDataset(args,path_to_extract, split='test', is_debug=False)
#         pids = dataset_test.get_pids().tolist()
#         print(f"Loaded test dataset n={len(pids)} patients using extract at: `{path_to_extract}`")
#     # return dataset_train, dataset_val,dataset_test #eta age use korc
#     return dataset_train, dataset_val, dataset_test
#
# def get_label(dataset):
#     # return torch.ones(len(dataset), dtype=torch.long)
#     return torch.from_numpy(dataset.get_labels()).long()    # classification labels
#

def ppv_threshold(probs, labels, k):
    threshold = np.percentile(probs, 100 - k)
    mask = probs >= threshold
    return labels[mask].mean()



def threshold_metrics(y_true, probs, threshold):
    preds = (probs >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return {
        "threshold": threshold,
        "accuracy": sklearn.metrics.accuracy_score(y_true, preds),
        "balanced_accuracy": balanced_accuracy_score(y_true, preds),
        "precision": precision_score(y_true, preds, zero_division=0),
        "recall": recall,
        "specificity": specificity,
        "f1": sklearn.metrics.f1_score(y_true, preds, zero_division=0),
    }


def pick_best_threshold(
    y_true,
    probs,
    thresholds,
    optimize_for="balanced_accuracy",
):
    rows = []

    for t in thresholds:
        m = threshold_metrics(y_true, probs, t)
        rows.append(m)

    threshold_df = pd.DataFrame(rows)

    best_row = (
        threshold_df
        .sort_values(
            [optimize_for, "recall", "precision"],
            ascending=[False, False, False],
        )
        .iloc[0]
    )
    best_metrics = best_row.to_dict()

    return best_metrics


def print_metrics(args,y_true, y_proba,inference_time=None,max_length=2048,padding="left",epochs=10,train_batch_size=4,tokenizer_dir=None,label_df=None,result_dir=None,fold=None,printing=True):
    batch_times=inference_time
    # title=label_df.split(".")[0]
    # y_pred = y_proba > 0.5
    auroc = sklearn.metrics.roc_auc_score(y_true, y_proba)
    aps = sklearn.metrics.average_precision_score(y_true, y_proba)
    # accuracy = sklearn.metrics.accuracy_score(y_true, y_pred)
    # f1 = sklearn.metrics.f1_score(y_true, y_pred)
    # Calculate micro-averaged precision and recall
    # micro_precision = sklearn.metrics.precision_score(y_true, y_pred, average='micro')
    # micro_recall = sklearn.metrics.recall_score(y_true, y_pred, average='micro')
    # micro_f1 = sklearn.metrics.f1_score(y_true, y_pred, average='micro')
    # Calculate macro-averaged precision and recall
    # macro_precision = sklearn.metrics.precision_score(y_true, y_pred, average='macro')
    # macro_recall = sklearn.metrics.recall_score(y_true, y_pred, average='macro')
    ppv_1=ppv_threshold(y_proba, y_true, 1)
    ppv_5 = ppv_threshold(y_proba, y_true, 5)
    ppv_10 = ppv_threshold(y_proba, y_true, 10)
    # ---- Confusion matrix ----
    # tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    #
    # # ---- Specificity ---- TN / (TN + FP)
    # specificity = tn / (tn + fp)
    # balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    brier_score = brier_score_loss(y_true, y_proba)

    # ---- PPV = Precision ---- TP / (TP + FP)
    # ppv = precision_score(y_true, y_pred)
    auprc = average_precision_score(y_true, y_proba)
    if args.kfold==1:
        thrs=np.array([  0.50])
    else:
        thrs = np.array([ 0.30,0.40, 0.50])
    metrics_to_save = pick_best_threshold(
        y_true,
        y_proba,
        thresholds =  thrs,
        optimize_for="balanced_accuracy",
    )
    counts = Counter(y_true)
    metrics_to_save['case'] = counts[1]
    metrics_to_save['control'] = counts[0]
    metrics_to_save['auroc']=auroc
    metrics_to_save['aps'] = aps
    metrics_to_save['auprc'] = auprc
    metrics_to_save['ppv_1'] = ppv_1
    metrics_to_save['ppv_5'] = ppv_5
    metrics_to_save['ppv_10'] = ppv_10
    metrics_to_save['brier_score'] = brier_score

    # metrics_to_save = {
    #     'case': counts[1],
    #     'control': counts[0],
    #
    #     'auroc': auroc,
    #     'aps': aps,
    #     'micro_precision': micro_precision,
    #     'micro_recall': micro_recall,
    #     'micro_f1': micro_f1,
    #     'macro_precision': macro_precision,
    #     'macro_recall': macro_recall,
    #     'macro_f1': macro_f1,
    #     'ppv': ppv,
    #     'ppv_1': ppv_1,
    #     'ppv_5': ppv_5,
    #     'ppv_10': ppv_10,
    #     'specificity': specificity,
    #     'auprc': auprc,
    #     'balanced_accuracy': balanced_accuracy,
    #     'brier_score': brier_score
    #
    # }
    print("fold no." )
    print(fold)
    # print("Specificity:", specificity)
    # print("PPV (Precision):", ppv)
    # Macro F1 score
    # macro_f1 = sklearn.metrics.f1_score(y_true, y_pred, average='macro')

    if printing:
        for key, value in metrics_to_save.items():
            print(f"{key}: {value}\n")

        # print(f"Macro F1 Score: {macro_f1}")
        # print("\tAUROC:", auroc)
        # print("\tAPS:", aps)
        # print("\tAccuracy:", accuracy)
        # print("\tF1 Score:", f1)
        # print("\tmicro_precision:", micro_precision)
        # print("\tmicro_recall:", micro_recall)
        # print(f"\tMicro F1 Score:", micro_f1)
        # print("\tmacro_precision:", macro_precision)
        # print("\tmacro_recall:", macro_recall)
        # print(f"\tMacro F1 Score:", macro_f1)
        # print(f"\tMacro F1 Score:", macro_f1)
        # print(f"\tPPV:", ppv)
        # print(f"\tPPV_1:", ppv_1)
        # print(f"\tPPV_5:", ppv_5)
        # print(f"\tPPV_10:", ppv_10)
        # print(f"\tspecificity:", specificity),
        # print(f"\tauprc:", auprc),
        # print(f"\tbalanced_accuracy:", balanced_accuracy)
        # print(f"\tbrier_score:", brier_score)
        # brier_score


        # print(f"\tmean_batch_time_s", float(batch_times.mean()))
        # print(f"\tstd_batch_time_s", float(batch_times.std()))
        # print(f"median_batch_time_s", float(np.median(batch_times)))
        # print(f"\tp95_batch_time_s", float(np.percentile(batch_times, 95)))


    # if fold is None:
    #     # os.makedirs(os.path.dirname(result_dir), exist_ok=True)
    #     #
    #     # file_path_name=tokenizer_dir.split('/')[1]+"_epoch_"+str(epochs)+"_bs_"+str(train_batch_size)+"_maxlen_"+str(max_length)+"_pad_"+padding+title
    #     # file_path = result_dir+'model_metrics_'+file_path_name+'.json'
    #     # with open(file_path, 'w') as f:
    #     #     json.dump(metrics_to_save, f, indent=4)  # indent for pretty printing
    #     # pd.DataFrame({'true':y_true,'pred':y_pred,'prob':y_proba}).to_csv(result_dir+'raw_result_'+file_path_name+'.csv')
    #     return None
    # else:
    return metrics_to_save
