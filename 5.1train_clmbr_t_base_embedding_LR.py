import json
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from utils_dp import *
from utils_dp import param_grid_xgboost,param_grid_lightgbm

#load all emb w pid label
# param_grid = [
#     {"C": 0.001, "penalty": "l2", "solver": "liblinear", "class_weight": "balanced"},
#     {"C": 0.01,  "penalty": "l2", "solver": "liblinear", "class_weight": "balanced"},
#     {"C": 0.1,   "penalty": "l2", "solver": "liblinear", "class_weight": "balanced"},
#     {"C": 1.0,   "penalty": "l2", "solver": "liblinear", "class_weight": "balanced"},
#
#     {"C": 0.001, "penalty": "l2", "solver": "liblinear", "class_weight": None},
#     {"C": 0.01,  "penalty": "l2", "solver": "liblinear", "class_weight": None},
#     {"C": 0.1,   "penalty": "l2", "solver": "liblinear", "class_weight": None},
#     {"C": 1.0,   "penalty": "l2", "solver": "liblinear", "class_weight": None},
#
# ]
param_grid_l2 = [
    {"C": 0.001, "penalty": "l2", "solver": "liblinear", "class_weight": "balanced","l1_ratio":None},




    {"C": 0.001, "penalty": "l2", "solver": "liblinear", "class_weight": None,"l1_ratio":None},




]
param_grid_elasticnet = [

{"C": 0.001, "penalty": "elasticnet", "solver": "liblinear", "class_weight": "balanced","l1_ratio":0.1},
    {"C": 0.01,  "penalty": "elasticnet", "solver": "liblinear", "class_weight": "balanced","l1_ratio":0.1},



    {"C": 0.001, "penalty": "elasticnet", "solver": "liblinear", "class_weight": None,"l1_ratio":0.1},
    {"C": 0.01,  "penalty": "elasticnet", "solver": "liblinear", "class_weight": None,"l1_ratio":0.1},
{"C": 0.001, "penalty": "elasticnet", "solver": "liblinear", "class_weight": "balanced","l1_ratio":0.3},
    {"C": 0.01,  "penalty": "elasticnet", "solver": "liblinear", "class_weight": "balanced","l1_ratio":0.3},



    {"C": 0.001, "penalty": "elasticnet", "solver": "liblinear", "class_weight": None,"l1_ratio":0.3},
    {"C": 0.01,  "penalty": "elasticnet", "solver": "liblinear", "class_weight": None,"l1_ratio":0.3},






]
def load_utilisation_baseline_file(path):
    data = np.load(path, allow_pickle=True)
    X = np.log1p(
        data["x_util"]
    )


    pids = data["pids"]
    y = data["labels"]

    pid_to_idx = {
        pid: i for i, pid in enumerate(pids)
    }

    return X, pids, y, pid_to_idx


def select_lifestyle_npz_by_pid_order(args,npz_paths, pid_order):
    """
        Load multiple lifestyle NPZ files, reorder each to pid_order,
        concatenate features, and return one aligned matrix.
        """

    if isinstance(npz_paths, str):
        npz_paths = [npz_paths]

    pid_order = np.asarray(pid_order).astype(int)

    X_list = []
    feature_name_list = []
    pids_ref = None

    for path in npz_paths:
        if "covariates" in args.include:
            print(f"##############Extract demographics covariates values from######### {path}")
        else:
            print(f"##############Extract lifestyle values from######### {path}")
        data = np.load(path, allow_pickle=True)

        X = data["features"]
        pids = data["pids"].astype(int)
        feature_names = data["feature_names"].astype(str)
        # if "covariates" in args.include:
        #     age = data[f"age_{args.label_col}"].reshape(-1, 1)
        #
        #     X = np.hstack([X, age])
        #
        #     feature_names = np.append(
        #         feature_names,
        #         f"age_{args.label_col}"
        #     )

        print(f"# of feat: {len(feature_names)}, name: {feature_names}")

        pid_to_idx = {
            int(pid): i
            for i, pid in enumerate(pids)
        }

        missing = [
            int(pid)
            for pid in pid_order
            if int(pid) not in pid_to_idx
        ]

        if missing:
            raise ValueError(
                f"{path}: Missing {len(missing)} pids, first: {missing[:10]}"
            )

        idx = [
            pid_to_idx[int(pid)]
            for pid in pid_order
        ]

        X_sel = X[idx]
        pids_sel = pids[idx]

        assert np.array_equal(
            pids_sel.astype(int),
            pid_order.astype(int),
        ), f"PID order mismatch in {path}"

        X_list.append(X_sel.astype(np.float32))

        # prefix feature names by file stem to avoid duplicate names
        prefix = path.split("/")[-1].replace(".npz", "")
        feature_name_list.extend(
            [f"{prefix}__{name}" for name in feature_names]
        )

        if pids_ref is None:
            pids_ref = pids_sel
        else:
            assert np.array_equal(pids_ref.astype(int), pids_sel.astype(int))

    X_concat = np.concatenate(X_list, axis=1)

    return X_concat,pids_ref,np.array(feature_name_list, dtype=str)

    ############previous############
    # data = np.load(npz_path, allow_pickle=True)
    #
    # X = data["features"]
    # pids = data["pids"].astype(int)
    # feature_names = data["feature_names"]
    #
    # pid_to_idx = {int(pid): i for i, pid in enumerate(pids)}
    #
    # missing = [int(pid) for pid in pid_order if int(pid) not in pid_to_idx]
    # if missing:
    #     raise ValueError(f"Missing {len(missing)} pids, first: {missing[:10]}")
    #
    # idx = [pid_to_idx[int(pid)] for pid in pid_order]
    #
    # X_sel = X[idx]
    # pids_sel = pids[idx]
    #
    # assert np.array_equal(pids_sel.astype(int), np.array(pid_order).astype(int))
    #
    # return X_sel, pids_sel, feature_names

def load_embedding_file(path):
    data = np.load(path, allow_pickle=True)

    X = data["embeddings"]
    pids = data["pids"]
    y = data["labels"]

    pid_to_idx = {
        pid: i for i, pid in enumerate(pids)
    }

    return X, pids, y, pid_to_idx


#select rows by train/val/test ids
def select_by_ids(X, pids, y, pid_to_idx, selected_ids):
    idx = [
        pid_to_idx[pid]
        for pid in selected_ids
        if pid in pid_to_idx
    ]

    missing = [
        pid for pid in selected_ids
        if pid not in pid_to_idx
    ]

    if len(missing) > 0:
        print("Missing IDs:", len(missing))
        print("First missing:", missing[:10])
    unique, counts = np.unique(y[idx], return_counts=True)

    label_counts = {
        int(k): int(v)
        for k, v in zip(unique, counts)
    }



    return X[idx], y[idx], pids[idx],missing
def concat_lifestyle_with_clmbr(X_clmbr_train,X_life_train,pid_clmbr_train,pid_life_train):
    # X_life_train, pid_life_train, life_names = select_lifestyle_by_ids(
    #     f"mental_substance_features_offset_{offset_month}.npz",
    #     pid_clmbr_train,
    # )
    # print(len(pid_life_train))
    # print(len(pid_life_train))
    assert np.array_equal(pid_life_train.astype(int), pid_clmbr_train.astype(int))

    X_fused_train = np.concatenate(
        [X_clmbr_train, X_life_train],
        axis=1,
    )
    return X_fused_train
def make_model(args,params):
    print("########debug###########")
    print(args.lr_penalty)
    if args.lr_penalty=='elasticnet':
        model = Pipeline([
            ("scaler", StandardScaler()),

            ("clf", LogisticRegression(
                penalty="elasticnet",
                solver="saga",

                C=params.get("C", 1.0),
                l1_ratio=params.get("l1_ratio", None),
                class_weight=params.get("class_weight", "balanced"),
                max_iter=params.get("max_iter", 2000),
                random_state=42,
            ))
        ])
    else:
        model=Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            penalty= "l2",
            C=params.get("C", 1.0),
            solver= "liblinear",
            class_weight=params.get("class_weight", "balanced"),
            max_iter=params.get("max_iter", 2000),
            random_state=42,
        ))
    ])
    return model

def make_lifestyle_npz_file_list(args):
    # lifestyle_npz_paths = [
    #     f"./engineered_features/smoking_features_offset_{args.label_col}.npz",
    #     f"./engineered_features/alcohol_features_offset_{args.label_col}.npz",
    #     f"./engineered_features/sleep_features_offset_{args.label_col}.npz",
    #     f"./engineered_features/ms_features_offset_{args.label_col}.npz",
    #     f"./engineered_features/fd_features_offset_{args.label_col}.npz",
    #     f"./engineered_features/sp_features_offset_{args.label_col}.npz",
    # ]
    #
    lifestyle_npz_paths=[]
    if len(args.include)>0 and ("smoking" in args.include or 'all' in args.include):
        lifestyle_npz_paths.append(f"./engineered_features/smoking_features_offset_{args.label_col}.npz")
    if len(args.include)>0 and ("alcohol" in args.include or 'all' in args.include):
        print("Alcohol only")
        lifestyle_npz_paths.append(f"./engineered_features/alcohol_features_offset_{args.label_col}.npz")
    if len(args.include)>0 and ("sleep" in args.include or 'all' in args.include):
        lifestyle_npz_paths.append(f"./engineered_features/sleep_features_offset_{args.label_col}.npz")
    if len(args.include)>0 and ("mental_substance" in args.include or 'all' in args.include):
        lifestyle_npz_paths.append(f"./engineered_features/ms_features_offset_{args.label_col}.npz")
    if len(args.include)>0 and ("dementia" in args.include  or 'all' in args.include):
        lifestyle_npz_paths.append(f"./engineered_features/fd_features_offset_{args.label_col}.npz")
    if len(args.include)>0 and ("social_phobia" in args.include or 'all' in args.include):
        lifestyle_npz_paths.append(f"./engineered_features/sp_features_offset_{args.label_col}.npz")
    if len(args.include)>0 and ("covariates" in args.include):
        lifestyle_npz_paths.append(f"./engineered_features/demographics.npz")
    print(lifestyle_npz_paths)
    return lifestyle_npz_paths

def create_lifestyle_concatenated_df(args,pid_tr,pid_va,X_tr,X_va):


    lifestyle_npz_path = make_lifestyle_npz_file_list(args)
    if len(lifestyle_npz_path) == 0:
        raise ValueError("No lifestyle npz files found")

    X_tr_ls, pid_tr_ls, life_names = select_lifestyle_npz_by_pid_order(args,
                                                                       lifestyle_npz_path,
                                                                       pid_tr,
                                                                       )

    X_va_ls, pid_va_ls, _ = select_lifestyle_npz_by_pid_order(args,
                                                              lifestyle_npz_path,
                                                              pid_va,
                                                              )

    assert np.array_equal(pid_tr_ls.astype(int), pid_tr.astype(int))
    assert np.array_equal(pid_va_ls.astype(int), pid_va.astype(int))
    print("########actual feat shape######")
    print(f"train: {X_tr.shape}")
    print(f"val: {X_va.shape}")
    print("########cov feat shape######")
    print(f"train: {X_tr_ls.shape}")
    print(f"val: {X_va_ls.shape}")

    X_tr = concat_lifestyle_with_clmbr(X_tr, X_tr_ls, pid_tr, pid_tr_ls)
    X_va = concat_lifestyle_with_clmbr(X_va, X_va_ls, pid_va, pid_va_ls)
    print("########after concat feat shape######")
    print(f"train: {X_tr.shape}")
    print(f"val: {X_va.shape}")
    return X_tr, X_va


#  For 3-fold CV on the 70% training set:
def run_lr_embedding_outer_cv(args,
    X_all, pids_all, y_all, pid_to_idx,path_cv,model_config

):

    metrics_per_param = []
    model_name = define_model_name(args)
    if args.classifier=="lr":
        if args.lr_penalty==   'elasticnet':
            param_grid=param_grid_elasticnet
        elif args.lr_penalty==   'l2':
            param_grid=param_grid_l2
    elif args.classifier=="xgboost":
        param_grid=param_grid_xgboost
    elif args.classifier == "lightgbm":
        param_grid=param_grid_lightgbm

    for config_id, params in enumerate(param_grid):

        metrics_per_fold = []

        for k in range(args.kfold):
            print("fold %d " % (k))
            cv_path = path_cv + str(k) + "/full.parquet"
            splits = pd.read_parquet(cv_path, engine='pyarrow')
            tr_ids = splits[
                (splits["split"] == "train")
            ]["patient_id"].to_numpy()
            va_ids = splits[
                (splits["split"] == "test")
            ]["patient_id"].to_numpy()

            X_tr, y_tr, pid_tr,missing = select_by_ids(
                X_all,pids_all, y_all,  pid_to_idx, tr_ids
            )
            if len(missing) > 0:
                raise ValueError("Missing IDs:", len(missing))
                sys.exit()

            X_va, y_va, pid_va,missing = select_by_ids(
                X_all,pids_all, y_all,  pid_to_idx, va_ids
            )
            if len(missing) > 0:
                raise ValueError("Missing IDs:", len(missing))
                sys.exit()
            if len(args.include)>0:
                X_tr,X_va=create_lifestyle_concatenated_df(args, pid_tr, pid_va, X_tr, X_va)


            model = make_model(args,params)
            model.fit(X_tr, y_tr)

            probs = model.predict_proba(X_va)[:, 1]

            metrics = print_metrics(args,y_va, probs)
            metrics["fold"] = k
            metrics["config_id"] = config_id

            metrics_per_fold.append(metrics)

            print(
                f"config={config_id}, fold={k}, "
                f"AUROC={metrics['auroc']:.4f}, APS={metrics['aps']:.4f}"
            )
        fold_df = pd.DataFrame(metrics_per_fold)

        penalty = args.lr_penalty
        if args.lr_penalty == None:
            penalty = ""
        row = {
            "config_id": config_id,
            "cohort_definition": args.cohort_definition,
            "offset_month": args.label_col,
            "train_test_type": 'cv_' + str(args.kfold) if args.kfold >= 1 else 'test',
            'model_name':model_name ,
            'classifier':args.classifier if args.lr_penalty=='l2' else args.classifier+"_"+penalty,
            'max_len':args.max_length,
            **params,
        }

        # numeric columns only (avoid fold, config_id, etc.)
        metric_cols = fold_df.select_dtypes(include=[np.number]).columns

        for col in metric_cols:
            if col in ["fold", "config_id","offset_month"]:
                continue

            row[f"{col}_mean"] = fold_df[col].mean()

            # std only if >1 fold
            row[f"{col}_std"] = fold_df[col].std(ddof=1) if len(fold_df) > 1 else 0.0
        if config_id == 0:
            header = row.keys()
        else:
            header = None

        # log_row(args.result_dir + 'results_clmbr_embed_'+args.classifier+'.csv',
        #         row=list(row.values()),
        #         header=header)
        log_row_dict(args.result_dir + 'results_clmbr_embed_'+args.classifier+'v4.csv',row)
        metrics_per_param.append(row)


    summary = pd.DataFrame(metrics_per_param)
    save_df_local_gcs(summary,'results_clmbr_embed_'+args.classifier,args.result_dir)
    best_params = None
    try:
        if 'auroc_mean' in summary.columns and 'auroc_std' in summary.columns:
            summary["stability_score"] = summary["auroc_mean"] - summary["auroc_std"]

            best_row = summary.sort_values(
                ["stability_score", "auroc_mean"],
                ascending=[False, False]
            ).iloc[0]

            best_params = {}

            for k, v in param_grid[0].items():

                if isinstance(v, int):
                    best_params[k] = int(best_row[k])

                elif isinstance(v, float):
                    best_params[k] = float(best_row[k])

                elif isinstance(v, bool):
                    best_params[k] = bool(best_row[k])

                else:
                    best_params[k] = best_row[k]
            print("#############for model config: ###################")
            print(model_config)
            print("###############best params are: ##############")
            print(best_params)
    except Exception as e:
        print(e)


    return summary, best_params
def clean_params(params):
    params = {
        k: (None if pd.isna(v) else v)
        for k, v in params.items()
    }
    return params

# python 5.1train_clmbr_t_base_embedding_LR.py  > log_emb_clmbr_t_base_lr.txt 2>&1
# python 5.1train_clmbr_t_base_embedding_LR.py  --sensitivity_curve> log_emb_clmbr_t_base_lr.txt 2>&1
# python 5.1train_clmbr_t_base_embedding_LR.py  --reliability_curve> log_emb_clmbr_t_base_lr.txt 2>&1

# python 5.1train_clmbr_t_base_embedding_LR.py --subcohort_analysis  > log_emb_clmbr_t_base_lr.txt 2>&1

# python 5.1train_clmbr_t_base_embedding_LR.py  --baseline_utils > log_emb_base_utils_lr.txt 2>&1
# python 5.1train_clmbr_t_base_embedding_LR.py  --smoking > log_emb_base_utils_lr.txt 2>&1
# python 5.1train_clmbr_t_base_embedding_LR.py  --baseline_utils --include  > log_emb_base_utils_lr.txt 2>&1
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-GPU GPT-CLMBR Fine-tuning with Gradient Accumulation")

    parser.add_argument("--kfold", type=int, default=1)  # torchrun passes this

    parser.add_argument("--log_file", type=str, default="logs/train_log.txt", help="Path to log file")

    parser.add_argument("--train_test_path", type=str, default='./train_test_split_',
                        help="Path to log file")
    parser.add_argument("--model_dir", type=str, default="StanfordShahLab/clmbr-t-base",
                        help="Path to log file")
    parser.add_argument("--baseline_utils",  action="store_true", help="baseline_utils feature or not")
    parser.add_argument("--result_dir", type=str, default="./clmbr/",
                        help="Path to log file")

    parser.add_argument("--lr_penalty", type=str, default="l2",
                        help="Path to log file")



    parser.add_argument("--label_df", type=str, default="./checkpoints/matched_final",
                        help="Path to log file")
    parser.add_argument("--cohort_definition", type=str, default="icd_confirmed_first",
                        help="Path to log file")
    parser.add_argument("--classifier", type=str, default="lr",
                        help="Path to log file")
    parser.add_argument("--embedding_type", type=str, default="w_value_dedup",
                        help="Path to log file")
    parser.add_argument("--label_col", type=int, default=5,
                        help="Path to log file")
    parser.add_argument("--max_length", type=int, default=2048, help="Path to log file")
    parser.add_argument("--save_dir", type=str, default="./clmbr/", help="Path to log file")
    parser.add_argument("--exclude", type=str, default="",
                        help="name of lifestyle domain u want to exclude from model, all means no lifestyle feature modeled, "" means add all lf feat ")
    parser.add_argument("--include", type=str, default="", help="what lifestyle feat. to include")
    parser.add_argument("--subcohort_analysis", action="store_true", help="subcohort_analysis")
    parser.add_argument("--reliability_curve", action="store_true", help="reliability_curve")
    parser.add_argument("--sensitivity_curve", action="store_true", help="sensitivity_curve")





    # parser.add_argument("--last_pooling", action="store_true", help="do training or not")
    # parser.add_argument("--cls_pooling", action="store_true", help="do training or not")

    # parser.add_argument(
    #     "--smoking",
    #     action="store_true",
    #     help="use smoking feature"
    # )

    args = parser.parse_args()

    param_df=None
    if args.classifier == "lr" and args.kfold == 1:
        param_df=pd.read_csv("./clmbr/results_clmbr_embed_lrv4.csv")
    # for iters,config in enumerate([{"model_name":"StanfordShahLab/mamba-tiny-16384-clmbr","max_length":16384,"offset_month":6,"cd":"icd_confirmed_first"},
    #                {"model_name":"StanfordShahLab/mamba-tiny-16384-clmbr","max_length":16384,"offset_month":12,"cd":"icd_confirmed_first"},
    #                {"model_name":"StanfordShahLab/mamba-tiny-16384-clmbr","max_length":16384,"offset_month":24,"cd":"icd_confirmed_first"},
    #                {"model_name":"StanfordShahLab/mamba-tiny-16384-clmbr","max_length":16384,"offset_month":36,"cd":"icd_confirmed_first"},
    #                {"model_name": "StanfordShahLab/mamba-tiny-16384-clmbr", "max_length": 16384, "offset_month": 12,
    #                 "cd": "icd_or_drug"},
    #                {"model_name": "StanfordShahLab/mamba-tiny-16384-clmbr", "max_length": 16384, "offset_month": 24,
    #                 "cd": "icd_or_drug"},
    #                {"model_name": "StanfordShahLab/mamba-tiny-16384-clmbr", "max_length": 16384, "offset_month": 36,
    #                 "cd": "icd_or_drug"},
    #                {"model_name": "StanfordShahLab/mamba-tiny-16384-clmbr", "max_length": 16384, "offset_month": 6,
    #                 "cd": "icd_or_drug"},
    #                {"model_name": "StanfordShahLab/mamba-tiny-16384-clmbr", "max_length": 16384, "offset_month": 12,
    #                 "cd": "drug_only"},
    #                {"model_name": "StanfordShahLab/mamba-tiny-16384-clmbr", "max_length": 16384, "offset_month": 24,
    #                 "cd": "drug_only"},
    #                {"model_name": "StanfordShahLab/mamba-tiny-16384-clmbr", "max_length": 16384, "offset_month": 36,
    #                 "cd": "drug_only"},
    #                {"model_name": "StanfordShahLab/mamba-tiny-16384-clmbr", "max_length": 16384, "offset_month": 6,
    #                 "cd": "drug_only"},
    #                ]):
    config_list = make_model_config()
    for covariate in ["smoking","all","","alcohol"]:
        for iters, config in enumerate(config_list):
            print("###################model_config###############")
            print(config)

            args.model_dir = config["model_name"].split('/')[1]
            args.classifier = config["classifier"]

            args.max_length = config["max_length"]
            args.cohort_definition = config["cd"]

            args.label_col = config["offset_month"]
            args.include=covariate
            print(f"#######      Offset_Month      #########{args.label_col}")


            # -------------------------------
            # load embeddings
            # -------------------------------
            if args.baseline_utils:
                out_dir = "./baseline/"
                file_name="utilization_features_"+str(args.label_col)+".npz"
                X_all, pids_all,y_all, pid_to_idx = load_utilisation_baseline_file(out_dir + file_name)
            else:
                if 'w_value_dedup' in args.embedding_type:
                    out_dir = "./clmbr/embeddings_w_value_dedup/"
                    if "clmbr-t-base" in args.model_dir:
                        file_name = "clmbr_embeddings_" + args.model_dir + "_" + str(args.label_col) + ".npz"
                    else:
                        file_name = args.model_dir.replace("-", "_") + "_embeddings_"+ str(args.label_col) + ".npz"

                    # print(embedding_name)

                    # file_name ="mamba_tiny_16384_clmbr_embeddings_"  + str(args.label_col) + ".npz"
                elif 'value' in args.embedding_type:
                    out_dir = "./clmbr/embeddings_w_value/"
                    file_name = "clmbr_embeddings_" + args.model_dir + "_" + str(args.label_col) + ".npz"
                else:
                    out_dir = "./clmbr/embeddings/"

                    file_name = "clmbr_embeddings_" + args.model_dir + "_" + str(args.label_col) + ".npz"
                X_all, pids_all,y_all, pid_to_idx = load_embedding_file(out_dir+file_name)

            print("X:", X_all.shape)
            print("y:", y_all.shape)
            print("pids:", pids_all.shape)


            if args.kfold > 2:
                # -------------------------------
                # load saved train/test/CV ids
                # -------------------------------
                fullpath = args.train_test_path + args.cohort_definition + '/cv_3/fold_'





                run_lr_embedding_outer_cv(args,
                                          X_all, pids_all, y_all, pid_to_idx,fullpath,config

                                          )

            elif args.kfold == 1:
                # stability score
                if args.subcohort_analysis and args.cohort_definition == 'icd_or_drug':
                    if args.label_col==36:
                        if "llama-base-2048" in args.model_dir:
                            args.include="all"
                            print(f"############{args.include}########")
                        # elif args.model_dir.str.contains("gpt-base-2048"):
                model_name = define_model_name(args)
                classifier_name = args.classifier if args.lr_penalty == 'l2' else args.classifier + "_" + args.lr_penalt

                # stability score
                if args.cohort_definition == "cd4":
                    summary = param_df[
                        (param_df['cohort_definition'] == "icd_confirmed_first") & (
                                param_df['offset_month'] == args.label_col) & (
                                param_df['model_name'] == model_name) & (param_df['classifier'] == classifier_name)]

                else:
                    summary = param_df[
                        (param_df['cohort_definition'] == args.cohort_definition) & (
                                    param_df['offset_month'] == args.label_col) & (param_df['model_name'] == model_name) & (
                                    param_df['classifier'] == classifier_name)]
                summary = summary.copy()
                summary["stability_score"] = summary["auroc_mean"] - summary["auroc_std"]

                # best param row per offset_month
                best_rows = (
                    summary
                    .sort_values(["offset_month", "stability_score"], ascending=[True, False])
                    .groupby(["offset_month", "cohort_definition", "model_name", "classifier"], as_index=False)
                    .head(1)
                    .reset_index(drop=True)
                )


                if args.classifier == "lr":
                    param_col = ['C',
                                 'penalty',
                                 'solver',
                                 'class_weight','l1_ratio',"threshold_mean"]
                if len(best_rows)>0:
                    print(f"more than 1 best param for {args.cohort_definition} offsetm {args.label_col}")
                config_id=best_rows['config_id'].values[0]
                params = clean_params(best_rows[param_col].iloc[0].to_dict())
                fullpath = args.train_test_path + args.cohort_definition + "/train_test_70_30.parquet"
                splits = pd.read_parquet(fullpath, engine='pyarrow')
                train_ids = splits[
                    (splits["split"] == "train")
                ]["patient_id"].to_numpy()
                test_ids = splits[
                    (splits["split"] == "test")
                ]["patient_id"].to_numpy()
                X_train70, y_train70, pid_train70,missing = select_by_ids(
                    X_all, pids_all,y_all,  pid_to_idx, train_ids
                )
                if len(missing) > 0:
                    raise ValueError("Missing IDs in train 70-30:", len(missing))
                    sys.exit()

                X_test, y_test, pid_test,missing = select_by_ids(
                    X_all,pids_all, y_all,  pid_to_idx, test_ids
                )
                if len(missing) > 0:
                    raise ValueError("Missing IDs in test 70-30:", len(missing))
                    sys.exit()
                if len(args.include)>0:
                    X_train70,X_test=create_lifestyle_concatenated_df(args, pid_train70, pid_test, X_train70, X_test)
                print("Train70 in cohort:", X_train70.shape, np.bincount(y_train70),args.cohort_definition)
                print("Test30 in cohort:", X_test.shape, np.bincount(y_test),args.cohort_definition)
                # best param row per offset_month
                if args.subcohort_analysis and args.cohort_definition == 'icd_or_drug':

                    meta_df=pd.read_parquet('./checkpoints/meta_df_case_control_all_v1.parquet', engine='pyarrow')
                    test_df = make_subcohort_df(meta_df, X_test, y_test,pid_test, model_path=args.result_dir + "final_clmbr_lr_model" + args.cohort_definition + "_" + str(
                                    args.label_col) + "_" + model_name + "_" + classifier_name + ".joblib", model_family=model_name,
                                                classifier=classifier_name,
                                             cohort_definition=args.cohort_definition, offset_month=args.label_col)
                    run_subcohort_analysis(args, test_df)
                    continue
                if args.sensitivity_curve:
                    make_sensitivity_analysis_df(X_test,y_test,args.result_dir + "final_clmbr_lr_model" + args.cohort_definition + "_" + str(
                                        args.label_col) + "_" + model_name + "_" + classifier_name + ".joblib",model_name=model_name,
                                                 classifier=classifier_name,cohort_definition=args.cohort_definition,offset_month=args.label_col,cv_3_threshold=params['threshold_mean'])
                    continue
                if args.reliability_curve:
                    make_reliability_analysis_df(X_test, y_test, model_path=args.result_dir + "final_clmbr_lr_model" + args.cohort_definition + "_" + str(
                                        args.label_col) + "_" + model_name + "_" + classifier_name + ".joblib", model_family=model_name, classifier=classifier_name,
                                                 cohort_definition=args.cohort_definition, offset_month=args.label_col)

                    continue







                final_model = make_model(args,params)
                print(final_model)
                final_model.fit(X_train70, y_train70)

                test_probs = final_model.predict_proba(X_test)[:, 1]
                metrics = print_metrics(args,y_test, test_probs)



                model_name = define_model_name(args)
                penalty = args.lr_penalty
                if args.lr_penalty == None:
                    penalty = ""
                row = {
                    "config_id": config_id,
                    "cohort_definition": args.cohort_definition,
                    "offset_month": args.label_col,
                    "train_test_type": 'cv_' + str(args.kfold) if args.kfold > 2 else 'test',
                    'model_name': model_name,
                    'classifier':args.classifier if args.lr_penalty=='l2' else args.classifier+"_"+penalty,
                    'max_len': args.max_length,
                    **params,
                }



                for col in metrics.keys():
                    if col in ["fold", "config_id", "offset_month"]:
                        continue

                    row[f"{col}"] = metrics[col]

                    # std only if >1 fold
                    row[f"{col}_std"] = 0.0
                if iters == 0:
                    header = row.keys()
                else:
                    header = None
                try:
                    # log_row(args.result_dir + 'results_clmbr_embed_train_test_70_30_' + args.classifier + '.csv',
                    #         row=list(row.values()),
                    #         header=header)
                    log_row_dict(args.result_dir + 'results_clmbr_embed_train_test_70_30_' + args.classifier + 'v4.csv',row)
                except Exception as e:
                    print(e)




                try:
                    out_file = f"test_pred_70_30_{args.cohort_definition}.parquet"

                    prob_col = f"prob_{model_name}_{classifier_name}_{args.label_col}"
                    pred_col = f"pred_{model_name}_{classifier_name}_{args.label_col}"

                    new_df = pd.DataFrame({
                        "pid": pid_test.astype(int),
                        "label": y_test.astype(int),
                        pred_col: test_probs,
                    })
                    if not os.path.exists(args.result_dir +out_file):
                        new_df.to_parquet(args.result_dir +out_file, index=False)
                        print(f"saved test pred in new file {out_file}")
                    else:
                        existing_df = pd.read_parquet(args.result_dir +out_file)

                        # sanity checks
                        assert (existing_df["pid"].values == pid_test.astype(int)).all()
                        assert (existing_df["label"].values == y_test.astype(int)).all()

                        existing_df[prob_col] = test_probs
                        existing_df[pred_col] = (test_probs >= 0.5).astype(int)

                        existing_df.to_parquet(args.result_dir +out_file, index=False)
                        print(f"saved test pred in existing file {out_file}")
                except Exception as e:
                    print(e)
                    try:
                        test_pred_df = pd.DataFrame({
                            "pid": pid_test.astype(int),
                            "label": y_test.astype(int),
                            "prob": test_probs,
                            "pred": (test_probs >= 0.5).astype(int),
                        })
                        test_pred_df.to_parquet(args.result_dir + "test_predictions_" + args.cohort_definition + "_" + str(
                            args.label_col) + "_" + model_name + "_" + classifier_name + ".parquet", index=False)
                        print(f"saved test pred in individual file ")

                        joblib.dump(final_model,
                                    args.result_dir + "final_clmbr_lr_model" + args.cohort_definition + "_" + str(
                                        args.label_col) + "_" + model_name + "_" + classifier_name + ".joblib")
                        print(f"saved model ")

                        # except Exceptio
                    except Exception as e:
                        print(e)





