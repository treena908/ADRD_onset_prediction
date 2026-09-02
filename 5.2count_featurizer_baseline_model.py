
from utils_dp import *
from utils_dp import param_grid_xgboost,param_grid_lightgbm
from scipy import sparse
from topk_feat_shap_value_analysis import *
from reliability_curve_train_test import *
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.linear_model import LogisticRegression

from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import OneHotEncoder
#load all emb w pid label
param_grid_l2 = [
    {"C": 0.001, "penalty": "l2", "solver": "liblinear", "class_weight": "balanced","l1_ratio":None},




    {"C": 0.001, "penalty": "l2", "solver": "liblinear", "class_weight": None,"l1_ratio":None},




]
param_grid_elasticnet = [


    {"C": 0.01,  "penalty": "elasticnet", "solver": "saga", "class_weight": "balanced","l1_ratio":0.1},

    {"C": 0.01,  "penalty": "elasticnet", "solver": "saga", "class_weight": None,"l1_ratio":0.1},

    {"C": 0.01,  "penalty": "elasticnet", "solver": "saga", "class_weight": "balanced","l1_ratio":0.3},

    {"C": 0.01,  "penalty": "elasticnet", "solver": "saga", "class_weight": None,"l1_ratio":0.3}

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


def select_lifestyle_npz_by_pid_order(npz_paths, pid_order):
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
        print(f"##############Extract lifestyle values from######### {path}")
        data = np.load(path, allow_pickle=True)

        X = data["features"]
        pids = data["pids"].astype(int)
        feature_names = data["feature_names"].astype(str)
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
def concat_lifestyle_with_count_features(X_clmbr_train,X_life_train,pid_clmbr_train,pid_life_train):
    # X_life_train, pid_life_train, life_names = select_lifestyle_by_ids(
    #     f"mental_substance_features_offset_{offset_month}.npz",
    #     pid_clmbr_train,
    # )
    # print(len(pid_life_train))
    # print(len(pid_life_train))
    assert np.array_equal(pid_life_train.astype(int), pid_clmbr_train.astype(int))
    print(type(X_clmbr_train), np.shape(X_clmbr_train))
    print(type(X_life_train), np.shape(X_life_train))

    X_life_train = np.asarray(X_life_train)

    if not sparse.issparse(X_life_train):
        X_life_train = sparse.csr_matrix(X_life_train)
    print(X_clmbr_train.shape)
    print(X_life_train.shape)
    X_fused_train = sparse.hstack(
        [X_clmbr_train, X_life_train],
        format="csr"
    )

    print(X_fused_train.shape)


    return X_fused_train

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
    print(lifestyle_npz_paths)
    return lifestyle_npz_paths

def count_feature_qc(args,X_count_train,feature_names):
    feature_prevalence = (
            (X_count_train > 0)
            .sum(axis=0)
            .A1
            / X_count_train.shape[0]
    )

    feat_df = pd.DataFrame({
        "feature": feature_names,
        "prevalence": feature_prevalence,
    })

    print(
        feat_df.sort_values(
            "prevalence",
            ascending=False,
        ).head(50)
    )
    save_df_local_gcs(feat_df,'feature_prevalence_'+args.cohort_definition+'_'+str(args.label_col),'./checkpoints/')
    # qc patients with 0 feat
    row_nnz = np.diff(X_count_train.indptr)

    empty_rows = row_nnz == 0

    print(
        "Patients with zero generated features:",
        empty_rows.sum()
    )

    print(
        "Percent:",
        100 * empty_rows.mean()
    )
    return feat_df,empty_rows.sum()
# 4. QC demographic feature imbalance
def qc_demo_features(X, y, feature_names):
    feature_names = np.asarray(feature_names)

    rows = []

    for i, f in enumerate(feature_names):
        x = (X[:, i] > 0).toarray().ravel()

        rows.append({
            "feature": f,
            "case_prev": x[y == 1].mean(),
            "ctrl_prev": x[y == 0].mean(),
            "abs_diff": abs(x[y == 1].mean() - x[y == 0].mean()),
            "n_case": x[y == 1].sum(),
            "n_ctrl": x[y == 0].sum(),
        })

    out = pd.DataFrame(rows).sort_values("abs_diff", ascending=False)

    print(out)

    return out
#  For 3-fold CV on the 70% training set:
def create_lifestyle_concatenated_df(args,pid_tr,pid_va,X_tr,X_va):
    lifestyle_npz_path = make_lifestyle_npz_file_list(args)
    if len(lifestyle_npz_path) == 0:
        raise ValueError("No lifestyle npz files found")

    X_tr_ls, pid_tr_ls, life_names = select_lifestyle_npz_by_pid_order(
        lifestyle_npz_path,
        pid_tr,
    )

    X_va_ls, pid_va_ls, _ = select_lifestyle_npz_by_pid_order(
        lifestyle_npz_path,
        pid_va,
    )

    assert np.array_equal(pid_tr_ls.astype(int), pid_tr.astype(int))
    assert np.array_equal(pid_va_ls.astype(int), pid_va.astype(int))

    X_tr = concat_lifestyle_with_count_features(X_tr, X_tr_ls, pid_tr, pid_tr_ls)
    X_va = concat_lifestyle_with_count_features(X_va, X_va_ls, pid_va, pid_va_ls)
    return X_tr, X_va,life_names
def run_count_featurizer_outer_cv(args,patient_day_df,meta_df,
    path_cv,model_config

):

    metrics_per_param = []
    model_name = define_model_name(args)
    if args.classifier=='lr' and args.lr_penalty=='elasticnet':
        param_grid=param_grid_elasticnet
    elif args.classifier=='lr' and args.lr_penalty=='l2':
        param_grid=param_grid_l2
    elif args.classifier=='xgboost' :
        param_grid=param_grid_xgboost
    elif args.classifier=='lightgbm' :
        param_grid=param_grid_lightgbm
    #debug demo feat
    if 'baseline_utils' in args.model_dir:

        featurizer = TimeBinnedCountFeaturizer(
            offset_month=args.label_col,
            use_binary_counts=False,
            add_utilization=True,
            add_demographics=True
        )
    else:
        featurizer = TimeBinnedCountFeaturizer(
            offset_month=args.label_col,
            use_binary_counts=True,
            add_utilization=True,
            add_demographics=True
        )


    # featurizer = TimeBinnedCountFeaturizer(offset_month=args.label_col,
    #                                        use_binary_counts = True,
    # add_utilization = True,
    # add_demographics = False,
    # )
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
            y_tr = splits[
                (splits["split"] == "train")
            ]["label"].to_numpy()
            y_va = splits[
                (splits["split"] == "test")
            ]["label"].to_numpy()

            # overlap = set(tr_ids) & set(va_ids)
            #
            # print("overlap1: train len, val_len", len(overlap),len(tr_ids),len(va_ids))
            # pre_qc = featurizer.qc_before_featurization(
            #     patient_day_df=patient_day_df,
            #     meta_df=meta_df,
            #     pid_order=tr_ids,
            #     label_col="pheno",
            # )








            X_count_train, pid_count_train = featurizer.fit_transform(
                patient_day_df,
                meta_df,
                pid_order=tr_ids,
            )
            if args.qc:
                post_qc = featurizer.qc_after_featurization(
                    X=X_count_train,
                    pids=pid_count_train,
                    feature_names=featurizer.feature_names_,
                    y=y_tr,
                )

            X_count_val, pid_count_val = featurizer.transform(
                patient_day_df,
                meta_df,
                pid_order=va_ids,
            )
            # post_qc = featurizer.qc_after_featurization(
            #     X=X_count_val,
            #     pids=pid_count_val,
            #     feature_names=featurizer.feature_names_,
            #     y=y_va,
            # )

            # overlap = set(pid_count_train) & set(pid_count_val)
            #
            # print("overlap2: train len, val_len", len(overlap),len(pid_count_train),len(pid_count_val))
            assert np.array_equal(pid_count_train.astype(int), tr_ids.astype(int))
            assert np.array_equal(pid_count_val.astype(int), va_ids.astype(int))

            print("#########train val feat shape################")
            print(X_count_train.shape)
            print(X_count_val.shape)
            print("#########top10 feat debug################")
            print(featurizer.feature_names_[:10])
            if args.qc:

                print("demo feat qc train")
                demo_qc_train = qc_demo_features(
                    X_count_train,
                    y_tr,
                    featurizer.feature_names_,
                )
                print("demo feat qc val")
                demo_qc_val = qc_demo_features(
                    X_count_val,
                    y_va,
                    featurizer.feature_names_,
                )
                feat_df,patient_zero_feat=count_feature_qc(args,X_count_train,featurizer.feature_names_)
                if patient_zero_feat>0:
                    raise ValueError("# patient with Missing feat :", patient_zero_feat)
                    sys.exit()

            if args.prune_rare_feat:
                min_patients = 10

                keep_mask = (
                        (X_train > 0)
                        .sum(axis=0)
                        .A1
                        >= min_patients
                )

                X_count_train = X_count_train[:, keep_mask]
                X_count_val = X_count_val[:, keep_mask]
                feature_names = feature_names[keep_mask]
                print("#########train val feat shape after pruning rare feat :################")
                print(X_count_train.shape)
                print(X_count_val.shape)

            if len(args.include)>0:
                X_count_train,X_count_val,lifestyle_feature_names=create_lifestyle_concatenated_df(args, pid_count_train, pid_count_val, X_count_train, X_count_val)



            model = make_model(args,params)
            model.fit(X_count_train, y_tr)

            probs = model.predict_proba(X_count_val)[:, 1]
            # print("######y_probs############")
            # print(np.array(probs).shape)

            metrics = print_metrics(args,y_va, probs)
            metrics["fold"] = k
            metrics["config_id"] = config_id

            metrics_per_fold.append(metrics)

            print(
                f"config={config_id}, fold={k}, "
                f"AUROC={metrics['auroc']:.4f}, APS={metrics['aps']:.4f}"
            )
        fold_df = pd.DataFrame(metrics_per_fold)
        penalty=args.lr_penalty
        if args.lr_penalty==None:
            penalty=""

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
    print(params)

    int_params = {
        "n_estimators",
        "max_depth",
        "min_child_weight",
        "max_leaves",
        "num_leaves",
        "max_bin",
        "n_jobs",
    }

    for k in int_params:
        if k in params and params[k] is not None:
            params[k] = int(params[k])

    return params
# python 5.2count_featurizer_baseline_model.py  > log_emb_count_feat_lr.txt 2>&1
# python 5.2count_featurizer_baseline_model.py --sensitivity_curve > log_emb_count_feat_lr.txt 2>&1
# python 5.2count_featurizer_baseline_model.py --reliability_curve > log_emb_count_feat_lr.txt 2>&1
# python 5.2count_featurizer_baseline_model.py --shap_value > log_emb_count_feat_lr_shap.txt 2>&1
# python 5.2count_featurizer_baseline_model.py --subcohort_analysis > log_emb_count_feat_lr.txt 2>&1



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-GPU GPT-CLMBR Fine-tuning with Gradient Accumulation")

    parser.add_argument("--kfold", type=int, default=1)  # torchrun passes this

    parser.add_argument("--log_file", type=str, default="logs/train_log.txt", help="Path to log file")

    parser.add_argument("--train_test_path", type=str, default='./train_test_split_',
                        help="Path to log file")
    parser.add_argument("--model_dir", type=str, default="count_featurizer",
                        help="Path to log file")
    parser.add_argument("--baseline_utils",  action="store_true", help="baseline_utils feature or not")
    parser.add_argument("--result_dir", type=str, default="./clmbr/",
                        help="Path to log file")

    parser.add_argument("--lr_penalty", type=str, default="elasticnet",
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

    parser.add_argument("--include", type=str, default="", help="what lifestyle feat. to include")



    parser.add_argument("--subcohort_analysis", action="store_true", help="subcohort_analysis")
    parser.add_argument("--shap_value", action="store_true", help="shap_value explanation")
    parser.add_argument("--reliability_curve", action="store_true", help="reliability_curve")
    parser.add_argument("--sensitivity_curve", action="store_true", help="sensitivity_curve")

    parser.add_argument("--prune_rare_feat", action="store_true", help="prune rare feat of count feat")
    parser.add_argument("--qc", action="store_true", help="quality check")

    # parser.add_argument("--cls_pooling", action="store_true", help="do training or not")

    # parser.add_argument(
    #     "--smoking",
    #     action="store_true",
    #     help="use smoking feature"
    # )

    args = parser.parse_args()

    param_df=None

    meta_df = pd.read_parquet("./checkpoints/meta_df_case_control_all.parquet", engine='pyarrow')
    if args.embedding_type=="w_value_dedup":
        patient_day_df = pd.read_parquet("./checkpoints/patient_day_code_df_w_value_dedup_case_control_all.parquet",
                                         engine='pyarrow')
    elif args.embedding_type=="w_value":
        patient_day_df = pd.read_parquet("./checkpoints/patient_day_code_df_w_value_case_control_all.parquet",
                                         engine='pyarrow')
    else:
        patient_day_df = pd.read_parquet("./checkpoints/patient_day_code_df_case_control_all.parquet",
                                         engine='pyarrow')

    config_list=make_model_config()
    # for iters,config in enumerate([{"model_name":"StanfordShahLab/clmbr-t-base","max_length":496,"offset_month":6,"cd":"icd_confirmed_first"},
    #                {"model_name":"StanfordShahLab/clmbr-t-base","max_length":496,"offset_month":12,"cd":"icd_confirmed_first"},
    #                {"model_name":"StanfordShahLab/clmbr-t-base","max_length":496,"offset_month":24,"cd":"icd_confirmed_first"},
    #                {"model_name":"StanfordShahLab/clmbr-t-base","max_length":496,"offset_month":36,"cd":"icd_confirmed_first"},
    #                {"model_name": "StanfordShahLab/clmbr-t-base", "max_length": 496, "offset_month": 12,
    #                 "cd": "icd_or_drug"},
    #                {"model_name": "StanfordShahLab/clmbr-t-base", "max_length": 496, "offset_month": 24,
    #                 "cd": "icd_or_drug"},
    #                {"model_name": "StanfordShahLab/clmbr-t-base", "max_length": 496, "offset_month": 36,
    #                 "cd": "icd_or_drug"},
    #                {"model_name": "StanfordShahLab/clmbr-t-base", "max_length": 496, "offset_month": 6,
    #                 "cd": "icd_or_drug"},
    #                {"model_name": "StanfordShahLab/clmbr-t-base", "max_length": 496, "offset_month": 12,
    #                 "cd": "drug_only"},
    #                {"model_name": "StanfordShahLab/clmbr-t-base", "max_length": 496, "offset_month": 24,
    #                 "cd": "drug_only"},
    #                {"model_name": "StanfordShahLab/clmbr-t-base", "max_length": 496, "offset_month": 36,
    #                 "cd": "drug_only"},
    #                {"model_name": "StanfordShahLab/clmbr-t-base", "max_length": 496, "offset_month": 6,
    #                 "cd": "drug_only"},
    #                ]):
    # for covariate in ["","all","smoking","alcohol"]:
    for covariate in ["all"]:
        for iters, config in enumerate(config_list):
            config["include"]=covariate
            print("###################model_config###############")
            print(config)



            args.cohort_definition = config["cd"]

            args.label_col = config["offset_month"]
            args.model_dir = config["model_name"]
            args.classifier = config["classifier"]
            args.lr_penalty = config["lr_penalty"]
            args.include=covariate
            if args.kfold == 1:
                if args.classifier == "lr":
                    param_df = pd.read_csv("./clmbr/results_clmbr_embed_lrv4.csv")
                elif "xgboost" in args.classifier :
                    print('xgboost')
                    param_df = pd.read_csv("./clmbr/results_clmbr_embed_xgboostv4.csv")
                elif "lightgbm" in args.classifier:
                    print('lightgbm')
                    param_df = pd.read_csv("./clmbr/results_clmbr_embed_lightgbmv4.csv")


            if args.kfold > 2:
                # -------------------------------
                # load saved train/test/CV ids
                # -------------------------------
                fullpath = args.train_test_path + args.cohort_definition + '/cv_3/fold_'





                run_count_featurizer_outer_cv(args,patient_day_df,meta_df,
                                          fullpath,config

                                          )



            elif args.kfold == 1:
                if 'baseline_utils' in args.model_dir:

                    featurizer = TimeBinnedCountFeaturizer(
                        offset_month=args.label_col,
                        use_binary_counts=False,
                        add_utilization=True,
                        add_demographics=True
                    )
                else:
                    featurizer = TimeBinnedCountFeaturizer(
                        offset_month=args.label_col,
                        use_binary_counts=True,
                        add_utilization=True,
                        add_demographics=True
                    )
                model_name = define_model_name(args)
                classifier_name=args.classifier if args.lr_penalty=='l2' else args.classifier+"_"+args.lr_penalty

                # stability score
                if args.cohort_definition=="cd4":
                    summary = param_df[
                        (param_df['cohort_definition'] == "icd_confirmed_first") & (
                                    param_df['offset_month'] == args.label_col) & (
                                    param_df['model_name'] == model_name) & (param_df['classifier'] == classifier_name)]

                else:
                    summary = param_df[
                        (param_df['cohort_definition'] == args.cohort_definition) & (param_df['offset_month'] == args.label_col) & (param_df['model_name'] == model_name)& (param_df['classifier'] == classifier_name)]
                summary = summary.copy()
                summary["stability_score"] = summary["auroc_mean"] - summary["auroc_std"]

                # best param row per offset_month
                best_rows = (
                    summary
                    .sort_values(["offset_month", "stability_score"], ascending=[True, False])
                    .groupby(["offset_month","cohort_definition","model_name","classifier"], as_index=False)
                    .head(1)
                    .reset_index(drop=True)
                )
                # print(best_rows.columns.tolist())


                if "lr" in args.classifier:
                    param_col = ['C',
                                 'penalty',
                                 'solver',
                                 'class_weight','l1_ratio']
                elif "xgboost" in args.classifier :

                    param_col = ['max_depth',
                                 'learning_rate',
                                 'n_estimators',
                                 'subsample','colsample_bytree']

                elif "lightgbm" in args.classifier:
                    param_col = ["num_leaves",
                            "learning_rate",
                            "n_estimators",
                            "subsample",
                            "colsample_bytree","threshold_mean"]

                if len(best_rows)>0:
                    print(f"more than 1 best param for {args.cohort_definition} offsetm {args.label_col}")
                # print(best_rows)
                    config_id=best_rows['config_id'].values[0]
                    params = clean_params(best_rows[param_col].iloc[0].to_dict())
                elif len(best_rows)==0:
                    if "xgboost" in args.classifier:
                        params = clean_params(param_grid_xgboost[0])
                    elif "lightgbm" in args.classifier:
                        params = clean_params(param_grid_lightgbm[0])
                    config_id = 0
                    if len(params)==0:
                        print(f"no param found {config}")
                        continue
                    # if ((args.cohort_definition=="drug_only" )|(args.cohort_definition=="icd_confirmed_first"))& (("xgboost" in args.classifier) | ("lightgbm" in args.classifier)) & (args.label_col==36):
                    #     print("ekhane ashche")
                    #     if "xgboost" in args.classifier:
                    #         params=clean_params(param_grid_xgboost[0])
                    #     elif "lightgbm" in args.classifier:
                    #         params=clean_params(param_grid_lightgbm[0])
                    #     config_id=0
                    # else:
                    #     print(f"no param found {config}")
                    #     continue
                fullpath = args.train_test_path + args.cohort_definition + "/train_test_70_30.parquet"
                splits = pd.read_parquet(fullpath, engine='pyarrow')
                train_ids = splits[
                    (splits["split"] == "train")
                ]["patient_id"].to_numpy()
                test_ids = splits[
                    (splits["split"] == "test")
                ]["patient_id"].to_numpy()
                y_tr = splits[
                    (splits["split"] == "train")
                ]["label"].to_numpy()
                y_va = splits[
                    (splits["split"] == "test")
                ]["label"].to_numpy()
                X_count_train, pid_count_train = featurizer.fit_transform(
                    patient_day_df,
                    meta_df,
                    pid_order=train_ids,
                )
                if args.qc:
                    post_qc = featurizer.qc_after_featurization(
                        X=X_count_train,
                        pids=pid_count_train,
                        feature_names=featurizer.feature_names_,
                        y=y_tr,
                    )

                X_count_val, pid_count_val = featurizer.transform(
                    patient_day_df,
                    meta_df,
                    pid_order=test_ids,
                )


                print("Train70 in cohort:", X_count_train.shape, np.bincount(y_tr),args.cohort_definition)
                print("Test30 in cohort:", X_count_val.shape, np.bincount(y_va),args.cohort_definition)
                if len(args.include) > 0:
                    X_count_train, X_count_val,lifestyle_feature_names = create_lifestyle_concatenated_df(args, pid_count_train, pid_count_val,X_count_train, X_count_val)
                    print("Train70 in cohort:", X_count_train.shape, np.bincount(y_tr), args.cohort_definition)
                    print("Test30 in cohort:", X_count_val.shape, np.bincount(y_va), args.cohort_definition)

                # best param row per offset_month
                if args.subcohort_analysis and args.cohort_definition == 'icd_or_drug':
                    meta_df=pd.read_parquet('./checkpoints/meta_df_case_control_all_v1.parquet', engine='pyarrow')
                    test_df = make_subcohort_df(meta_df, X_count_val, y_va,pid_count_val, model_path=args.result_dir + "final_count_model" + args.cohort_definition + "_" + str(
                                    args.label_col) + "_" + model_name + "_" + classifier_name + ".joblib", model_family=model_name,
                                                classifier=classifier_name,
                                             cohort_definition=args.cohort_definition, offset_month=args.label_col)
                    run_subcohort_analysis(args, test_df)
                    continue
                if args.sensitivity_curve:
                    make_sensitivity_analysis_df(X_count_val,y_va,args.result_dir + "final_count_model" + args.cohort_definition + "_" + str(
                                    args.label_col) + "_" + model_name + "_" + classifier_name + ".joblib",model_name=model_name,
                                                 classifier=classifier_name,cohort_definition=args.cohort_definition,offset_month=args.label_col,cv_3_threshold=params['threshold_mean'])
                    continue

                if args.reliability_curve:
                    make_reliability_analysis_df(X_count_val, y_va, model_path=args.result_dir + "final_count_model" + args.cohort_definition + "_" + str(
                                    args.label_col) + "_" + model_name + "_" + classifier_name + ".joblib", model_family=model_name, classifier=classifier_name,
                                                 cohort_definition=args.cohort_definition, offset_month=args.label_col)

                    continue
                if args.shap_value:
                    count_feature_names = featurizer.feature_names_
                    lifestyle_feature_names = np.asarray(
                        lifestyle_feature_names,
                        dtype=str,
                    )

                    feature_names = np.concatenate(
                        [count_feature_names, lifestyle_feature_names]
                    )

                    assert X_count_val.shape[1] == len(feature_names)

                    run_final_model_explanations(
                       args.result_dir + "final_count_model" + args.cohort_definition + "_" + str(
                            args.label_col) + "_" + model_name + "_" + classifier_name + ".joblib", X_count_val, y_va,pid=pid_count_val, feature_names=feature_names, cohort_definition=args.cohort_definition, offset_month=args.label_col,
                        output_dir="./figures/shap_values/",
                    top_n= 20,
                    max_rows = 2000,
                    )
                    continue
                final_model = make_model(args,params)
                final_model.fit(X_count_train, y_tr)

                test_probs = final_model.predict_proba(X_count_val)[:, 1]
                metrics = print_metrics(args,y_va, test_probs)




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
                        "pid": pid_count_val.astype(int),
                        "label": y_va.astype(int),
                        prob_col: test_probs,
                        pred_col:(test_probs >= 0.5).astype(int)
                    })
                    if not os.path.exists(args.result_dir +out_file):
                        new_df.to_parquet(args.result_dir +out_file, index=False)
                        print(f"saved test pred in new file {out_file}")
                    else:
                        existing_df = pd.read_parquet(args.result_dir +out_file)

                        # sanity checks
                        assert (existing_df["pid"].values == pid_count_val.astype(int)).all()
                        assert (existing_df["label"].values == y_va.astype(int)).all()

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


                        # except Exceptio
                    except Exception as e:
                        print(e)
                try:
                    joblib.dump(final_model,
                                args.result_dir + "final_count_model" + args.cohort_definition + "_" + str(
                                    args.label_col) + "_" + model_name + "_" + classifier_name + ".joblib")
                    print(f"saved model ")
                except Exception as e:
                    print('model save failed')
                    print(e)

                try:
                    if args.subcohort_analysis :
                        test_df=make_subcohort_df( meta_df, pid_count_val, y_va, test_probs)
                        run_subcohort_analysis(args,test_df)
                except Exception as e:
                    print(e)

                # test_pred_df.to_parquet(args.result_dir + "test_predictions_"+args.cohort_definition+"_"+str(args.label_col)+"_"+model_name+"_"+classifier_name+".parquet", index=False)
                #
                #
                #
                # joblib.dump(final_model, args.result_dir + "final_clmbr_lr_model"+args.cohort_definition+"_"+str(args.label_col)+"_"+model_name+"_"+classifier_name+".joblib")
                # except Exception as e:
                #     print(e)

        # if args.model_dir=="baseline_utils":
        #     break
        # break