import os
import gc
import numpy as np
import pandas as pd
import torch
from utils_dp import *
from tqdm import tqdm
from typing import List, Dict, Optional, Any
from transformers import AutoModelForCausalLM
from hf_ehr.data.tokenization import CLMBRTokenizer
from hf_ehr.config import Event
from huggingface_hub import login
HF_TOKEN="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"
login(token=HF_TOKEN)
model_specs = {
    # "clmbr-t-base":
    #     {
    #     "model_name": "StanfordShahLab/clmbr-t-base",
    #     "max_len": 496,
    #
    #     },

    "llama_base_2048_clmbr": {
        "model_name": "StanfordShahLab/llama-base-2048-clmbr",
        "max_len": 2048,
    },
#
#
#     "mamba_tiny_16384_clmbr": {
#         "model_name": "StanfordShahLab/mamba-tiny-16384-clmbr",
#         "max_len": 16384,
#     },
"gpt_base_2048_clmbr": {
        "model_name": "StanfordShahLab/gpt-base-2048-clmbr",
        "max_len": 2048,
    },
# "llama_base_512_clmbr": {
#             "model_name": "StanfordShahLab/llama-base-512-clmbr",
#             "max_len": 512,
#         },
# "gpt_base_512_clmbr": {
#         "model_name": "StanfordShahLab/gpt-base-512-clmbr",
#         "max_len": 512,
#     },
}
def measurement_to_event(m, event_date):
    d = dict(m)

    code = d.get("code")
    if code is None:
        return None

    event_dt = pd.to_datetime(
        event_date,
        errors="coerce",
        utc=True,
    )

    if pd.isna(event_dt):
        start = None
        end = None
    else:
        start = event_dt.to_pydatetime()
        end = event_dt.to_pydatetime()

    value = d.get("numeric_value", None)
    unit = d.get("unit", None)
    domain = d.get("domain", None)

    if pd.isna(value):
        value = None

    if pd.isna(unit):
        unit = None
    elif unit is not None:
        unit = str(unit)

    return Event(
        code=str(code),
        value=value,
        unit=unit,
        start=start,
        end=end,
        omop_table=domain,
    )
def patient_df_to_hfehr_events(
    patient_df,
    birth_datetime=None,
    final_index_date=None,
    offset_month=None,
    birth_code='SNOMED/3950001',
):
    df = patient_df.copy()

    df["event_date"] = pd.to_datetime(
        df["event_date"],
        utc=True,
        errors="coerce",
    )

    # prediction time filtering
    if (
        final_index_date is not None
        and offset_month is not None
    ):
        prediction_time = (
            pd.to_datetime(
                final_index_date,
                utc=True,
            )
            - pd.DateOffset(months=offset_month)
        )

        df = df[
            df["event_date"] < prediction_time
        ]

    df = df.sort_values("event_date")

    events = []

    # birth event
    if birth_datetime is not None:

        birth_dt = pd.to_datetime(
            birth_datetime,
            utc=True,
            errors="coerce",
        )

        if pd.notna(birth_dt):
            events.append(
                Event(
                    code=birth_code,
                    start=birth_dt.to_pydatetime(),
                    end=birth_dt.to_pydatetime(),omop_table='person'
                )
            )

    # convert measurements
    for _, row in df.iterrows():

        event_dt = pd.to_datetime(
            row["event_date"],
            utc=True,
        ).to_pydatetime()

        measurements = row["measurements"]
        if measurements is None:
            continue

        if isinstance(measurements, np.ndarray):
            measurements = measurements.tolist()
        for m in measurements:
            ev = measurement_to_event(
                m,
                event_date=event_dt
            )

            if ev is not None:
                events.append(ev)



    return events


def build_hfehr_patients(
    patient_day_df,
    meta_df,
    pid_order,
    pid_col="person_id",
    birth_col="birth_datetime",offset_month=6
):
    patient_day_df = patient_day_df.copy()
    meta_df = meta_df.copy()
    patient_event_len=[]

    patient_day_df[pid_col] = patient_day_df[pid_col].astype(int)
    meta_df[pid_col] = meta_df[pid_col].astype(int)

    birth_lookup = (
        meta_df[[pid_col, birth_col]]
        .drop_duplicates(pid_col)
        .set_index(pid_col)[birth_col]
        .to_dict()
    )
    index_date_lookup = (
        meta_df[[pid_col, "final_index_date"]]
        .drop_duplicates(pid_col)
        .set_index(pid_col)["final_index_date"]
        .to_dict()
    )

    grouped = {
        int(pid): g
        for pid, g in patient_day_df.groupby(pid_col)
    }

    patients = []
    pids_out = []

    for pid in np.asarray(pid_order).astype(int):
        g = grouped.get(int(pid))

        birth_datetime = birth_lookup.get(int(pid), None)
        final_index_date = index_date_lookup.get(int(pid), None)

        if g is None:

            events = patient_df_to_hfehr_events(
                pd.DataFrame(columns=patient_day_df.columns),
                birth_datetime=birth_datetime,final_index_date=final_index_date,offset_month=offset_month
            )
            patient_event_len.append(len(events))
        else:
            events = patient_df_to_hfehr_events(
                g,
                birth_datetime=birth_datetime,final_index_date=final_index_date,offset_month=offset_month
            )
            patient_event_len.append(len(events))

        patients.append(events)
        pids_out.append(int(pid))
    print('patient_event_len_raw')
    print(pd.Series(patient_event_len).describe())
    return patients, np.array(pids_out, dtype=np.int64)
def truncate_batch_to_recent(batch, max_len):
    out = {}

    for k, v in batch.items():
        if torch.is_tensor(v) and v.ndim == 2:
            out[k] = v[:, -max_len:]
        else:
            out[k] = v

    return out
def extract_hfehr_embeddings(
    patients,
    pids,
    model_name,
    max_len,
    batch_size=1,
    device=None,
    pooling="last_hidden",
):
    """
    pooling:
      - "last_logits": representation = logits[:, -1, :]
        matches hf_ehr model card example.
      - "last_hidden": uses hidden_states[-1][:, -1, :]
        may be preferable if model supports output_hidden_states.
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = CLMBRTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
    )

    model.to(device)
    model.eval()

    all_embs = []
    all_pids = []
    raw_lens = []
    input_lens = []

    with torch.no_grad():
        for i in tqdm(range(0, len(patients), batch_size), desc=model_name):
            batch_patients = patients[i:i + batch_size]
            batch_pids = pids[i:i + batch_size]

            # raw token length QC
            for pid, patient in zip(batch_pids, batch_patients):
                raw_lens.append(
                    len(tokenizer.convert_events_to_tokens(patient))
                )

            batch = tokenizer(
                batch_patients,
                add_special_tokens=True,
                return_tensors="pt",
            )

            batch = truncate_batch_to_recent(
                batch,
                max_len=max_len,
            )

            # llama requires removing token_type_ids
            batch.pop("token_type_ids", None)

            batch = {
                k: v.to(device)
                for k, v in batch.items()
                if torch.is_tensor(v)
            }

            if "attention_mask" in batch:
                input_lens.extend(
                    batch["attention_mask"].sum(dim=1)
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(int)
                    .tolist()
                )
            else:
                input_lens.extend(
                    [batch["input_ids"].shape[1]] * len(batch_pids)
                )

            if pooling == "last_logits":
                outputs = model(**batch)
                emb = outputs.logits[:, -1, :]

            elif pooling == "last_hidden":
                outputs = model(
                    **batch,
                    output_hidden_states=True,
                )
                emb = outputs.hidden_states[-1][:, -1, :]

            else:
                raise ValueError(f"Unknown pooling: {pooling}")

            all_embs.append(
                emb.detach().cpu().numpy().astype(np.float32)
            )

            all_pids.extend(batch_pids)

    X = np.vstack(all_embs)
    pids_out = np.array(all_pids, dtype=np.int64)

    qc = pd.DataFrame({
        "person_id": pids_out,
        "raw_token_len": raw_lens,
        "input_token_len": input_lens,
        "was_truncated": np.array(raw_lens) + 3 > np.array(input_lens),
        # optional
        "tokens_removed":
            np.maximum(
                np.array(raw_lens) + 3 - np.array(input_lens),
                0,
            ),

        "pct_tokens_kept":
            np.array(input_lens)
            / (np.array(raw_lens) + 3),
    })

    del model
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return X, pids_out, qc
def save_embedding_npz(
    out_path,
    X,
    pids,
    labels=None,
    qc=None,
):
    payload = {
        "embeddings": X.astype(np.float32),
        "pids": pids.astype(np.int64),
    }

    if labels is not None:
        payload["labels"] = labels.astype(np.int32)

    if qc is not None:
        payload["raw_token_len"] = qc["raw_token_len"].values.astype(np.int32)
        payload["input_token_len"] = qc["input_token_len"].values.astype(np.int32)
        payload["was_truncated"] = qc["was_truncated"].values.astype(np.int8)

    np.savez_compressed(
        out_path,
        **payload,
    )

# python 4.3generate_hf_ehr_clmbr_embedding.py > log_emb_generate.txt 2>&1
if __name__ == "__main__":
    EMBED_QC=True

    # patient_day_df = pd.read_parquet('./checkpoints/patient_day_code_df_w_value_dedup_case_control_all.parquet',engine='pyarrow')
    # meta_df=pd.read_parquet('./checkpoints/meta_df_case_control_all.parquet', engine='pyarrow')
    patient_day_df = pd.read_parquet('./data/patient_day_df_dedup_w_val_uchicago.parquet',
                                     engine='pyarrow')
    meta_df = pd.read_csv('./data/meta_df_case_control_all.csv', engine='pyarrow')
    person_measurements = (
        patient_day_df.groupby("person_id")["n_codes"]
        .sum()
    )
    print("##########before########")
    print(person_measurements.describe())
    pid_order = meta_df["person_id"].values.astype(int)
    for offset_month in [36,6,12,24]:

        patients, pids = build_hfehr_patients(
            patient_day_df=patient_day_df,
            meta_df=meta_df,
            pid_order=pid_order,offset_month=offset_month
        )
        # continue
        all_outputs = {}

        for tag, spec in model_specs.items():
            print(f"     ###############offset_m {offset_month} model name {tag}#################    ")
            X, pids_out, qc = extract_hfehr_embeddings(
                patients=patients,
                pids=pids,
                model_name=spec["model_name"],
                max_len=spec["max_len"],
                batch_size=1,
                pooling="last_hidden",
            )

            all_outputs[tag] = {
                "features": X,
                "pids": pids_out,
                "qc": qc,
            }

            print(tag)
            print("X:", X.shape)
            print(qc["raw_token_len"].describe())
            print(qc["input_token_len"].describe())
            print(qc["was_truncated"].mean())
            bad_id = qc.loc[(qc["raw_token_len"] == 0) ]['person_id']
            print(f"patients having no event after tokenization {len(bad_id)}")
            if len(bad_id) >0:
                print(bad_id[0])
                idx=np.where(pids==bad_id[0])[0]
                print(patients[idx])
                print(bad_id)
                raise ValueError(f"bad id with no event after tokenization {len(bad_id)}")
            # bad_id=qc.loc[(qc["raw_token_len"]>2)&(qc["input_token_len"]==0)]['person_id']
            # print(f"patients having no event after tokenization {len(bad_id)}")
            # bad_id = qc.loc[(qc["raw_token_len"] ==1) & (qc["input_token_len"] == 0)]['person_id']
            # print(f"patients having no event after tokenization {len(bad_id)}")
            # bad_id = qc.loc[(qc["raw_token_len"]>2 ) & (qc["input_token_len"] == 1)]['person_id']
            # print(f"patients having 1 event after tokenization {len(bad_id)}")
            # bad_id = qc.loc[ (qc["input_token_len"] == 0)]['person_id']
            # print(f"patients having no event after tokenization {len(bad_id)}")
            labels = meta_df.set_index("person_id").loc[pids, "pheno"].values

            for tag, obj in all_outputs.items():
                out_path = f"./clmbr/embeddings_w_value_dedup/{tag}_embeddings_{str(offset_month)}.npz"
                if out_path:
                    os.makedirs("./clmbr/embeddings_w_value_dedup/", exist_ok=True)



                save_embedding_npz(
                    out_path=out_path,
                    X=obj["features"],
                    pids=obj["pids"],
                    labels=labels,
                    qc=obj["qc"],
                )
                print("locally saved")

                # try:
                #     import subprocess
                #     WORKSPACE_BUCKET = os.environ["WORKSPACE_BUCKET"]
                #     GCS_OUT_DIR = f"{WORKSPACE_BUCKET}/ADRD_onset/checkpoints/"
                #     subprocess.run(["gsutil", "-m", "cp", str(out_path), GCS_OUT_DIR + "/"], check=True)
                #     print("gcs saved")
                # except Exception as e:
                #     print(e)
            if EMBED_QC:


                qc_embed(name=out_path)


