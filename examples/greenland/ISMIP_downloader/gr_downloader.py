#!/usr/bin/env python3

import os
from pathlib import Path

import yaml
from globus_sdk import (
    TransferClient,
    TransferData,
    NativeAppAuthClient,
    AccessTokenAuthorizer,
)

# =========================
# LOAD CONFIG
# =========================

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


# =========================
# GET GLOBUS CLIENT
# =========================
from globus_sdk import (
    NativeAppAuthClient,
    TransferClient,
    RefreshTokenAuthorizer,
)

CLIENT_ID = "8b26e77b-f743-4ebf-ab2f-a8a15ed853a4"


from globus_sdk import NativeAppAuthClient, TransferClient
from globus_sdk import RefreshTokenAuthorizer, AccessTokenAuthorizer
import time

def get_transfer_client():
    client = NativeAppAuthClient(client_id=CLIENT_ID)

    client.oauth2_start_flow(
        requested_scopes="urn:globus:auth:scope:transfer.api.globus.org:all",
        refresh_tokens=True,
    )

    print("Login here:\n", client.oauth2_get_authorize_url())

    auth_code = input("Enter auth code: ").strip()

    tokens = client.oauth2_exchange_code_for_tokens(auth_code)

    transfer_tokens = tokens.by_resource_server["transfer.api.globus.org"]

    access_token = transfer_tokens["access_token"]

    # 🔴 safest fallback: do NOT use refresh unless guaranteed
    authorizer = AccessTokenAuthorizer(access_token)

    return TransferClient(authorizer=authorizer)
# =========================
# BUILD FILE LIST
# =========================

def build_files(dataset, scenario, start_year, end_year):
    files = []

    for year in range(start_year, end_year + 1):

        fname = (
            f"{dataset}_GrIS_CESM2-WACCM_"
            f"{scenario}_dEBM2-1000m_v1_{year}.nc"
        )

        files.append(fname)

    return files


# =========================
# SUBMIT TRANSFER
# =========================

def submit_transfer(
    tc,
    source_ep,
    dest_ep,
    source_base,
    dataset_root,
    dataset,
    scenario,
    files,
):

    print("\n" + "=" * 70)
    print(f"[TRANSFER] Dataset : {dataset}")
    print(f"[TRANSFER] Scenario: {scenario}")
    print(f"[FILES] {len(files)}")
    print("=" * 70)

    tdata = TransferData(
        source_ep,
        dest_ep,
        label=f"ISMIP7 {dataset} {scenario}",
    )

    remote_base = (
        f"{source_base}/{scenario}/dEBM2-1000m/{dataset}/v1/"
    )

    local_base = (
        dataset_root
        / dataset
        / scenario
    )

    local_base.mkdir(
        parents=True,
        exist_ok=True,
    )

    for fname in files:

        source_file = f"{remote_base}{fname}"

        dest_file = str(local_base / fname)

        tdata.add_item(
            source_file,
            dest_file,
        )

    result = tc.submit_transfer(tdata)

    task_id = result["task_id"]

    print(f"[OK] Submitted task: {task_id}")

    return task_id


# =========================
# MAIN
# =========================

def main():

    print("[INFO] Loading config.yaml")

    config = load_config()

    source_ep = config["source"]["endpoint"]
    source_base = config["source"]["base_path"]

    dataset_root = Path(
        config["destination"]["dataset_root"]
    ).expanduser()

    tc = get_transfer_client()
    dest_ep = config["destination"]["endpoint"]
    print(f"[INFO] Source endpoint : {source_ep}")
    print(f"[INFO] Destination endpoint : {dest_ep}")
    print(f"[INFO] Dataset root : {dataset_root}")

    for dataset, dataset_cfg in config["datasets"].items():

        scenarios = dataset_cfg["scenarios"]

        for scenario, scenario_cfg in scenarios.items():

            start_year = scenario_cfg["start"]
            end_year = scenario_cfg["end"]

            files = build_files(
                dataset,
                scenario,
                start_year,
                end_year,
            )

            submit_transfer(
                tc,
                source_ep,
                dest_ep,
                source_base,
                dataset_root,
                dataset,
                scenario,
                files,
            )


if __name__ == "__main__":
    main()



