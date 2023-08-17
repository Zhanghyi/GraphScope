#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2020 Alibaba Group Holding Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import base64
import json
import logging
import os.path as osp
import sys

import graphscope.learning.graphlearn_torch as glt
import torch

logger = logging.getLogger("graphscope")
logger.setLevel(logging.INFO)

log_file = "/root/examples/gs.log"
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


def decode_arg(arg):
    if isinstance(arg, dict):
        return arg
    return json.loads(
        base64.b64decode(arg.encode("utf-8", errors="ignore")).decode(
            "utf-8", errors="ignore"
        )
    )


def run_server_proc(proc_rank, handle, config, server_rank, dataset):
    glt.distributed.init_server(
        num_servers=handle["num_servers"],
        num_clients=handle["num_clients"],
        server_rank=server_rank,
        dataset=dataset,
        master_addr=handle["master_addr"],
        master_port=handle["server_client_master_port"],
        num_rpc_threads=16,
        # server_group_name="dist_train_supervised_sage_server",
    )

    logger.info(f"-- [Server {server_rank}] Waiting for exit ...")
    glt.distributed.wait_and_shutdown_server()
    logger.info(f"-- [Server {server_rank}] Exited ...")


def launch_graphlearn_torch_server(handle, config, server_rank):
    # TODO(hongyi): hard code arxiv for test now
    dataset_name = "ogbn-arxiv"
    dataset_root_dir = "/root/arxiv"
    root_dir = osp.join(osp.dirname(osp.realpath(__file__)), dataset_root_dir)
    dataset = glt.distributed.DistDataset()
    dataset.load(
        root_dir=osp.join(root_dir, f"{dataset_name}-partitions"),
        partition_idx=0,
        graph_mode="CPU",
        feature_with_gpu=False,
        whole_node_label_file=osp.join(root_dir, f"{dataset_name}-label", "label.pt"),
    )

    logger.info(f"-- [Server {server_rank}] Initializing server ...")

    torch.multiprocessing.spawn(
        fn=run_server_proc, args=(handle, config, server_rank, dataset), nprocs=1
    )


if __name__ == "__main__":
    if len(sys.argv) < 3:
        logger.info(
            "Usage: ./launch_graphlearn_torch.py <handle> <config> <server_index>",
            file=sys.stderr,
        )
        sys.exit(-1)

    handle = decode_arg(sys.argv[1])
    config = decode_arg(sys.argv[2])
    server_index = int(sys.argv[3])
    launch_graphlearn_torch_server(handle, config, server_index)
