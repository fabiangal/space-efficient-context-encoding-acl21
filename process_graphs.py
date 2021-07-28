""" process graphs

This script is supposed to show how the authors encoded the knowledge graphs.

Processed data is stored in ./data/processed/ for each set of (dataset, depth, encoding).

"""
import os
import sys
import json
import pickle
import transformers
from argparse import ArgumentParser

from komodis import Komodis
from opendialkg import OpenDialKG


parser = ArgumentParser()
parser.add_argument("--dataset", type=str, help="Name of the dataset (komodis or opendialkg).")
parser.add_argument("--depth", type=int, help="Graph depth (0, 1 or 2). See paper for more information.")
parser.add_argument("--encoding", type=str, help="Encoding type (series or parallel) See paper for more information.")
parser.add_argument("--lr", type=float, default=6.0e-5, help="Learning rate for training.")
parser.add_argument("--epochs", type=int, default=3, help="Number of epochs for training.")
parser.add_argument("--batch_size", type=int, default=4, help="Train and valid batch size for training.")
parser.add_argument("--device", type=str, default="cpu", help="Only cpu support in with this repository.")
args = parser.parse_args()


def run():
    if args.dataset not in ["komodis", "opendialkg"]:
        print("Argument dataset={} is not valid!".format(args.dataset))
        sys.exit()

    if args.depth not in [0, 1, 2]:
        print("Argument depth={} is not valid!".format(args.dataset))
        sys.exit()

    if args.encoding not in ["series", "parallel"]:
        print("Argument encoding={} is not valid!".format(args.dataset))
        sys.exit()

    # read knowledge graphs
    graphs_raw = {}
    for split in ["train", "valid", "test"]:
        file_path = "data/knowledge_graphs/{}_graphs_d{}_{}.json".format(args.dataset, args.depth, split)

        if not os.path.isfile(file_path):
            print("{} does not exist. Please unzip data first or ask authors for graphs with depth > 1.")
            sys.exit()

        graphs_raw[split] = json.load(open(file_path, "r"))

    # process graphs
    tokenizer = transformers.GPT2Tokenizer.from_pretrained("gpt2")

    if args.dataset == "komodis":
        dataset_helper = Komodis(tokenizer=tokenizer)
    elif args.dataset == "opendialkg":
        dataset_helper = OpenDialKG(tokenizer=tokenizer)
    else:
        print("Argument dataset={} is not valid!".format(args.dataset))
        sys.exit()

    graphs_processed = {}
    for split in ["train", "valid", "test"]:
        graphs_processed[split] = {k: dataset_helper.process_subgraph(subgraph=v, encoding=args.encoding)
                                   for k, v in graphs_raw[split].items()}

    # save graphs
    with open("data/processed/{}_graphs_d{}_{}-enc.pkl".format(args.dataset, args.depth, args.encoding), "wb") as f:
        pickle.dump(graphs_processed, f)


if __name__ == "__main__":
    run()
