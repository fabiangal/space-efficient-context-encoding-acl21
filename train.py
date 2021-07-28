""" train

This script is supposed to show how the authors trained a GPT-2 model.

Please use transformers 4.5.1 to train the model. Other versions might work as well.

To feed the knowledge graph edges and nodes into the transformer model, some changes of the model were needed. They
can be found with full explanation in model.py

"""

import os
import sys
import json
import math
import pickle

from pprint import pformat
from datetime import datetime

import torch

import transformers
from transformers import AdamW
from argparse import ArgumentParser

from komodis import Komodis
from opendialkg import OpenDialKG
from model import GPT2LMHeadModel

from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from ignite.contrib.handlers import ProgressBar
from ignite.contrib.handlers import PiecewiseLinear
from ignite.metrics import Loss, MetricsLambda, RunningAverage


# --- argument parsing ------------------------------------------------------------------------------------------------
parser = ArgumentParser()
parser.add_argument("--dataset", type=str, help="Name of the dataset (komodis or opendialkg).")
parser.add_argument("--depth", type=int, help="Graph depth (0, 1 or 2). See paper for more information.")
parser.add_argument("--encoding", type=str, help="Encoding type (series or parallel) See paper for more information.")
parser.add_argument("--lr", type=float, default=6.0e-5, help="Learning rate for training.")
parser.add_argument("--epochs", type=int, default=3, help="Number of epochs for training.")
parser.add_argument("--batch_size", type=int, default=4, help="Train and valid batch size for training.")
parser.add_argument("--device", type=str, default="cpu", help="Only cpu support in with this repository.")
args = parser.parse_args()

if args.dataset not in ["komodis", "opendialkg"]:
    print("Argument dataset={} is not valid!".format(args.dataset))
    sys.exit()

if args.depth not in [0, 1, 2]:
    print("Argument depth={} is not valid!".format(args.dataset))
    sys.exit()

if args.encoding not in ["series", "parallel"]:
    print("Argument encoding={} is not valid!".format(args.dataset))
    sys.exit()

# --- load data -------------------------------------------------------------------------------------------------------
if not os.path.exists("data/datasets/"):
    print("Please create a datasets directory in ./data, download and unpack the datasets.")
    sys.exit()

if args.dataset == "komodis":
    # --- open original dataset ---
    dataset = {}
    for split in ["train", "valid", "test"]:
        if not os.path.isfile("data/datasets/komodis/komodis_dialogues_{}.json".format(split)):
            print("Please unpack the komodis dataset in './data/datasets/'.")
            sys.exit()
        with open("data/datasets/komodis/komodis_dialogues_{}.json".format(split), "r") as f:
            dataset[split] = json.load(f)
elif args.dataset == "opendialkg":
    # --- open original dataset ---
    dataset = {}
    for split in ["train", "valid", "test"]:
        if not os.path.isfile("data/datasets/opendialkg/{}_opendialkg.json".format(split)):
            print("Please unpack the opendialkg dataset in './data/datasets/'.")
            sys.exit()
        with open("data/datasets/opendialkg/{}_opendialkg.json".format(split), "r") as f:
            dataset[split] = json.load(f)
else:
    print("Argument dataset={} is not valid!".format(args.dataset))
    sys.exit()

# --- create and/or open knowledge graphs ---
file = "data/processed/{}_graphs_d{}_{}-enc.pkl".format(args.dataset, args.depth, args.encoding)
if not os.path.isfile(file):
    import process_graphs
    exec(open("process_graphs.py").read())

with open(file, "rb") as f:
    graphs = pickle.load(f)

# --- prepare training ------------------------------------------------------------------------------------------------
tokenizer = transformers.GPT2Tokenizer.from_pretrained("gpt2")
if args.dataset == "komodis":
    dataset_helper = Komodis(tokenizer=tokenizer)
elif args.dataset == "opendialkg":
    dataset_helper = OpenDialKG(tokenizer=tokenizer)
else:
    print("Argument dataset={} is not valid!".format(args.dataset))
    sys.exit()

# data preparation: see komodis.py or opendialkg.py for explanations
dataset_helper.prepare_dataset(dataset=dataset, graphs=graphs)
train_loader = dataset_helper.get_torch_features(split="train", batch_size=args.batch_size)
valid_loader = dataset_helper.get_torch_features(split="valid", batch_size=args.batch_size)

# loads the pretrained gpt-2 model weights and adapts the embedding matrix to the new (bigger) vocab size
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.resize_token_embeddings(len(dataset_helper.tokenizer.encoder) + dataset_helper.num_added_tokens)

# simple optimizer and learning rate schedule initialization. Used like this in our experiments.
optimizer = AdamW(model.parameters(), lr=args.lr, correct_bias=True)
scheduler = PiecewiseLinear(optimizer, "lr", [(0, args.lr), (args.epochs * len(train_loader), 0.0)])


def average_distributed_scalar(scalar, targs):
    """ Average a scalar over the nodes if we are in distributed training. We use this for distributed evaluation. """
    if targs.local_rank == -1:
        return scalar
    scalar_t = torch.tensor(scalar, dtype=torch.float, device=targs.device) / torch.distributed.get_world_size()
    torch.distributed.all_reduce(scalar_t, op=torch.distributed.ReduceOp.SUM)
    return scalar_t.item()


# metrics initialization. ignore_index is set to -100 and is used for masking out tokens in the label, that should not
# contribute to the loss. This needs to be aligned with the data preprocessing.
metrics = {
    "nll": Loss(torch.nn.CrossEntropyLoss(ignore_index=-100), output_transform=lambda x: (x[0], x[1]))
}
metrics["average_nll"] = MetricsLambda(average_distributed_scalar, metrics["nll"], args)
metrics["average_ppl"] = MetricsLambda(math.exp, metrics["average_nll"])


# --- update and inference functions ----------------------------------------------------------------------------------
def update(engine, batch):
    """ Pytorch update function.

    Decoding one batch needs to be aligned with the get_torch_features processing function!
    """
    model.train()

    batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
    input_ids, token_type_ids, kg_attn_matrix, lm_labels = batch

    output = model(
        input_ids, token_type_ids=token_type_ids,
        labels=lm_labels, kg_masks=kg_attn_matrix
    )

    loss = output["loss"]
    loss.sum().backward()

    optimizer.step()
    optimizer.zero_grad()

    return loss.mean().item()


def inference(engine, batch):
    """ Pytorch inference function.

    """
    model.eval()

    batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
    input_ids, token_type_ids, kg_attn_matrix, lm_labels = batch

    output = model(
        input_ids, token_type_ids=token_type_ids, kg_masks=kg_attn_matrix
    )

    lm_logits_flat_shifted = output["logits"][..., :-1, :].contiguous().view(-1, output["logits"].size(-1))
    lm_labels_flat_shifted = lm_labels[..., 1:].contiguous().view(-1)

    return lm_logits_flat_shifted, lm_labels_flat_shifted
# ---------------------------------------------------------------------------------------------------------------------


# prepare training with the Ignite package
trainer = Engine(update)
evaluator = Engine(inference)

trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda _: evaluator.run(valid_loader))
trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)
RunningAverage(output_transform=lambda x: x).attach(trainer, "loss")
for name, metric in metrics.items():
    metric.attach(evaluator, name)

validation_status = {}
pbar = ProgressBar(persist=True)
pbar.attach(trainer, metric_names=["loss"])

evaluator.add_event_handler(Events.COMPLETED,
                            lambda _: pbar.log_message("Validation: %s" % pformat(evaluator.state.metrics)))

current_time = datetime.now().strftime('%b%d_%H-%M-%S')
log_dir = os.path.join('runs', current_time + '_gpt2')

checkpoint_handler = ModelCheckpoint(log_dir, "checkpoint", n_saved=3)

trainer.add_event_handler(Events.EPOCH_COMPLETED,
                          checkpoint_handler,
                          {"mymodel": getattr(model, "module", model)})

torch.save(args, log_dir + "/model_training_args.bin")

model.to(args.device)

# Ignite training
trainer.run(train_loader, max_epochs=args.epochs)
