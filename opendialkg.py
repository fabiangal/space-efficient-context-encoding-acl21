"""

"""
import json
import copy
import numpy as np
from numpy import random
from itertools import chain

import torch
from torch.utils.data import DataLoader, TensorDataset


PRE_TRAINED_SPECIAL_TOKENS = ["<SST>", "<END>", "<PAD>"]
ADDITIONAL_SPECIAL_TOKENS = ["<SPK:S>", "<SPK:O>", "<KG:NODE>", "<KG:EDGE>"]
SPECIAL_TOKENS = PRE_TRAINED_SPECIAL_TOKENS + ADDITIONAL_SPECIAL_TOKENS
ATTR_TO_SPECIAL_TOKENS = {"bos_token": PRE_TRAINED_SPECIAL_TOKENS[0],
                          "eos_token": PRE_TRAINED_SPECIAL_TOKENS[1],
                          "pad_token": PRE_TRAINED_SPECIAL_TOKENS[2],
                          "additional_special_tokens": ADDITIONAL_SPECIAL_TOKENS}


class OpenDialKG:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.num_added_tokens = self.tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKENS)
        self.spk_s, self.spk_o = self.tokenizer.convert_tokens_to_ids(ADDITIONAL_SPECIAL_TOKENS[0:2])
        self.kg_node, self.kg_edge = self.tokenizer.convert_tokens_to_ids(ADDITIONAL_SPECIAL_TOKENS[2:4])

        self._dataset = None

    def prepare_dataset(self, dataset, graphs):
        """ """
        self._dataset = {}
        for split in ["train", "valid", "test"]:
            self._dataset[split] = []
            for item in dataset[split]:
                graph = graphs[split][str(item["id"])]
                if len(graph["sequence"]["first_speaker"]) == 0 or len(graph["sequence"]["second_speaker"]) == 0:
                    continue
                dialogue = [d["message"].lower() for d in item["messages"] if "message" in d]
                dialogue = [self.tokenizer.encode(d) for d in dialogue]
                self._dataset[split].append({
                    "dialogue": dialogue,
                    "sequence": graph["sequence"],
                    "attn_matrix": graph["attn_matrix"]
                })

    def get_torch_features(self, split, batch_size):
        """ """
        samples = []
        for dialogue in self._dataset[split]:
            samples += self._convert_dialogue_to_samples(dialogue, history_length=3, max_length=256)

        features = {
            "input_ids": [],
            "token_type_ids": [],
            "kg_attn_matrix": [],
            "lm_labels": []
        }

        for sample in samples:
            seqs = self._convert_sample_to_sequences(history=sample["history"],
                                                     reply=sample["label"][0],
                                                     kg_nodes=sample["kg_sequence"],
                                                     kg_attn_matrix=sample["kg_attn_matrix"])

            features["input_ids"].append(seqs["input_ids"])
            features["token_type_ids"].append(seqs["token_type_ids"])
            features["kg_attn_matrix"].append(seqs["kg_attn_matrix"])
            features["lm_labels"].append(seqs["lm_labels"])

        features_padded = OpenDialKG._pad_features(features=features, padding=self.tokenizer.pad_token_id)

        torch_features = []
        for key, value in features_padded.items():
            torch_features.append(torch.tensor(value))
        dataset = TensorDataset(*torch_features)

        loader = DataLoader(dataset, sampler=None, batch_size=batch_size, shuffle=True, drop_last=True)

        return loader

    def _convert_dialogue_to_samples(self, dialogue, history_length, max_length):
        """ """
        samples = []

        for num in range(len(dialogue["dialogue"]) - 1):
            # determine which speaker is system for current sample
            if num % 2 == 0:
                speaker = "second_speaker"
            else:
                speaker = "first_speaker"

            # number of previous utterances
            lower = num + 1 - history_length
            if lower < 0:
                lower = 0

            # check for max length
            t = 0
            skip = False
            len_context = len(list(chain(*[x[0] for x in dialogue["sequence"][speaker]])))
            while True:
                len_hist = len(list(chain(*dialogue["dialogue"][lower + t:num + 1])))
                len_label = len(dialogue["dialogue"][num + 1])

                # 3 tokens:              start token, end token, token for reply
                # (num + 1 - lower -t):  plus one token per utterance in the history
                num_special_tokens = 3 + (num + 1 - lower - t)

                if (len_hist + len_label + len_context + num_special_tokens) <= max_length:
                    break

                t += 1

                if lower + t == num + 1:
                    skip = True
                    break

            # --- knowledge graph encoding ---
            nodes_shuffled, indices = OpenDialKG._shuffle_nodes(dialogue["sequence"][speaker])
            node_lengths = [len(x[0]) for x in nodes_shuffled]
            matrix_shuffled = OpenDialKG._shuffle_matrix(dialogue["attn_matrix"][speaker], indices)
            kg_nodes = nodes_shuffled
            kg_attn_matrix = OpenDialKG._expand_matrix(matrix_shuffled, node_lengths)

            if not skip:
                samples.append({
                    "label": [dialogue["dialogue"][num + 1]],
                    "history": dialogue["dialogue"][lower + t:num + 1],
                    "kg_sequence": kg_nodes,
                    "kg_attn_matrix": kg_attn_matrix,
                    "len_hist": len_hist,
                    "len_context": len_context
                })

        return samples

    def _convert_sample_to_sequences(self, history, reply, kg_nodes, kg_attn_matrix):
        """ """
        context_input_ids = list(chain(*[x[0] for x in kg_nodes]))
        context_token_type_ids = list(chain(*[x[1] for x in kg_nodes]))

        hist_length = len(history)
        if hist_length % 2 == 0:
            first_utt_type = self.spk_s
            second_utt_type = self.spk_o
        else:
            first_utt_type = self.spk_o
            second_utt_type = self.spk_s

        sequence = copy.deepcopy([[self.tokenizer.bos_token_id] + context_input_ids] + history + [reply])

        sequence[-1] += [self.tokenizer.eos_token_id]
        sequence = [sequence[0]] + [[second_utt_type if i % 2
                                     else first_utt_type] + s
                                    for i, s in enumerate(sequence[1:])]

        seqs = {
            "input_ids": list(chain(*sequence)),
            "lm_labels": ([-100] * sum(len(s) for s in sequence[:-1])) + [-100] + sequence[-1][1:],
            "kg_attn_matrix": kg_attn_matrix
        }

        def cond(i):
            if i % 2:
                return second_utt_type
            return first_utt_type

        seqs["token_type_ids"] = [self.tokenizer.bos_token_id] + context_token_type_ids + \
                                 [cond(i) for i, s in enumerate(sequence[1:]) for _ in s]

        return seqs

    def process_subgraph(self, subgraph, encoding, max_clen=-1):
        """ """
        def process_series():
            """ Converts each node into a type-content sequence to explicitly encode the relations.
            kg encoding (3) in context_length_evaluation.xlsx
            """
            c_node = self.tokenizer.encode(node["content"].lower())
            c_edge = self.tokenizer.encode(node["type"].replace("_", " ").lower())
            sub_graph_sequence[speaker].append((
                c_edge + c_node,
                len(c_edge) * [self.kg_edge] +
                len(c_node) * [self.kg_node],
                node["content"].lower(),
                node["type"].replace("_", " ").lower()
            ))

        def process_parallel():
            """ Converts each node into a type-content sequence, where the type is added on the token-type dimension.
            kg encoding (4) in context_length_evaluation.xlsx
            """
            c_node = self.tokenizer.encode(node["content"].lower())
            c_type = self.tokenizer.encode(node["type"].replace("_", " ").lower())
            node_length = max(len(c_node), len(c_type))

            c_node = c_node + [self.tokenizer.pad_token_id] * (node_length - len(c_node))
            c_type = c_type + [self.tokenizer.pad_token_id] * (node_length - len(c_type))

            sub_graph_sequence[speaker].append((
                c_node,
                c_type,
                node["content"].lower(),
                node["type"].replace("_", " ").lower()
            ))

        def sequence_length(seq):
            """ Returns the length of the whole sequence. """
            full_length = 0
            for item in seq:
                full_length += len(item[0])
            return full_length

        def shorten_context(sequence, matrix):
            """ """
            while sequence_length(sequence) > max_clen:
                remove_candidate = np.random.choice(range(len(sequence)))
                matrix = np.delete(matrix, remove_candidate, axis=0)
                matrix = np.delete(matrix, remove_candidate, axis=1)
                sequence.pop(remove_candidate)

            return sequence, matrix

        sub_graph_sequence = {"first_speaker": [], "second_speaker": []}
        sub_graph_attn_matrix = {"first_speaker": [], "second_speaker": []}

        for speaker in ["first_speaker", "second_speaker"]:
            for node in subgraph[speaker]["nodes"]:
                if encoding == "series":
                    process_series()
                elif encoding == "parallel":
                    process_parallel()
                else:
                    raise ValueError("Could not find a kg_encoding_type "
                                     "with name: {}".format(encoding))
            sub_graph_attn_matrix[speaker] = np.array(json.loads(subgraph[speaker]["matrix"]))

        # make knowledge graphs shorter, if needed
        if max_clen > -1:
            for speaker in ["first_speaker", "second_speaker"]:
                sub_graph_sequence[speaker], sub_graph_attn_matrix[speaker] = \
                    shorten_context(sub_graph_sequence[speaker], sub_graph_attn_matrix[speaker])

        return {"sequence": sub_graph_sequence, "attn_matrix": sub_graph_attn_matrix}

    @staticmethod
    def _pad_features(features, padding):
        """ """
        keys = ["input_ids", "token_type_ids", "lm_labels"]
        max_l = max(len(feature) for feature in features["input_ids"])
        for name in keys:
            features[name] = [x + [padding if name != "lm_labels" else -100] * (max_l - len(x)) for x in
                              features[name]]

        max_l = max(m.shape[0] for m in features["kg_attn_matrix"])
        for num, matrix in enumerate(features["kg_attn_matrix"]):
            back = np.tril(max_l * [1])
            d1, d2 = matrix.shape
            back[:d1, :d2] = matrix
            features["kg_attn_matrix"][num] = back

        return features

    @staticmethod
    def _shuffle_nodes(nodes):
        """ Shuffles nodes and returns shuffle-indices. """
        indices = list(range(len(nodes)))
        nodes_indices = list(zip(nodes, indices))
        random.shuffle(nodes_indices)
        return zip(*nodes_indices)

    @staticmethod
    def _shuffle_matrix(matrix, indices):
        """ Shuffles an attention-matrix based on shuffled indices. """
        temp_matrix_1 = np.ndarray((matrix.shape[0], matrix.shape[1]))
        temp_matrix_2 = np.ndarray((matrix.shape[0], matrix.shape[1]))

        for n, i in enumerate(indices):
            temp_matrix_1[:, n] = matrix[:, i]
        for n, i in enumerate(indices):
            temp_matrix_2[n, :] = temp_matrix_1[i, :]

        return temp_matrix_2

    @staticmethod
    def _expand_matrix(matrix, lengths, fix_length=None):
        """ Expands the attention-matrix based on the node-lengths """
        temp_matrix_1 = np.ndarray((sum(lengths), matrix.shape[0]))
        temp_matrix_2 = np.ndarray((sum(lengths), sum(lengths)))

        curr = 0
        for n, i in enumerate(lengths):
            temp_matrix_1[curr:curr+i, :] = np.repeat([matrix[n, :]], repeats=i, axis=0)
            curr += i
        curr = 0
        for n, i in enumerate(lengths):
            temp_matrix_2[:, curr:curr+i] = np.transpose(np.repeat([temp_matrix_1[:, n]], repeats=i, axis=0))
            curr += i

        if fix_length is not None:
            background_matrix = np.zeros((fix_length, fix_length), dtype=int)
            background_matrix[0:temp_matrix_2.shape[0], 0:temp_matrix_2.shape[1]] = temp_matrix_2
            temp_matrix_2 = background_matrix

        return temp_matrix_2