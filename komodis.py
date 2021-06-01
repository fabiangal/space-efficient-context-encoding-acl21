"""

"""
import json
import copy
import numpy as np
from numpy import random
from itertools import chain

import torch
from torch.utils.data import DataLoader, TensorDataset

SPECIAL_TOKENS = ["<SST>", "<END>", "<PAD>", "<SPK:S>", "<SPK:O>"]

ATTR_TO_SPECIAL_TOKENS = {"bos_token": "<SST>", "eos_token": "<END>", "pad_token": "<PAD>",
                          "additional_special_tokens": ["<SPK:S>", "<SPK:O>"]}

KG_ENCODING_MAPPING = {
    "movie": "<DEL:MOVIE>",
    "actor": "<DEL:ACTOR0>",
    "person": "<DEL:ACTOR1>",
    "writer": "<DEL:WRITER>",
    "director": "<DEL:DIRECTOR>",
    "role": "<FACT:ACTOR0>",
    "age restriction": "<DEL:CERTIFICATE>",
    "certificate": "<DEL:CERTIFICATE>",
    "budget": "<DEL:BUDGET>",
    "shot location": "<DEL:COUNTRY>",
    "release year": "<DEL:YEAR>",
    "genre": "<DEL:GENRE0>",
    "plot": "<FACT:PLOT>",
    "trivia": "<FACT:OBJECT>",
    "attitude": "<OPINION:MOVIE>",
    "random_attitude": "<OPINION:MOVIE>"
}


class Komodis:
    def __init__(self, tokenizer):
        """
        Args:
            tokenizer   A transformers tokenizer object.
        """
        self.tokenizer = tokenizer
        self.kg_encoding_mapping = {}
        for key, value in KG_ENCODING_MAPPING.items():
            value_id = self.tokenizer.convert_tokens_to_ids(value)
            self.kg_encoding_mapping[key] = value_id
        self._dataset = None

        self.spk_s, self.spk_o = self.tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[3:5])

    @staticmethod
    def _convert_dialogue_to_samples(dialogue, history_length, max_length):
        """ Converts a dialogue into multiple samples.

        Args:
            dialogue        A dict. A KOMODIS dialogue.
            history_length  An integer. Maximum number of previous utterances for a sample.
            max_length      An integer. Maximum number of tokens per sample.

        """
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

                # check if sequence length fits max_length (if so, we don't need to shorten the history)
                if (len_hist + len_label + len_context + num_special_tokens) <= max_length:
                    break

                # this will remove the left most utterance from the history, to shorten the sequence
                t += 1

                # if the sequence is too long even if there is only one additional utterance, it must be skipped
                if lower + t == num + 1:
                    skip = True
                    break

            # --- knowledge graph encoding ---
            nodes_shuffled, indices = Komodis._shuffle_nodes(dialogue["sequence"][speaker])
            node_lengths = [len(x[0]) for x in nodes_shuffled]
            matrix_shuffled = Komodis._shuffle_matrix(dialogue["attn_matrix"][speaker], indices)
            kg_nodes = nodes_shuffled
            kg_attn_matrix = Komodis._expand_matrix(matrix_shuffled, node_lengths)

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

    def get_torch_features(self, split, batch_size):
        """ Returns a torch dataset object for training.

        Args:
            split       A string. One of: train, valid, test.
            batch_size  The batch size of the data object.

        """
        samples = []
        for dialogue in self._dataset[split]:
            samples += Komodis._convert_dialogue_to_samples(dialogue, history_length=3, max_length=256)

        features = {
            "input_ids": [],
            "token_type_ids": [],
            "kg_attn_matrix": [],
            "lm_labels": []
        }

        for sample in samples:
            seqs = self._convert_sample_to_sequences(history=sample["history"],
                                                     reply=sample["label"],
                                                     kg_nodes=sample["kg_sequence"],
                                                     kg_attn_matrix=sample["kg_attn_matrix"])

            features["input_ids"].append(seqs["input_ids"])
            features["token_type_ids"].append(seqs["token_type_ids"])
            features["kg_attn_matrix"].append(seqs["kg_attn_matrix"])
            features["lm_labels"].append(seqs["lm_labels"])

        features_padded = Komodis._pad_features(features=features, padding=self.tokenizer.pad_token_id)

        torch_features = []
        for key, value in features_padded.items():
            torch_features.append(torch.tensor(value))
        dataset = TensorDataset(*torch_features)

        loader = DataLoader(dataset, sampler=None, batch_size=batch_size, shuffle=True, drop_last=True)

        return loader

    def process_subgraph(self, subgraph, encoding, inference=False, max_clen=-1):
        """ Processes one subgraph into sequences.

        Args:
             subgraph   A dict. A KOMODIS subgraph.
             encoding   A string. One of: series, parallel.
             inference  A boolean. If True, the function don't require first- and second-speaker separation.
             max_clen   An integer. The maximum number of tokens for the subgraph encoding.
        """

        def process_series():
            """ Uses the specific encoding for komodis, where relations are not explicitly encoded.
            kg encoding (1) in context_length_evaluation.xlsx
            """
            ss = self.tokenizer.encode(node["content"])
            subgraph_sequence[speaker].append((
                ss,
                len(ss) * [self.kg_encoding_mapping[node["type"]]],
                node["content"],
                node["type"]
            ))

        def process_parallel():
            """ Converts each node into a type-content sequence, where the type is added on the token-type dimension.
            kg encoding (4) in context_length_evaluation.xlsx
            """
            c_node = self.tokenizer.encode(node["content"])
            c_type = self.tokenizer.encode(node["type"])
            node_length = max(len(c_node), len(c_type))

            c_node = c_node + [self.tokenizer.pad_token_id] * (node_length - len(c_node))
            c_type = c_type + [self.tokenizer.pad_token_id] * (node_length - len(c_type))

            subgraph_sequence[speaker].append((
                c_node,
                c_type,
                node["content"],
                node["type"]
            ))

        def sequence_length(seq):
            """ Returns the length of the whole sequence. """
            full_length = 0
            for item in seq:
                full_length += len(item[0])
            return full_length

        def shorten_context(sequence, matrix):
            """ Shortens the subgraph by removing one node and corresponding edges. """
            while sequence_length(sequence) > max_clen:

                # estimate a random node
                remove_candidates = []
                for idx, item in enumerate(sequence):
                    if item[3] not in ["movie", "attitude"]:
                        remove_candidates.append(idx)
                remove_candidate = random.choice(remove_candidates)

                # delete node in matrix
                matrix = np.delete(matrix, remove_candidate, axis=0)
                matrix = np.delete(matrix, remove_candidate, axis=1)

                # delete node in sequence
                sequence.pop(remove_candidate)

                # look for edges that belong to the node
                for idx, item in enumerate(sequence):
                    if item[3] == "attitude":
                        att_rels = 0
                        for att_rel in matrix[idx]:
                            att_rels += att_rel
                        if att_rels == 1:
                            # delete them from the matrix
                            matrix = np.delete(matrix, idx, axis=0)
                            matrix = np.delete(matrix, idx, axis=1)
                            # delete them from the sequence
                            sequence.pop(idx)
                            break

            return sequence, matrix

        if inference:
            subgraph_sequence = {"inference": []}
            subgraph_attn_mask = {"inference": []}
            speakers = ["inference"]
        else:
            subgraph_sequence = {"first_speaker": [], "second_speaker": []}
            subgraph_attn_mask = {"first_speaker": [], "second_speaker": []}
            speakers = ["first_speaker", "second_speaker"]

        for speaker in speakers:
            for node in subgraph[speaker]["nodes"]:
                if str(node["content"]) == "-1" and node["type"] == "age restriction":
                    node["content"] = "unknown"
                if encoding == "series":
                    process_series()
                elif encoding == "parallel":
                    process_parallel()
                else:
                    raise ValueError("Could not find a kg_encoding_type "
                                     "with name: {}".format(encoding))
            subgraph_attn_mask[speaker] = np.array(json.loads(subgraph[speaker]["matrix"]))

        # make knowledge graphs shorter, if needed
        if max_clen > -1:
            for speaker in speakers:
                subgraph_sequence[speaker], subgraph_attn_mask[speaker] = \
                    shorten_context(subgraph_sequence[speaker], subgraph_attn_mask[speaker])

        return {"sequence": subgraph_sequence, "attn_matrix": subgraph_attn_mask}

    def _convert_sample_to_sequences(self, history, reply, kg_nodes, kg_attn_matrix):
        """ Converts one sample into sequences for GPT-2 training. """

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

    @staticmethod
    def _pad_features(features, padding):
        """ Pads the features to it's maximum sequence with a given padding token. """

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
    def _replace_special_moviecorpus_tokens(dialogue):
        """ Replaces [eou] tokens and add [end] tokens.
        """
        new_dialogue = []
        for utterance in dialogue:
            tokens = utterance.split(" ")
            new_tokens = []
            for i in range(len(tokens)):
                if i == 0:
                    new_tokens.append(tokens[i])
                else:
                    if tokens[i] in ["[eou]", "[EOU]"]:
                        if tokens[i - 1] in ["?", ".", ",", "!", ";", ":"]:
                            continue
                        else:
                            new_tokens.append(".")
                    else:
                        new_tokens.append(tokens[i])
            new_dialogue.append(" ".join(new_tokens))
        return new_dialogue

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
