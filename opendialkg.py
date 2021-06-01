"""

"""
import json
import numpy as np
from numpy import random


PRE_TRAINED_SPECIAL_TOKENS = ["<SST>", "<END>", "<PAD>"]
ADDITIONAL_SPECIAL_TOKENS = ["<SPK:S>", "<SPK:O>", "<KG:NODE>", "<KG:EDGE>"]
SPECIAL_TOKENS = PRE_TRAINED_SPECIAL_TOKENS + ADDITIONAL_SPECIAL_TOKENS
ATTR_TO_SPECIAL_TOKENS = {"bos_token": PRE_TRAINED_SPECIAL_TOKENS[0],
                          "eos_token": PRE_TRAINED_SPECIAL_TOKENS[1],
                          "pad_token": PRE_TRAINED_SPECIAL_TOKENS[2],
                          "additional_special_tokens": ADDITIONAL_SPECIAL_TOKENS}


class OpenDialKG:
    def __init__(self, tokenizer):
        """
        Args:
            tokenizer   A transformers tokenizer object.
        """
        self.tokenizer = tokenizer

        self.spk_s, self.spk_o = self.tokenizer.convert_tokens_to_ids(ADDITIONAL_SPECIAL_TOKENS[0:2])
        self.kg_node, self.kg_edge = self.tokenizer.convert_tokens_to_ids(ADDITIONAL_SPECIAL_TOKENS[2:4])

    def process_sub_graph(self, sub_graph, encoding, max_clen=-1):
        """ Processes one subgraph into sequences.

        Args:
             subgraph   A dict. A OpenDialKG subgraph.
             encoding   A string. One of: series, parallel.
             max_clen   An integer. The maximum number of tokens for the subgraph encoding.
        """
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
            for node in sub_graph[speaker]["nodes"]:
                if encoding == "series":
                    process_series()
                elif encoding == "parallel":
                    process_parallel()
                else:
                    raise ValueError("Could not find a kg_encoding_type "
                                     "with name: {}".format(encoding))
            sub_graph_attn_matrix[speaker] = np.array(json.loads(sub_graph[speaker]["matrix"]))

        # make knowledge graphs shorter, if needed
        if max_clen > -1:
            for speaker in ["first_speaker", "second_speaker"]:
                sub_graph_sequence[speaker], sub_graph_attn_matrix[speaker] = \
                    shorten_context(sub_graph_sequence[speaker], sub_graph_attn_matrix[speaker])

        return {"sequence": sub_graph_sequence, "attn_matrix": sub_graph_attn_matrix}