import os
import pdb
import time
import torch
import ctcdecode
import numpy as np
from itertools import groupby
import torch.nn.functional as F

from .ngram_rescorer import NGramRescorer


def _collapsed_beam(beam, length, blank_id):
    values = beam[:int(length)].tolist()
    if values:
        values = [value for value, _ in groupby(values)]
    return [int(value) for value in values if int(value) != blank_id]


def _select_beam_index(
    sequences,
    ctc_scores,
    rescorer=None,
    alpha=0.0,
    beta=0.0,
):
    if rescorer is None or not sequences:
        return 0
    best_index = 0
    best_score = float("-inf")
    for index, sequence in enumerate(sequences):
        ctc_log_probability = -float(ctc_scores[index])
        score = (
            ctc_log_probability
            + float(alpha) * rescorer.score(sequence)
            + float(beta) * len(sequence)
        )
        if score > best_score:
            best_score = score
            best_index = index
    return best_index


class Decode(object):
    def __init__(
        self,
        gloss_dict,
        num_classes,
        search_mode,
        blank_id=0,
        beam_width=10,
        num_processes=10,
        rescorer_path=None,
        rescorer_alpha=0.0,
        rescorer_beta=0.0,
    ):
        self.i2g_dict = dict((v[0], k) for k, v in gloss_dict.items())
        self.g2i_dict = {v: k for k, v in self.i2g_dict.items()}
        self.num_classes = num_classes
        self.search_mode = search_mode
        self.blank_id = blank_id
        self.rescorer_alpha = float(rescorer_alpha)
        self.rescorer_beta = float(rescorer_beta)
        self.rescorer = (
            NGramRescorer.from_json(rescorer_path)
            if rescorer_path
            else None
        )
        if self.rescorer is not None and self.rescorer.vocabulary_size != num_classes:
            raise ValueError(
                "rescorer vocabulary size does not match the classifier"
            )
        vocab = [chr(x) for x in range(20000, 20000 + num_classes)]
        self.ctc_decoder = ctcdecode.CTCBeamDecoder(
            vocab,
            beam_width=int(beam_width),
            blank_id=blank_id,
            num_processes=int(num_processes),
        )

    def decode(self, nn_output, vid_lgt, batch_first=True, probs=False):
        if not batch_first:
            nn_output = nn_output.permute(1, 0, 2)
        if self.search_mode == "max":
            return self.MaxDecode(nn_output, vid_lgt)
        else:
            return self.BeamSearch(nn_output, vid_lgt, probs)

    def BeamSearch(self, nn_output, vid_lgt, probs=False):
        '''
        CTCBeamDecoder Shape:
                - Input:  nn_output (B, T, N), which should be passed through a softmax layer
                - Output: beam_resuls (B, N_beams, T), int, need to be decoded by i2g_dict
                          beam_scores (B, N_beams), p=1/np.exp(beam_score)
                          timesteps (B, N_beams)
                          out_lens (B, N_beams)
        '''
        if not probs:
            nn_output = nn_output.softmax(-1).cpu()
        vid_lgt = vid_lgt.cpu()
        beam_result, beam_scores, timesteps, out_seq_len = (
            self.ctc_decoder.decode(nn_output, vid_lgt)
        )
        ret_list = []
        for batch_idx in range(len(nn_output)):
            sequences = [
                _collapsed_beam(
                    beam_result[batch_idx][beam_idx],
                    out_seq_len[batch_idx][beam_idx],
                    self.blank_id,
                )
                for beam_idx in range(beam_result.size(1))
            ]
            selected = _select_beam_index(
                sequences,
                beam_scores[batch_idx],
                self.rescorer,
                self.rescorer_alpha,
                self.rescorer_beta,
            )
            ret_list.append([
                (self.i2g_dict[int(gloss_id)], index)
                for index, gloss_id in enumerate(sequences[selected])
            ])
        return ret_list

    def MaxDecode(self, nn_output, vid_lgt):
        index_list = torch.argmax(nn_output, axis=2)
        batchsize, lgt = index_list.shape
        ret_list = []
        for batch_idx in range(batchsize):
            group_result = [x[0] for x in groupby(index_list[batch_idx][:vid_lgt[batch_idx]])]
            filtered = [*filter(lambda x: x != self.blank_id, group_result)]
            if len(filtered) > 0:
                max_result = torch.stack(filtered)
                max_result = [x[0] for x in groupby(max_result)]
            else:
                max_result = filtered
            ret_list.append([(self.i2g_dict[int(gloss_id)], idx) for idx, gloss_id in
                             enumerate(max_result)])
        return ret_list
