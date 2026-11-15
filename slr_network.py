import pdb
import copy
import utils
import torch
import types
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from modules.criterions import SeqKD
from modules.weighted_ctc import weighted_emission_log_probs
from modules import BiLSTMLayer, TemporalConv


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class NormLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(NormLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_dim, out_dim))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, x):
        outputs = torch.matmul(x, F.normalize(self.weight, dim=0))
        return outputs


def finite_ctc_mean(losses, branch_name):
    losses = torch.as_tensor(losses)
    valid = torch.isfinite(losses)
    if not torch.any(valid):
        raise FloatingPointError(
            "{} produced no finite CTC losses".format(branch_name)
        )
    return losses[valid].mean(), valid.detach()


class SLRModel(nn.Module):
    def __init__(
            self, num_classes, c2d_type, conv_type, use_bn=False,
            hidden_size=1024, gloss_dict=None, loss_weights=None,
            weight_norm=True, share_classifier=True,
            beam_width=10, num_processes=10, rescorer_path=None,
            rescorer_alpha=0.0, rescorer_beta=0.0
    ):
        super(SLRModel, self).__init__()
        self.decoder = None
        self.loss = dict()
        self.criterion_init()
        self.num_classes = num_classes
        self.loss_weights = loss_weights
        self.conv2d = getattr(models, c2d_type)(pretrained=True)
        self.conv2d.fc = Identity()
        self.conv1d = TemporalConv(input_size=512,
                                   hidden_size=hidden_size,
                                   conv_type=conv_type,
                                   use_bn=use_bn,
                                   num_classes=num_classes)
        self.decoder = utils.Decode(
            gloss_dict,
            num_classes,
            'beam',
            beam_width=beam_width,
            num_processes=num_processes,
            rescorer_path=rescorer_path,
            rescorer_alpha=rescorer_alpha,
            rescorer_beta=rescorer_beta,
        )
        self.temporal_model = BiLSTMLayer(rnn_type='LSTM', input_size=hidden_size, hidden_size=hidden_size,
                                          num_layers=2, bidirectional=True)
        if weight_norm:
            self.classifier = NormLinear(hidden_size, self.num_classes)
            self.conv1d.fc = NormLinear(hidden_size, self.num_classes)
        else:
            self.classifier = nn.Linear(hidden_size, self.num_classes)
            self.conv1d.fc = nn.Linear(hidden_size, self.num_classes)
        if share_classifier:
            self.conv1d.fc = self.classifier
        self.register_backward_hook(self.backward_hook)

    def backward_hook(self, module, grad_input, grad_output):
        for g in grad_input:
            g[g != g] = 0

    def masked_bn(self, inputs, len_x):
        def pad(tensor, length):
            return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).zero_()])

        x = torch.cat([inputs[len_x[0] * idx:len_x[0] * idx + lgt] for idx, lgt in enumerate(len_x)])
        x = self.conv2d(x)
        x = torch.cat([pad(x[sum(len_x[:idx]):sum(len_x[:idx + 1])], len_x[0])
                       for idx, lgt in enumerate(len_x)])
        return x

    def forward(self, x, len_x, label=None, label_lgt=None, decode=True):
        if len(x.shape) == 5:
            # videos
            batch, temp, channel, height, width = x.shape
            inputs = x.reshape(batch * temp, channel, height, width)
            framewise = self.masked_bn(inputs, len_x)
            framewise = framewise.reshape(batch, temp, -1).transpose(1, 2)
        else:
            # frame-wise features
            framewise = x

        conv1d_outputs = self.conv1d(framewise, len_x)
        # x: T, B, C
        x = conv1d_outputs['visual_feat']
        lgt = conv1d_outputs['feat_len']
        tm_outputs = self.temporal_model(x, lgt)
        outputs = self.classifier(tm_outputs['predictions'])
        pred = None if self.training or not decode \
            else self.decoder.decode(outputs, lgt, batch_first=False, probs=False)
        conv_pred = None if self.training or not decode \
            else self.decoder.decode(conv1d_outputs['conv_logits'], lgt, batch_first=False, probs=False)

        return {
            "framewise_features": framewise,
            "visual_features": x,
            "feat_len": lgt,
            "conv_logits": conv1d_outputs['conv_logits'],
            "sequence_logits": outputs,
            "conv_sents": conv_pred,
            "recognized_sents": pred,
        }

    def criterion_calculation(self, ret_dict, label, label_lgt):
        loss = 0
        self.last_ctc_validity = {}
        for k, weight in self.loss_weights.items():
            if k == 'ConvCTC':
                branch_loss = self._ctc_branch_loss(
                    ret_dict["conv_logits"],
                    label,
                    ret_dict["feat_len"],
                    label_lgt,
                    "ConvCTC",
                )
                loss += weight * branch_loss
            elif k == 'SeqCTC':
                branch_loss = self._ctc_branch_loss(
                    ret_dict["sequence_logits"],
                    label,
                    ret_dict["feat_len"],
                    label_lgt,
                    "SeqCTC",
                )
                loss += weight * branch_loss
            elif k == 'Dist':
                loss += weight * self.loss['distillation'](ret_dict["conv_logits"],
                                                           ret_dict["sequence_logits"].detach(),
                                                           use_blank=False)
        return loss

    def _ctc_branch_loss(
        self,
        logits,
        label,
        input_lengths,
        target_lengths,
        branch_name,
        log_probs=None,
        class_weights=None,
    ):
        if log_probs is not None and log_probs.shape != logits.shape:
            raise ValueError(
                "{} log_probs must match the emission shape".format(
                    branch_name
                )
            )
        if log_probs is not None and class_weights is not None:
            raise ValueError(
                "{} cannot combine normalized log_probs with class weights".format(
                    branch_name
                )
            )
        emission = logits if log_probs is None else log_probs
        if not torch.isfinite(emission).all():
            raise FloatingPointError(
                "{} emissions contain non-finite values".format(branch_name)
            )
        valid_losses = []
        validity = []
        label_offset = 0
        for sample_idx, target_length in enumerate(
            target_lengths.cpu().int().tolist()
        ):
            target = label[
                label_offset:label_offset + target_length
            ].cpu().int()
            label_offset += target_length
            if log_probs is None:
                sample_logits = logits[:, sample_idx:sample_idx + 1, :]
                if class_weights is None:
                    sample_log_probs = sample_logits.log_softmax(-1)
                else:
                    sample_log_probs = weighted_emission_log_probs(
                        sample_logits,
                        class_weights,
                    )
            else:
                sample_log_probs = log_probs[
                    :, sample_idx:sample_idx + 1, :
                ]
            sample_loss = self.loss['CTCLoss'](
                sample_log_probs,
                target,
                input_lengths[sample_idx:sample_idx + 1].cpu().int(),
                target_lengths[sample_idx:sample_idx + 1].cpu().int(),
            ).reshape(())
            sample_valid = torch.isfinite(sample_loss)
            validity.append(sample_valid.detach().cpu())
            if sample_valid:
                valid_losses.append(sample_loss)
        if not valid_losses:
            raise FloatingPointError(
                "{} produced no finite CTC losses".format(branch_name)
            )
        valid = torch.stack(validity)
        self.last_ctc_validity[branch_name] = valid
        return torch.stack(valid_losses).mean()

    def criterion_init(self):
        self.loss['CTCLoss'] = torch.nn.CTCLoss(reduction='none', zero_infinity=False)
        self.loss['distillation'] = SeqKD(T=8)
        return self.loss
