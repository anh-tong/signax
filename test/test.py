import torch
import signatory

batch_num, path_len, feature_num = 10, 4, 3
sig_depth = 3
a = torch.randn((batch_num, path_len, feature_num))

sig = signatory.signature(a, sig_depth)


def get_sig_size():
    total_features_size = 0
    acc_features_size = feature_num

    for i in range(sig_depth):
        total_features_size += acc_features_size
        acc_features_size *= feature_num

    return (batch_num, total_features_size)
