# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/wavlm/hubconf.py ]
#   Synopsis     [ the WavLM torch hubconf ]
#   Author       [ Microsoft ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import os

# -------------#
from s3prl.utility.download import _urls_to_filepaths
from .expert import UpstreamExpert as _UpstreamExpert


def wav2vec_u2_pt5_local(ckpt, *args, **kwargs):
    """
    The model from local ckpt
        ckpt (str): PATH
    """
    assert os.path.isfile(ckpt)
    return _UpstreamExpert(ckpt, *args, **kwargs)

def wav2vec_u2_pt5_base(refresh=False, *args, **kwargs):
    """
        The Base model
            refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs["ckpt"] = '/usr0/home/jiatongs/project/s3prl/s3prl/upstream/wav2vec_u/joint_wav2vec_u_model.pth'
    return wav2vec_u2_pt5_local(*args, ppg = True, hidden = True, **kwargs)


def wav2vec_u2_pt5_base_ppg(refresh=False, *args, **kwargs):
    """
        The Base model
            refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs["ckpt"] = '/usr0/home/jiatongs/project/s3prl/s3prl/upstream/wav2vec_u/joint_wav2vec_u_model.pth'
    return wav2vec_u2_pt5_local(*args, ppg = True, **kwargs)

def wav2vec_u2_pt5_base_hidden(refresh=False, *args, **kwargs):
    """
        The Base model
            refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs["ckpt"] = '/usr0/home/jiatongs/project/s3prl/s3prl/upstream/wav2vec_u/joint_wav2vec_u_model.pth'
    return wav2vec_u2_pt5_local(*args, hidden = True, **kwargs)

def wav2vec_u2_pt5_base_ppg_token(refresh=False, *args, **kwargs):
    """
        The Base model
            refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs["ckpt"] = '/usr0/home/jiatongs/project/s3prl/s3prl/upstream/wav2vec_u/joint_wav2vec_u_model.pth'
    return wav2vec_u2_pt5_local(*args, use_tokenizer=True, ppg = True, **kwargs)

def wav2vec_u2_pt5_base_hidden_token(refresh=False, *args, **kwargs):
    """
        The Base model
            refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs["ckpt"] = '/usr0/home/jiatongs/project/s3prl/s3prl/upstream/wav2vec_u/joint_wav2vec_u_model.pth'
    return wav2vec_u2_pt5_local(*args, use_tokenizer=True, hidden = True, **kwargs)

def wav2vec_u2_pt5_base_token(refresh=False, *args, **kwargs):
    """
        The Base model
            refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs["ckpt"] = '/usr0/home/jiatongs/project/s3prl/s3prl/upstream/wav2vec_u/joint_wav2vec_u_model.pth'
    return wav2vec_u2_pt5_local(*args, use_tokenizer=True, ppg = True, hidden = True, **kwargs)

def wav2vec_u2_pt5_base_ppg_randomize(refresh=False, *args, **kwargs):
    """
        The Base model
            refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs["ckpt"] = '/usr0/home/jiatongs/project/s3prl/s3prl/upstream/wav2vec_u/joint_wav2vec_u_model.pth'
    return wav2vec_u2_pt5_local(*args, ppg = True, randomize=True, **kwargs)

def wav2vec_u2_pt5_base_ppg_randomize_multilayer(refresh=False, *args, **kwargs):
    """
        The Base model
            refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs["ckpt"] = '/usr0/home/jiatongs/project/s3prl/s3prl/upstream/wav2vec_u/joint_wav2vec_u_model.pth'
    return wav2vec_u2_pt5_local(*args, ppg = True, randomize=True, use_t5_hidden=True, **kwargs)

def wav2vec_u2_pt5_base_ppg_multilayer(refresh=False, *args, **kwargs):
    """
        The Base model
            refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs["ckpt"] = '/usr0/home/jiatongs/project/s3prl/s3prl/upstream/wav2vec_u/joint_wav2vec_u_model.pth'
    return wav2vec_u2_pt5_local(*args, ppg = True, use_t5_hidden = True, **kwargs)

def wav2vec_u2_pt5_base_ppg_token_multilayer(refresh=False, *args, **kwargs):
    """
        The Base model
            refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs["ckpt"] = '/usr0/home/jiatongs/project/s3prl/s3prl/upstream/wav2vec_u/joint_wav2vec_u_model.pth'
    return wav2vec_u2_pt5_local(*args, ppg = True, use_t5_hidden = True, use_tokenizer=True, **kwargs)

def wav2vec_u2_pt5_base_kmeans50(refresh=False, *args, **kwargs):
    """
        The Base model
            refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs["ckpt"] = '/usr0/home/jiatongs/project/s3prl/s3prl/upstream/wav2vec_u/joint_wav2vec_u_model.pth'
    return wav2vec_u2_pt5_local(*args, use_kmeans="50", **kwargs)


def wav2vec_u2_pt5_base_ppg_org(refresh=False, *args, **kwargs):
    """
        The Base model
            refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs["ckpt"] = '/usr0/home/jiatongs/project/s3prl/s3prl/upstream/wav2vec_u/joint_wav2vec_u_model.pth'
    return wav2vec_u2_pt5_local(*args, ppg=True, t5="org", **kwargs)