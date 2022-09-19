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


def wav2vec_u2_local(ckpt, *args, **kwargs):
    """
    The model from local ckpt
        ckpt (str): PATH
    """
    assert os.path.isfile(ckpt)
    return _UpstreamExpert(ckpt, *args, **kwargs)

def wav2vec_u2_base(refresh=False, *args, **kwargs):
    """
        The Base model
            refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs["ckpt"] = '/usr0/home/jiatongs/project/s3prl/s3prl/upstream/wav2vec_u/joint_wav2vec_u_model.pth'
    return wav2vec_u2_local(*args, ppg = True, hidden = True, **kwargs)


def wav2vec_u2_base_ppg(refresh=False, *args, **kwargs):
    """
        The Base model
            refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs["ckpt"] = '/usr0/home/jiatongs/project/s3prl/s3prl/upstream/wav2vec_u/joint_wav2vec_u_model.pth'
    return wav2vec_u2_local(*args, ppg = True, **kwargs)

def wav2vec_u2_base_hidden(refresh=False, *args, **kwargs):
    """
        The Base model
            refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs["ckpt"] = '/usr0/home/jiatongs/project/s3prl/s3prl/upstream/wav2vec_u/joint_wav2vec_u_model.pth'
    return wav2vec_u2_local(*args, hidden = True, **kwargs)
