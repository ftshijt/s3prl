import argparse
import torch
import logging

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )

    # TODO(jiatong): change to argument
    # wav2vec_u_model_path = "/usr0/home/jiatongs/project/fairseq/examples/wav2vec/unsupervised/multirun/2022-06-14/04-15-20/0/checkpoint_best.pt"
    # wav2vec_model_path = "/usr0/home/jiatongs/project/fairseq/examples/wav2vec/unsupervised/librispeech/wav2vec_vox_new.pt"
    # wav2vec_u_model_dict = "/usr0/home/jiatongs/project/fairseq/examples/wav2vec/unsupervised/librispeech/text/phones/dict.phn.txt"
    wav2vec_u_model_path = "wav2vec_u_model.pt"
    wav2vec_model_path = "wav2vec_vox_new.pt"
    wav2vec_u_model_dict = "dict.phn.txt"
    output_path = "joint_wav2vec_u_model.pth"
    layer_index = 14

    logging.info("start merging config")

    # u_model = torch.load(wav2vec_u_model_path)
    # logging.info("loaded unsupervised model")
    # model = torch.load(wav2vec_model_path)
    # logging.info("loaded ssl model")

    # u_model_param = u_model["model"]
    # model_param = model["model"]
    # u_model_args = u_model["cfg"]["model"]
    # model_args = model["args"]
    # u_model_args["use_layer"] = layer_index


    # joint_dict = {}
    # joint_dict["ssl"] = model_param
    # joint_dict["ssl_config"] = model_args
    # joint_dict["u_model"] = u_model_param
    # joint_dict["u_model_args"] = u_model_args

    joint_dict = {}
    joint_dict["ssl"] = wav2vec_model_path
    joint_dict["u_model"] = wav2vec_u_model_path
    joint_dict["u_dict"] = wav2vec_u_model_dict
    joint_dict["interface"] = {
        "mode": "single_layer",
        "value": 14,
        "u_downsample_rate": 3,
    } 

    logging.info("saving joint dictionary")
    torch.save(joint_dict, output_path)

    




    
