# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ dataset.py ]
#   Synopsis     [ the speaker diarization dataset ]
#   Author       [ Jiatong Shi ]
#   Copyright    [ Copyleft(c), Johns Hopkins University ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import os
import random
#-------------#
import pandas as pd
#-------------#
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataset import Dataset
#-------------#
import torchaudio


def _count_frames(data_len, size, step):
    # no padding at edges, last remaining samples are ignored
    return int((data_len - size + step) / step)


def _gen_frame_indices(
        data_length, size=2000, step=2000,
        use_last_samples=False,
        label_delay=0,
        subsampling=1):
    i = -1
    for i in range(_count_frames(data_length, size, step)):
        yield i * step, i * step + size
    if use_last_samples and i * step + size < data_length:
        if data_length - (i + 1) * step - subsampling * label_delay > 0:
            yield (i + 1) * step, data_length


#######################
# Diarization Dataset #
#######################
class DiarizationDataset(Dataset):

    def __init__(
            self,
            data_dir,
            dtype=np.float32,
            chunk_size=2000,
            frame_shift=256,
            subsampling=1,
            rate=16000,
            input_transform=None,
            use_last_samples=False,
            label_delay=0,
            n_speakers=None,
            max_frame_num=2000,
        ):
        super(DiarizationDataset, self).__init__()

        self.data_dir = data_dir
        self.dtype = dtype
        self.chunk_size = chunk_size
        self.frame_shift = frame_shift
        self.subsampling = subsampling
        self.n_speakers = n_speakers
        self.chunk_indices = []
        self.label_delay = label_delay
        self.max_frame_num = max_frame_num

        self.data = kaldi_data.KaldiData(self.data_dir)

        # make chunk indices: filepath, start_frame, end_frame
        for rec in self.data.wavs:
            data_len = int(self.data.reco2dur[rec] * rate / frame_shift)
            data_len = int(data_len / self.subsampling)
            for st, ed in _gen_frame_indices(
                    data_len, chunk_size, chunk_size, use_last_samples,
                    label_delay=self.label_delay,
                    subsampling=self.subsampling):
                self.chunk_indices.append(
                        (rec, st * self.subsampling, ed * self.subsampling))
        print(len(self.chunk_indices), " chunks")
            
    def __len__(self):
        return len(self.chunk_indices)

    def __getitem__(self, i):
        rec, st, ed = self.chunk_indices[i]
        Y, T = self._get_labeled_speech(
            self.data,
            rec,
            st,
            ed,
            self.frame_shift,
            self.n_speakers)
        return Y_ss, T_ss
    
    def _get_labeled_speech(self,         
        rec, start, end, frame_shift,
        n_speakers=None,
        use_speaker_id=False):
        """ Extracts speech chunks and corresponding labels

        Extracts speech chunks and corresponding diarization labels for
        given recording id and start/end times

        Args:
            rec (str): recording id
            start (int): start frame index
            end (int): end frame index
            frame_shift (int): number of shift samples
            n_speakers (int): number of speakers
                if None, the value is given from data
        Returns:
            Y: speech chunk
                (n_samples)
            T: label
                (n_frmaes, n_speakers)-shaped np.int32 array.
        """
        data, rate = self.data.load_wav(
            rec, start * frame_shift, end * frame_shift)
        filtered_segments = self.data.segments[rec]
        # filtered_segments = self.data.segments[self.data.segments['rec'] == rec]
        speakers = np.unique(
                [self.data.utt2spk[seg['utt']] for seg
                    in filtered_segments]).tolist()
        if n_speakers is None:
            n_speakers = len(speakers)
        T = np.zeros((Y.shape[0], n_speakers), dtype=np.int32)

        if use_speaker_id:
            all_speakers = sorted(self.data.spk2utt.keys())
            S = np.zeros((Y.shape[0], len(all_speakers)), dtype=np.int32)

        for seg in filtered_segments:
            speaker_index = speakers.index(self.data.utt2spk[seg['utt']])
            if use_speaker_id:
                all_speaker_index = all_speakers.index(
                    self.data.utt2spk[seg['utt']])
            start_frame = np.rint(
                    seg['st'] * rate / frame_shift).astype(int)
            end_frame = np.rint(
                    seg['et'] * rate / frame_shift).astype(int)
            rel_start = rel_end = None
            if start <= start_frame and start_frame < end:
                rel_start = start_frame - start
            if start < end_frame and end_frame <= end:
                rel_end = end_frame - start
            if rel_start is not None or rel_end is not None:
                T[rel_start:rel_end, speaker_index] = 1
                if use_speaker_id:
                    S[rel_start:rel_end, all_speaker_index] = 1

        if use_speaker_id:
            return Y, T, S
        else:
            return Y, T
    
    def collate_fn(self, batch):
        batch_size = len(batch)
        feat_dim = batch[0][0].shape[1]
        len_list = [len(batch[i][0]) for i in range(batch_size)]
        feat = np.zeros((batch_size, self.max_len, feat_dim))
        label = np.zeros((batch_size, self.max_len, 2))
        for i in range(batch_size):
            length = len_list[i]
            feat[i, :length, :] = batch[i][0]
            label[i, :length, :] = batch[i][1]
        length = np.array(len_list)
        feat = torch.from_numpy(feat)
        label = torch.from_numpy(label)
        length = torch.from_numpy(length)
        return feat, label, length


#######################
# Kaldi-style Dataset #
#######################
class KaldiData:
    """This class holds data in kaldi-style directory."""

    def __init__(self, data_dir):
        """Load kaldi data directory."""
        self.data_dir = data_dir
        self.segments = self.load_segments_rechash(
            os.path.join(self.data_dir, 'segments'))
        self.utt2spk = self.load_utt2spk(
            os.path.join(self.data_dir, 'utt2spk'))
        self.wavs = self.load_wav_scp(
            os.path.join(self.data_dir, 'wav.scp'))
        self.reco2dur = self.load_reco2dur(
            os.path.join(self.data_dir, 'reco2dur'))
        self.spk2utt = self.load_spk2utt(
            os.path.join(self.data_dir, 'spk2utt'))

    def load_wav(self, recid, start=0, end=None):
        """Load wavfile given recid, start time and end time."""
        data, rate = load_wav(
            self.wavs[recid], start, end)
        return data, rate

    def load_segments(self, segments_file):
        """Load segments file as array."""
        if not os.path.exists(segments_file):
            return None
        return np.loadtxt(
            segments_file,
            dtype=[('utt', 'object'),
                ('rec', 'object'),
                ('st', 'f'),
                ('et', 'f')],
            ndmin=1)


    def load_segments_hash(self, segments_file):
        """Load segments file as dict with uttid index."""
        ret = {}
        if not os.path.exists(segments_file):
            return None
        for line in open(segments_file):
            utt, rec, st, et = line.strip().split()
            ret[utt] = (rec, float(st), float(et))
        return ret


    def load_segments_rechash(self, segments_file):
        """Load segments file as dict with recid index."""
        ret = {}
        if not os.path.exists(segments_file):
            return None
        for line in open(segments_file):
            utt, rec, st, et = line.strip().split()
            if rec not in ret:
                ret[rec] = []
            ret[rec].append({'utt': utt, 'st': float(st), 'et': float(et)})
        return ret


    def load_wav_scp(self, wav_scp_file):
        """Return dictionary { rec: wav_rxfilename }."""
        lines = [line.strip().split(None, 1) for line in open(wav_scp_file)]
        return {x[0]: x[1] for x in lines}


    @lru_cache(maxsize=1)
    def load_wav(self, wav_rxfilename, start=0, end=None):
        """This function reads audio file and return data in numpy.float32 array.
        "lru_cache" holds recently loaded audio so that can be called
        many times on the same audio file.
        OPTIMIZE: controls lru_cache size for random access,
        considering memory size
        """
        if wav_rxfilename.endswith('|'):
            # input piped command
            p = subprocess.Popen(
                wav_rxfilename[:-1],
                shell=True,
                stdout=subprocess.PIPE,
            )
            data, samplerate = sf.read(
                io.BytesIO(p.stdout.read()),
                dtype='float32',
            )
            # cannot seek
            data = data[start:end]
        elif wav_rxfilename == '-':
            # stdin
            data, samplerate = sf.read(sys.stdin, dtype='float32')
            # cannot seek
            data = data[start:end]
        else:
            # normal wav file
            data, samplerate = sf.read(wav_rxfilename, start=start, stop=end)
        return data, samplerate


    def load_utt2spk(self, utt2spk_file):
        """Returns dictionary { uttid: spkid }."""
        lines = [line.strip().split(None, 1) for line in open(utt2spk_file)]
        return {x[0]: x[1] for x in lines}


    def load_spk2utt(self, spk2utt_file):
        """Returns dictionary { spkid: list of uttids }."""
        if not os.path.exists(spk2utt_file):
            return None
        lines = [line.strip().split() for line in open(spk2utt_file)]
        return {x[0]: x[1:] for x in lines}


    def load_reco2dur(self, reco2dur_file):
        """Returns dictionary { recid: duration }."""
        if not os.path.exists(reco2dur_file):
            return None
        lines = [line.strip().split(None, 1) for line in open(reco2dur_file)]
        return {x[0]: float(x[1]) for x in lines}


    def process_wav(self, wav_rxfilename, process):
        """This function returns preprocessed wav_rxfilename.
        Args:
            wav_rxfilename:
                input
            process:
                command which can be connected via pipe, use stdin and stdout
        Returns:
            wav_rxfilename: output piped command
        """
        if wav_rxfilename.endswith('|'):
            # input piped command
            return wav_rxfilename + process + '|'
        # stdin "-" or normal file
        return 'cat {0} | {1} |'.format(wav_rxfilename, process)


    def extract_segments(self, wavs, segments=None):
        """This function returns generator of segmented audio.
        Yields (utterance id, numpy.float32 array).
        TODO?: sampling rate is not converted.
        """
        if segments is not None:
            # segments should be sorted by rec-id
            for seg in segments:
                wav = wavs[seg['rec']]
                data, samplerate = load_wav(wav)
                st_sample = np.rint(seg['st'] * samplerate).astype(int)
                et_sample = np.rint(seg['et'] * samplerate).astype(int)
                yield seg['utt'], data[st_sample:et_sample]
        else:
            # segments file not found,
            # wav.scp is used as segmented audio list
            for rec in wavs:
                data, samplerate = load_wav(wavs[rec])
                yield rec, data