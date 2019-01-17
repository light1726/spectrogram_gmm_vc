import pickle
import argparse
import os
import librosa
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from audio import AcousticExtractor


def main():
    parser = argparse.ArgumentParser(
        description='Feature extraction to src and tgt speaker')
    parser.add_argument('-s', '--src',
                        help='source speaker data directory', type=str)
    parser.add_argument('-t', '--tgt',
                        help='target speaker data directory', type=str)
    parser.add_argument('-x', '--x_lst',
                        help='source speaker wav name list file', type=str)
    parser.add_argument('-y', '--y_lst',
                        help='target speaker wav name list file', type=str)
    parser.add_argument('-o', '--out',
                        help='data output directory')
    parser.add_argument('--minf', help='minimum f0', default=40.0)
    parser.add_argument('--maxf', help='minimum f0', default=500.0)
    parser.add_argument('--sr', help='sample rate', default=16000)
    parser.add_argument('--n_fft', help='number of fft points', default=512)
    parser.add_argument('--win_lenms', help='window length in ms', default=25.0)
    parser.add_argument('--win_shiftms', help='window shift in ms', default=5.0)
    parser.add_argument('--n_mels', help='number of mel bins', default=80)
    args = parser.parse_args()
    src_dir = args.src
    tgt_dir = args.tgt
    src_lst_f = args.x_lst
    tgt_lst_f = args.y_lst
    out_dir = args.out
    src_wav_lst = []
    tgt_wav_lst = []
    with open(src_lst_f, 'r') as f:
        for line in f:
            src_wav_lst.append(os.path.join(src_dir, line.rstrip()))
    with open(tgt_lst_f, 'r') as f:
        for line in f:
            tgt_wav_lst.append(os.path.join(tgt_dir, line.rstrip()))
    assert len(src_wav_lst) == len(tgt_wav_lst)

    src_melspec_lst = []
    tgt_melspec_lst = []
    src_npow_lst = []
    tgt_npow_lst = []
    all_pow_lst_src = []
    all_pow_lst_tgt = []
    ae = AcousticExtractor(fs=args.sr, fftl=args.n_fft, shiftms=args.win_shiftms,
                           win_len=args.win_lenms, minf0=args.minf, maxf0=args.maxf,
                           n_mels=args.n_mels)
    # extract source speaker features
    for wav_f in src_wav_lst:
        wav_arr, sr = librosa.load(wav_f, sr=None)
        assert sr == args.sr
        melspec, _ = ae.spectrogram(wav_arr)
        src_melspec_lst.append(melspec)
        npow = ae.npow(S=melspec)
        src_npow_lst.append(npow)
        all_pow_lst_src += npow.tolist() if type(npow) is np.ndarray else npow
    # extract target speaker features
    for wav_f in tgt_wav_lst:
        wav_arr, sr = librosa.load(wav_f, sr=None)
        assert sr == args.sr
        melspec, _ = ae.spectrogram(wav_arr)
        tgt_melspec_lst.append(melspec)
        npow = ae.npow(S=melspec)
        tgt_npow_lst.append(npow)
        all_pow_lst_tgt += npow.tolist() if type(npow) is np.ndarray else npow

    # save source and target speaker features
    melspec_dict = {'src': src_melspec_lst, 'tgt': tgt_melspec_lst}
    npow_dict = {'src': src_npow_lst, 'tgt': tgt_npow_lst}

    # save data to output dir
    if not os.path.isdir(out_dir):
        print('Output directory {} does not exist, create one.'.format(out_dir))
        os.makedirs(out_dir)
    with open(os.path.join(out_dir, 'melspec_dict.pkl'), 'wb') as f:
        pickle.dump(melspec_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        print('Mel-spectrogram dict saved to {}'.format(
            os.path.join(out_dir, 'melspec_dict.pkl')))
    with open(os.path.join(out_dir, 'npow_dict.pkl'), 'wb') as f:
        pickle.dump(npow_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        print('NPow dict saved to {}'.format(os.path.join(out_dir, 'npow_dict.pkl')))

    # save power histogram
    plt.figure()
    plt.hist(all_pow_lst_src, 100, density=True)
    plt.title('src speaker power')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'src_pow_hist.png'))
    print('src speaker power histogram saved to {}'.format(
        os.path.join(out_dir, 'src_pow_hist.png')))
    plt.figure()
    plt.hist(all_pow_lst_tgt, 100, density=True)
    plt.title('tgt speaker power')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'tgt_pow_hist.png'))
    print('tgt speaker power histogram saved to {}'.format(
        os.path.join(out_dir, 'tgt_pow_hist.png')))
    return


if __name__ == '__main__':
    main()
