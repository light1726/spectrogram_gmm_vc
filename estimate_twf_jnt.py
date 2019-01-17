import os
import pickle
import argparse

from GMM import GMMConvertor, GMMTrainer
from misc import extsddata, transform_jnt
from twf import estimate_twf, align_data
from distance import melcd
from delta import static_delta


def get_alignment(odata, onpow, tdata, tnpow, opow=-20, tpow=-20,
                  sd=0, cvdata=None, given_twf=None, otflag=None,
                  distance='melcd'):
    """Get alignment between original and target

    Paramters
    ---------
    odata : array, shape (`T`, `dim`)
        Acoustic feature vector of original
    onpow : array, shape (`T`)
        Normalized power vector of original
    tdata : array, shape (`T`, `dim`)
        Acoustic feature vector of target
    tnpow : array, shape (`T`)
        Normalized power vector of target
    opow : float, optional,
        Power threshold of original
        Default set to -20
    tpow : float, optional,
        Power threshold of target
        Default set to -20
    sd : int , optional,
        Start dimension to be used for alignment
        Default set to 0
    cvdata : array, shape (`T`, `dim`), optional,
        Converted original data
        Default set to None
    given_twf : array, shape (`T_new`, `dim * 2`), optional,
        Alignment given twf
        Default set to None
    otflag : str, optional
        Alignment into the length of specification
        'org' : alignment into original length
        'tar' : alignment into target length
        Default set to None
    distance : str,
        Distance function to be used
        Default set to 'melcd'

    Returns
    -------
    jdata : array, shape (`T_new` `dim * 2`)
        Joint static and delta feature vector
    twf : array, shape (`T_new` `dim * 2`)
        Time warping function
    mcd : float,
        Mel-cepstrum distortion between arrays

    """

    oexdata = extsddata(odata[:, sd:], onpow,
                        power_threshold=opow)
    texdata = extsddata(tdata[:, sd:], tnpow,
                        power_threshold=tpow)

    if cvdata is None:
        align_odata = oexdata
    else:
        cvexdata = extsddata(cvdata, onpow,
                             power_threshold=opow)
        align_odata = cvexdata

    if given_twf is None:
        twf = estimate_twf(align_odata, texdata,
                           distance=distance, otflag=otflag)
    else:
        twf = given_twf

    jdata = align_data(oexdata, texdata, twf)
    mcd = melcd(align_odata[twf[0]], texdata[twf[1]])

    return jdata, twf, mcd


def align_feature_vectors(odata, onpows, tdata, tnpows, pconf,
                          opow=-100, tpow=-100, itnum=3, sd=0,
                          given_twfs=None, otflag=None):
    """Get alignment to create joint feature vector

    Paramters
    ---------
    odata : list, (`num_files`)
        List of original feature vectors
    onpows : list , (`num_files`)
        List of original npows
    tdata : list, (`num_files`)
        List of target feature vectors
    tnpows : list , (`num_files`)
        List of target npows
    opow : float, optional,
        Power threshold of original
        Default set to -100
    tpow : float, optional,
        Power threshold of target
        Default set to -100
    itnum : int , optional,
        The number of iteration
        Default set to 3
    sd : int , optional,
        Start dimension of feature vector to be used for alignment
        Default set to 0
    given_twf : array, shape (`T_new` `dim * 2`)
        Use given alignment while 1st iteration
        Default set to None
    otflag : str, optional
        Alignment into the length of specification
        'org' : alignment into original length
        'tar' : alignment into target length
        Default set to None

    Returns
    -------
    jfvs : list,
        List of joint feature vectors
    twfs : list,
        List of time warping functions
    """
    num_files = len(odata)
    cvgmm, cvdata = None, None
    for it in range(1, itnum + 1):
        print('{}-th joint feature extraction starts.'.format(it))
        twfs, jfvs = [], []
        for i in range(num_files):
            if it == 1 and given_twfs is not None:
                gtwf = given_twfs[i]
            else:
                gtwf = None
            if it > 1:
                cvdata = cvgmm.convert(static_delta(odata[i][:, sd:]),
                                       cvtype=pconf.cvtype)
            jdata, twf, mcd = get_alignment(odata[i],
                                            onpows[i],
                                            tdata[i],
                                            tnpows[i],
                                            opow=opow,
                                            tpow=tpow,
                                            sd=sd,
                                            cvdata=cvdata,
                                            given_twf=gtwf,
                                            otflag=otflag)
            twfs.append(twf)
            jfvs.append(jdata)
            print('distortion [dB] for {}-th file: {}'.format(i + 1, mcd))
        jnt_data = transform_jnt(jfvs)

        if it != itnum:
            # train GMM, if not final iteration
            datagmm = GMMTrainer(n_mix=pconf.n_mix,
                                 n_iter=pconf.n_iter,
                                 covtype=pconf.covtype)
            datagmm.train(jnt_data)
            cvgmm = GMMConvertor(n_mix=pconf.n_mix,
                                 covtype=pconf.covtype)
            cvgmm.open_from_param(datagmm.param)
        it += 1
    return jfvs, twfs


def pickle_read(pkl_f):
    with open(pkl_f, 'rb') as f:
        var = pickle.load(f)
        return var


def pickle_save(var, path):
    with open(path, 'wb') as f:
        pickle.dump(var, f, protocol=pickle.HIGHEST_PROTOCOL)
    print('Successfully save var to {}'.format(path))
    return


def main():
    parser = argparse.ArgumentParser(
        description='estimate the DTW function and the joint features')
    parser.add_argument('--melspec', help='Melspec data directory')
    parser.add_argument('--npow', help='Npow speaker data directory')
    parser.add_argument('--out_dir', help='Output directory')
    parser.add_argument('--s_npow_th', default=-20.,
                        help='Source speaker npow threshold')
    parser.add_argument('--t_npow_th', default=-15.,
                        help='Target speaker npow threshold')
    parser.add_argument('--n_mix', default=32,
                        help='Number of mixture components')
    parser.add_argument('--n_iter', default=100,
                        help='Number of iterations to get GMM model')
    parser.add_argument('--covtype', default='full',
                        help='GMM covariance type')
    parser.add_argument('--cvtype', default='mlpg',
                        help='conversion mode')
    parser.add_argument('--jnt_n_iter', default=3,
                        help='Number of iterations to get joint features')
    conf = parser.parse_args()
    melspec_dict = pickle_read(conf.melspec)
    npow_dict = pickle_read(conf.npow)
    print('## Alignment mlespec ##')
    jmelspc_lst, twfs = align_feature_vectors(odata=melspec_dict['src'],
                                              onpows=npow_dict['src'],
                                              tdata=melspec_dict['tgt'],
                                              tnpows=npow_dict['tgt'],
                                              pconf=conf,
                                              opow=conf.s_npow_th,
                                              tpow=conf.t_npow_th,
                                              itnum=conf.jnt_n_iter,
                                              sd=0)
    jnt_melspec = transform_jnt(jmelspc_lst)
    if not os.path.isdir(conf.out_dir):
        print('Output directory {} does not exist, create one'.format(conf.out_dir))
    print('Save jnt_melspec and twfs to {}'.format(conf.out_dir))
    pickle_save(jnt_melspec, os.path.join(conf.out_dir, 'jnt_melspc.pkl'))
    pickle_save(twfs, os.path.join(conf.out_dir, 'twf.pkl'))
    print('Done!')
    return


if __name__ == '__main__':
    main()
