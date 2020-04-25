import sys
import os
import configs
import ZSSR
import torch
import numpy as np


def main(input_img, ground_truth, kernels, gpu, conf_str, results_path):
    # Choose the wanted GPU
    

    # 0 input for ground-truth or kernels means None
    ground_truth = None if ground_truth == '0' else ground_truth
    print('*****', kernels)
    kernels = None if kernels == '0' else kernels.split(';')[:-1]

    # Setup configuration and results directory
    conf = configs.Config()
    if conf_str is not None:
        exec ('conf = configs.%s' % conf_str)
    conf.result_path = results_path

    # Run ZSSR on the image
    # os.environ["OMP_NUM_THREADS"] = "1"
    torch.set_num_threads(20)
    net = ZSSR.ZSSR(input_img, conf, ground_truth, kernels)
    output, gt = net.run()
    error = np.mean((output-gt)**2)
    psnr = -10*np.log10(error)
    print("Test Error {:.3f} PSNR {:.3f}".format(error, psnr))
    return error, psnr



if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])
