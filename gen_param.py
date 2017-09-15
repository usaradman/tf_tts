#!/usr/bin/env python
# -*- coding: utf-8 -*-
#  scripts/gen_wav.py Author "Jinba Xiao <usar@npu-aslp.org>" Date 30.08.2017

import os
import sys
import numpy as np


def decompose(cmp_dir, lf0_dir, mgc_dir, lsp_dim, binary):
    filelist = os.listdir(cmp_dir)
    if(not os.path.exists(lf0_dir)):
        os.mkdir(lf0_dir)
    if(not os.path.exists(mgc_dir)):
        os.mkdir(mgc_dir)

    for file in filelist:
        cmp_file = os.path.join(cmp_dir, file)
        lf0_file = os.path.join(lf0_dir, os.path.splitext(file)[0] + '.lf0')
        mgc_file = os.path.join(mgc_dir, os.path.splitext(file)[0] + '.mgc')
        if os.path.isdir(cmp_file):
            continue

        print(cmp_file)
        if(int(binary)):
            cmp = np.fromfile(cmp_file, dtype=np.float32)
        else:
            cmp = np.fromfile(cmp_file, dtype=np.float32, count=-1, sep=" ")
        cmp = cmp.reshape(len(cmp)/(int(lsp_dim)+9+1), int(lsp_dim)+9+1)

        #lsp = np.zeros((cmp.shape[0], int(lsp_dim)), dtype=np.float32)
        #lf0 = np.zeros(cmp.shape[0], dtype=np.float32)
        lsp = cmp[:,:int(lsp_dim)]
        lf0 = cmp[:,int(lsp_dim)+4]

        UV = cmp[:, int(lsp_dim)+9]
        UV[UV > 0.5] = 1
        UV[UV != 1] = 0
        lf0[UV == 0] = -10000000000.0

        lsp.tofile(mgc_file)
        lf0.tofile(lf0_file)


if __name__ == '__main__':
    # cmp_dir lf0_dir mgc_dir cmp_dim binary
    decompose(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
