import sys
import numpy as np
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_wave(x, path = './wave.png'):
    plt.gcf().clear()
    plt.plot(x)
    plt.xlabel('n')
    plt.ylabel('xn')
    plt.savefig(path)

def plot_ak(a, path = './freq.png'):
    plt.gcf().clear()

    # Only plot the mag of a
    a = np.abs(a)
    plt.plot(a)
    plt.xlabel('k')
    plt.ylabel('ak')
    plt.savefig(path)

def CosineTrans(x, B):
    # TODO
    # implement cosine transform
    a = np.linalg.inv(B).dot(x)
    return a

def InvCosineTrans(a, B):
    # TODO
    # implement inverse cosine transform
    x= B.dot(a)
    return x

def gen_basis(N):
    # TODO
    B=np.zeros((N,N))
    for n in range(N):
      for k in range(N):
        if k==0:
          B[n,k]=1/(N**0.5)
        else:
          B[n,k]=(2**0.5)/(N**0.5)*np.cos((n+0.5)*k*np.pi/N)

    return B

if __name__ == '__main__':
    # Do not modify these 2 lines
    signal_path = sys.argv[1]
    out_directory_path = sys.argv[2]

    # TODO
    # filter original waveform to f1-only and f3-only time-domain waveform
    x = np.loadtxt('example_data/test.txt', dtype = np.float32).reshape(-1, 1)
    B = gen_basis(len(x))
    a = CosineTrans(x, B)
    # f1 = ...
    a_abs=np.absolute(a)
    # print(a_abs.reshape(1,-1))
    top_5_idx=(a_abs.reshape(1,-1)[0]).argsort()[::-1][0:5]
    # print(top_5_idx)
    top_5_idx=np.sort(top_5_idx)
    # print(top_5_idx)
    F1=np.zeros(a.shape)
    F3=np.zeros(a.shape)
    F1[top_5_idx[0]]= a[top_5_idx[0]]
    F3[top_5_idx[2]]= a[top_5_idx[2]]
    f1=InvCosineTrans(F1, B)
    f3=InvCosineTrans(F3, B)
    # plot_wave(F1, path=os.path.join(out_directory_path, 'F1.png'))
    # plot_wave(F3, path=os.path.join(out_directory_path, 'F3.png'))

    # f3 = ...
    
    # Do not modify these 3 lines
    plot_ak(a, path=os.path.join(out_directory_path, 'freq.png'))
    plot_wave(f1, path=os.path.join(out_directory_path, 'f1.png'))
    plot_wave(f3, path=os.path.join(out_directory_path, 'f3.png'))

