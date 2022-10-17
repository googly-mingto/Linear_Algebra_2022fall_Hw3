import cv2
import numpy as np
from p1 import gen_basis
import os
import sys

# input: Basis (B), 2D image array (I)
# output: coefficient in 2D (a)
def CosineTrans2d(B, I):
    a = B.T.dot(I[:,:,0].dot(B))
    return a

# input: Basis (B), coefficient in 2D (a)
# output: reconstructed image (I')
def InvCosineTrans2d(B, a):
    I =B.dot(a.dot(B.T))
    return I


'''
function : compress_grid
---
2D DCT -> coefficient -> inverse 2D DCT -> reconstructed and compressed image 
'''
def encrypt(I):
  count = 0
  N = len(I)
    
  # generate basis "B" and compute variable "DCT" coefficient of grid I  
  B = gen_basis(N)
  # print(B.shape)
  A = CosineTrans2d(B, I)
  A= np.fliplr(A)
  A= np.flipud(A)
  Am=np.zeros(A.shape)
  Am=A
  # Am[1:,1:]=A[:-1,:-1]
  Am=Am.T
  

            
  # do inverse 2D DCT on "DCT_chopped" to reconstruct the grid, save as "reconstruct_I"
  reconstruct_I = InvCosineTrans2d(B, Am)

  return reconstruct_I + 128

if __name__ == '__main__':
  im_path = sys.argv[1]
  output_path = sys.argv[2]
  grid_sz = 702


  # read image
  I = cv2.imread(im_path).astype('float')
  I -= 128.0
  N = len(I)

  reconstruct_I = np.zeros((N, N))
  # crop original image to 8*8 grid and do JPEG compression for each grid
  for r in range(N // grid_sz):
    for c in range(N // grid_sz):
      grid = I[r*grid_sz:r*grid_sz+grid_sz, c*grid_sz:c*grid_sz+grid_sz]
      reconstruct_chopped = encrypt(grid)
    
      reconstruct_I[r*grid_sz:r*grid_sz+grid_sz, c*grid_sz:c*grid_sz+grid_sz] = reconstruct_chopped


  # save reconstructed image
  cv2.imwrite(os.path.join(output_path, 'messed.png'), reconstruct_I)
