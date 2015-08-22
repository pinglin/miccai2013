# Real-time dense stereo reconstruction

This repository maintains for the source code of the work

* Chang, P.-L., Stoyanov, D., Davison, A.: Real-time dense stereo reconstruction using convex optimisation with a cost-volume for image-guided robotic surgery. MICCAI 2013.

If you use the code, please cite the paper. Thank you.

## Install

Compile the CUDA MEX-Files in the directory `gpu`, see: [mex cuda setup](http://www.mathworks.co.uk/help/distcomp/create-and-run-mex-files-containing-cuda-code.html)

In MacOS for example, you have to: 

1. Copy mexopts.sh from the default path to `gpu`
2. Find the CUDA SDK header directory `/usr/local/cuda/samples/common/inc`

In OSX MATLAB R2013a, the `mexopts.sh` is in 

`/Applications/MATLAB_R2013a.app/toolbox/distcomp/gpu/extern/src/mex/maci64`


Depends on your CUDA SDK you may have to remove all 

`-gencode=arch=compute_13,code=sm_13` in `mexopts.sh`.

To compile:

`mex -v -O -largeArrayDims -I/usr/local/cuda/samples/common/inc HuberL1CVPrecond_mex.cu`

**Note that using -largeArrayDims is neccessary.**
