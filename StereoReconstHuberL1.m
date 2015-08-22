function StereoReconstHuberL1(left_img_file, right_img_file)
%STEREORECONSTHUBERL1 demonstrates the stereo reconstruction algorithm
%using convex optimisation with a cost-volume for solving the Huber-L1
%model
%
% SYNOPSIS: StereoReconstHuberL1(left_img_file, right_img_file)
%
% INPUT left_img_file: Left rectified image.
%       right_img_file: Right rectified image
%
% OUTPUT Nan
%
% REMARKS This demo requires a CUDA-capable graphic card onboard. To
% compile the CUDA MEX-Files in "gpu/", please refer to: 
% http://www.mathworks.co.uk/help/distcomp/create-and-run-mex-files-containing-cuda-code.html
% 
% In MacOS for example, you have to: 
%   1. Copy the mexopts.sh from your Matlab system
%   2. Set the path of nvcc 'setenv('MW_NVCC_PATH','/usr/local/CUDA/bin/nvcc')'
%	3. Find the CUDA SDK header directory '/usr/local/cuda/samples/common/inc'
%
% Therefore to compile:
% mex -v -O -largeArrayDims -I/usr/local/cuda/samples/common/inc HuberL1CVPrecond_mex.cu
%
% Note that using -largeArrayDims is neccessary.
%
% For the algorithm details, please refer to:
% [1] Ping-Lin Chang, Danail Stoyanov, Andrew J. Davison, Philip "Eddie" Edwards. 
%     Real-Time Dense Stereo Reconstruction Using Convex Optimisation with a Cost-Volume 
%     for Image-Guided Robotic Surgery, MICCAI, 2013.
%
% Please report issues to p.chang10@imperial.ac.uk, many thanks.
%
% created with MATLAB ver.: 8.0.0.783 (R2012b) on Mac OS X  Version: 10.8.4 Build: 12E55 
%
% created by: Ping-Lin Chang
% DATE: 20-Sep-2013

gpu_enable = false;
if(gpuDeviceCount > 0)
    gpu_enable = true;
    addpath('gpu');
    
    % Using only one gpu.
    gpu = gpuDevice(1);
end

if(gpu_enable)
    
    left_img = im2single(rgb2gray(imread(left_img_file)));
    right_img = im2single(rgb2gray(imread(right_img_file)));                    
        
    CostVoumeParams = struct('min_disp', uint8(0), ...
                             'max_disp', uint8(64), ...
                             'method', 'zncc', ...
                             'win_r', uint8(4), ...
                             'ref_left', true);

    PrimalDualParams = struct('num_itr', uint32(500), ...
                              'alpha', single(10.0), ...
                              'beta', single(1.0), ...
                              'epsilon', single(0.1), ...
                              'lambda', single(1e-3), ...
                              'aux_theta', single(10), ...
                              'aux_theta_gamma', single(1e-6));
    [d, primal, dual, primal_step, dual_step, errors_precond] =  HuberL1CVPrecond_mex(left_img, right_img, CostVoumeParams, PrimalDualParams);
    
    opt_disp = gather(primal);   
    opt_disp = (opt_disp-min(min(opt_disp)))/(max(max(opt_disp)) - min(min(opt_disp)));
        
    figure;
    imshow(opt_disp);
        
    figure;
    plot(gather(errors_precond), 'g');    
    grid on;
    legend('HuberL1+Cost-Volume');
    xlabel('Iterations');
    ylabel('Energy function');
    
    % Release GPU memory
    reset(gpu);
    rmpath('gpu')
end