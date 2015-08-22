#include <iostream>

#include <string>

#include "mex.h"
#include "gpu/mxGPUArray.h"

#include "HuberL1CVPrecond_kernels.cuh"

using namespace std;

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[])
{

    mxInitGPU();

    ////////////////////////////////////////////////////////////////////////////
    //  Check and init the input/output variables.
    ////////////////////////////////////////////////////////////////////////////
    if(nlhs < 3)
        mexErrMsgIdAndTxt("StereoReconstTest:gpu:HuberL1CV_mex:prhs",
                          "Output number doesn't match.");

    if(nrhs < 3)
        mexErrMsgIdAndTxt("StereoReconstTest:gpu:HuberL1CV_mex:prhs",
                          "Input number doesn't match.");

    if(!mxIsSingle(prhs[0]) || !mxIsSingle(prhs[1]))
        mexErrMsgIdAndTxt("StereoReconstTest:gpu:HuberL1CV_mex:prhs",
                          "Support Images (prhs[0] and prhs[1]) only float images.");

    if(mxGetM(prhs[0]) != mxGetM(prhs[1]) || mxGetN(prhs[0]) != mxGetN(prhs[1]))
        mexErrMsgIdAndTxt("StereoReconstTest:gpu:HuberL1CV_mex:prhs",
                          "prhs[0] and prhs[1] size inconsistent.");

    size_t width = mxGetM(prhs[0]);
    size_t height = mxGetN(prhs[0]);

    /* Check cost volume parameters */
    if(!mxIsStruct(prhs[2]))
        mexErrMsgIdAndTxt("StereoReconstTest:gpu:HuberL1CV_mex:prhs",
                          "prhs[2] should be a struct for cost volume parameters.");

    if(!mxIsUint8(mxGetField(prhs[2], 0, "min_disp")) || !mxIsUint8(mxGetField(prhs[2], 0, "max_disp")))
        mexErrMsgIdAndTxt("StereoReconstTest:gpu:HuberL1CV_mex:prhs",
                          "Disparity disp_min and disp_max must be a uint8 type.");

    if(!mxIsChar(mxGetField(prhs[2], 0, "method")))
        mexErrMsgIdAndTxt("StereoReconstTest:gpu:HuberL1CV_mex:prhs",
                          "Method for building the cost volume must be a string.");

    if(!mxIsLogical(mxGetField(prhs[2], 0, "ref_left")))
        mexErrMsgIdAndTxt("StereoReconstTest:gpu:HuberL1CV_mex:prhs",
                          "ref_left indicator must be logical type.");

    CostVolumeParams host_cv_params;
    host_cv_params.min_disp = *(uint8_t*)mxGetData(mxGetField(prhs[2], 0, "min_disp"));
    host_cv_params.max_disp = *(uint8_t*)mxGetData(mxGetField(prhs[2], 0, "max_disp"));
    host_cv_params.num_disp_layers = host_cv_params.max_disp-host_cv_params.min_disp+1;

    host_cv_params.method = mxArrayToString(mxGetField(prhs[2], 0, "method"));
    host_cv_params.win_r = *(uint8_t*)mxGetData(mxGetField(prhs[2], 0, "win_r"));
    host_cv_params.ref_img = *(bool*)mxGetData(mxGetField(prhs[2], 0, "ref_left")) ? LeftRefImage:RightRefImage;

    CostVolumeParams* dev_cv_params;
    checkCudaErrors(cudaMalloc((void**)&dev_cv_params, sizeof(CostVolumeParams)));
    checkCudaErrors(cudaMemcpy(dev_cv_params, &host_cv_params, sizeof(CostVolumeParams), cudaMemcpyHostToDevice));

    /* Check primal dual parameters */
    if(!mxIsStruct(prhs[3]))
        mexErrMsgIdAndTxt("StereoReconstTest:gpu:HuberL1CV_mex:prhs",
                          "prhs[3] should be a struct for the parameters of primal-dual optimisation.");

    if(!mxIsUint32(mxGetField(prhs[3], 0, "num_itr")))
        mexErrMsgIdAndTxt("StereoReconstTest:gpu:HuberL1CV_mex:prhs",
                          "The number of iterations for primal-dual optimisation must be uint32.");

    if(!mxIsSingle(mxGetField(prhs[3], 0, "alpha")))
        mexErrMsgIdAndTxt("StereoReconstTest:gpu:HuberL1CV_mex:prhs",
                          "Edge-weighting (alpha) must be float.");

    if(!mxIsSingle(mxGetField(prhs[3], 0, "beta")))
        mexErrMsgIdAndTxt("StereoReconstTest:gpu:HuberL1CV_mex:prhs",
                          "Edge-weighting (beta) must be float.");

    if(!mxIsSingle(mxGetField(prhs[3], 0, "epsilon")))
        mexErrMsgIdAndTxt("StereoReconstTest:gpu:HuberL1CV_mex:prhs",
                          "Huber range (epsilon) must be float.");

    if(!mxIsSingle(mxGetField(prhs[3], 0, "lambda")))
        mexErrMsgIdAndTxt("StereoReconstTest:gpu:HuberL1CV_mex:prhs",
                          "Data-regulariser weight (lambda) must be float.");

    if(!mxIsSingle(mxGetField(prhs[3], 0, "aux_theta")))
        mexErrMsgIdAndTxt("StereoReconstTest:gpu:HuberL1CV_mex:prhs",
                          "Auxiliary step theta (aux_theta) must be float.");

    PrimalDualParams host_pd_params;
    host_pd_params.num_itr = *(int*)mxGetData(mxGetField(prhs[3], 0, "num_itr"));
    host_pd_params.alpha = *(float*)mxGetData(mxGetField(prhs[3], 0, "alpha"));
    host_pd_params.beta = *(float*)mxGetData(mxGetField(prhs[3], 0, "beta"));
    host_pd_params.epsilon = *(float*)mxGetData(mxGetField(prhs[3], 0, "epsilon"));
    host_pd_params.lambda = *(float*)mxGetData(mxGetField(prhs[3], 0, "lambda"));
    host_pd_params.aux_theta = *(float*)mxGetData(mxGetField(prhs[3], 0, "aux_theta"));
    host_pd_params.aux_theta_gamma = *(float*)mxGetData(mxGetField(prhs[3], 0, "aux_theta_gamma"));
    host_pd_params.theta = 1.0;

    PrimalDualParams* dev_pd_params;
    checkCudaErrors(cudaMalloc((void**)&dev_pd_params, sizeof(PrimalDualParams)));
    checkCudaErrors(cudaMemcpy(dev_pd_params, &host_pd_params, sizeof(PrimalDualParams), cudaMemcpyHostToDevice));

    /* Allocate device memory and copy left and right image. */
    cudaArray *left_img_array, *right_img_array;
    cudaChannelFormatDesc channelDesc_float = cudaCreateChannelDesc<float>();
    checkCudaErrors(cudaMallocArray(&left_img_array, &channelDesc_float, width, height));
    checkCudaErrors(cudaMallocArray(&right_img_array, &channelDesc_float, width, height));
    checkCudaErrors(cudaMemcpyToArray(left_img_array, 0, 0, mxGetPr(prhs[0]), width*height*sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyToArray(right_img_array, 0, 0, mxGetPr(prhs[1]), width*height*sizeof(float), cudaMemcpyHostToDevice));

    /* Allocate cost-volume 3D memory. */
    cudaPitchedPtr cost_volume;
    checkCudaErrors(cudaMalloc3D(&cost_volume, make_cudaExtent(width*sizeof(float), height, host_cv_params.num_disp_layers)));

    ////////////////////////////////////////////////////////////////////////////
    //  Cost-volume building
    ////////////////////////////////////////////////////////////////////////////

    /* Bind to read-only textures */
    if(host_cv_params.ref_img == LeftRefImage)
    {
        checkCudaErrors(cudaBindTextureToArray(ref_img_tex, left_img_array));
        checkCudaErrors(cudaBindTextureToArray(target_img_tex, right_img_array));
    }
    else if(host_cv_params.ref_img == RightRefImage)
    {
        checkCudaErrors(cudaBindTextureToArray(ref_img_tex, right_img_array));
        checkCudaErrors(cudaBindTextureToArray(target_img_tex, left_img_array));
    }

    size_t THREAD_NUM_3D_BLOCK = 8;
    dim3 dimBlock(THREAD_NUM_3D_BLOCK, THREAD_NUM_3D_BLOCK,THREAD_NUM_3D_BLOCK);
    dim3 dimGrid((width+dimBlock.x-1)/dimBlock.x,
                 (height+dimBlock.y-1)/dimBlock.y,
                 (host_cv_params.num_disp_layers+dimBlock.z-1)/dimBlock.z);

    /* AD */
    if(host_cv_params.method == "ad")
        ADKernel<<<dimGrid, dimBlock>>>(cost_volume,
                                        dev_cv_params,
                                        width,
                                        height);

    /* ZNCC */
    if(host_cv_params.method == "zncc")
        ZNCCKernel<<<dimGrid, dimBlock>>>(cost_volume,
                                          dev_cv_params,
                                          width,
                                          height);

    /* Copy cost-volume to 3D array and bind it to 3D texture for fast accessing */
    cudaArray* cost_volume_array;
    checkCudaErrors(cudaMalloc3DArray(&cost_volume_array, &channelDesc_float, make_cudaExtent(width, height, host_cv_params.num_disp_layers), cudaArraySurfaceLoadStore));

    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr = cost_volume;
    copyParams.dstArray = cost_volume_array;
    copyParams.extent = make_cudaExtent(width, height, host_cv_params.num_disp_layers);
    copyParams.kind   = cudaMemcpyDeviceToDevice;
    checkCudaErrors(cudaMemcpy3D(&copyParams));

    checkCudaErrors(cudaBindTextureToArray(cost_volume_tex, cost_volume_array, channelDesc_float));
    checkCudaErrors(cudaFree(cost_volume.ptr));

    ////////////////////////////////////////////////////////////////////////////
    //  Winnder-take-all (WTA) scheme to initialise disp
    ////////////////////////////////////////////////////////////////////////////
    cudaPitchedPtr min_disp, min_disp_cost, max_disp_cost;
    checkCudaErrors(cudaMallocPitch((void **)&min_disp.ptr, &min_disp.pitch, width*sizeof(Primal), height));
    checkCudaErrors(cudaMemset2D(min_disp.ptr, min_disp.pitch, 0.0, width*sizeof(Primal), height));

    checkCudaErrors(cudaMallocPitch((void **)&min_disp_cost.ptr, &min_disp_cost.pitch, width*sizeof(float), height));
    checkCudaErrors(cudaMemset2D(min_disp_cost.ptr, min_disp_cost.pitch, 0.0, width*sizeof(float), height));

    checkCudaErrors(cudaMallocPitch((void **)&max_disp_cost.ptr, &max_disp_cost.pitch, width*sizeof(float), height));
    checkCudaErrors(cudaMemset2D(max_disp_cost.ptr, max_disp_cost.pitch, 0.0, width*sizeof(float), height));

    size_t THREAD_NUM_2D_BLOCK = 16;
    dimBlock = dim3(THREAD_NUM_2D_BLOCK, THREAD_NUM_2D_BLOCK);
    dimGrid = dim3((width+dimBlock.x-1)/dimBlock.x,
                   (height+dimBlock.y-1)/dimBlock.y);

    WTAKernel<<<dimGrid, dimBlock>>>(min_disp,
                                     min_disp_cost,
                                     max_disp_cost,
                                     dev_cv_params,
                                     width,
                                     height);

    checkCudaErrors(cudaBindTexture2D(0, min_disp_cost_tex, min_disp_cost.ptr,
                                      cudaCreateChannelDesc<float>(),
                                      width, height, min_disp_cost.pitch));

    checkCudaErrors(cudaBindTexture2D(0, max_disp_cost_tex, max_disp_cost.ptr,
                                      cudaCreateChannelDesc<float>(),
                                      width, height, max_disp_cost.pitch));

    ////////////////////////////////////////////////////////////////////////////
    //  Primal-dual + cost-volume optimisation
    ////////////////////////////////////////////////////////////////////////////

    /* Allocate and initialise variables */
    cudaPitchedPtr primal, old_primal, head_primal, dual, aux, diffuse_tensor, error_img;
    cudaPitchedPtr primal_step, dual_step;

    /* Primal variables */
    checkCudaErrors(cudaMallocPitch((void **)&primal.ptr, &primal.pitch, width*sizeof(Primal), height));
    checkCudaErrors(cudaMemcpy2D(primal.ptr, primal.pitch, min_disp.ptr, min_disp.pitch, width*sizeof(Primal), height, cudaMemcpyDeviceToDevice));

    checkCudaErrors(cudaMallocPitch((void **)&head_primal.ptr, &head_primal.pitch, width*sizeof(Primal), height));
    checkCudaErrors(cudaMemcpy2D(head_primal.ptr, head_primal.pitch, min_disp.ptr, min_disp.pitch, width*sizeof(Primal), height, cudaMemcpyDeviceToDevice));

    checkCudaErrors(cudaMallocPitch((void **)&old_primal.ptr, &old_primal.pitch, width*sizeof(Primal), height));

    checkCudaErrors(cudaMallocPitch((void **)&primal_step.ptr, &primal_step.pitch, width*sizeof(PrimalStep), height));
    checkCudaErrors(cudaMemset2D(primal_step.ptr, primal_step.pitch, 0.0, width*sizeof(PrimalStep), height));

    /* Dual variables */    
    checkCudaErrors(cudaMallocPitch((void **)&dual.ptr, &dual.pitch, width*sizeof(Dual), height));
    checkCudaErrors(cudaMemset2D(dual.ptr, dual.pitch, 0.0, width*sizeof(Dual), height));

    checkCudaErrors(cudaMallocPitch((void **)&dual_step.ptr, &dual_step.pitch, width*sizeof(DualStep), height));
    checkCudaErrors(cudaMemset2D(dual_step.ptr, dual_step.pitch, 0.0, width*sizeof(DualStep), height));

    /* Auxiliary variable */
    checkCudaErrors(cudaMallocPitch((void **)&aux.ptr, &aux.pitch, width*sizeof(Auxiliary), height));
    checkCudaErrors(cudaMemcpy2D(aux.ptr, aux.pitch, min_disp.ptr, min_disp.pitch, width*sizeof(Auxiliary), height, cudaMemcpyDeviceToDevice));

    /* Weighting matrix using 2x2 D tensor matrix */    
    checkCudaErrors(cudaMallocPitch((void **)&diffuse_tensor.ptr, &diffuse_tensor.pitch, width*sizeof(DiffuseTensor), height));
    checkCudaErrors(cudaMemset2D(diffuse_tensor.ptr, diffuse_tensor.pitch, 0.0, width*sizeof(DiffuseTensor), height));

    /* Point-wise errors */
    checkCudaErrors(cudaMallocPitch((void **)&error_img.ptr, &error_img.pitch, width*sizeof(float), height));
    checkCudaErrors(cudaMemset2D(error_img.ptr, error_img.pitch, 0.0, width*sizeof(float), height));

    /* Calculating diffusion tensor */
    DiffuseTensorKernel<<<dimGrid, dimBlock>>>(diffuse_tensor,
                                               dev_pd_params,
                                               dev_cv_params,
                                               width,
                                               height);

    /* Bind textures */
    checkCudaErrors(cudaBindTexture2D(0, diffuse_tensor_tex, diffuse_tensor.ptr,
                                      cudaCreateChannelDesc<float4>(),
                                      width, height, diffuse_tensor.pitch));

    checkCudaErrors(cudaBindTexture2D(0, head_primal_tex, head_primal.ptr,
                                      cudaCreateChannelDesc<float>(),
                                      width, height, head_primal.pitch));

    checkCudaErrors(cudaBindTexture2D(0, primal_tex, primal.ptr,
                                      cudaCreateChannelDesc<float>(),
                                      width, height, primal.pitch));

    checkCudaErrors(cudaBindTexture2D(0, old_primal_tex, old_primal.ptr,
                                      cudaCreateChannelDesc<float>(),
                                      width, height, old_primal.pitch));

    checkCudaErrors(cudaBindTexture2D(0, dual_tex, dual.ptr,
                                      cudaCreateChannelDesc<float2>(),
                                      width, height, dual.pitch));

    checkCudaErrors(cudaBindTexture2D(0,
                                      aux_tex,
                                      aux.ptr,
                                      cudaCreateChannelDesc<float>(),
                                      width,
                                      height,
                                      aux.pitch));

    checkCudaErrors(cudaBindTexture2D(0,
                                      primal_step_tex,
                                      primal_step.ptr,
                                      cudaCreateChannelDesc<float>(),
                                      width,
                                      height,
                                      primal_step.pitch));

    checkCudaErrors(cudaBindTexture2D(0,
                                      dual_step_tex,
                                      dual_step.ptr,
                                      cudaCreateChannelDesc<float2>(),
                                      width,
                                      height,
                                      dual_step.pitch));

    /* Do preconditioning on the linear operator (D and nabla) */
    DiffusionPrecondKernel<<<dimGrid, dimBlock>>>(primal_step,
                                                  dual_step,
                                                  dev_cv_params,
                                                  width,
                                                  height);

    float errors[host_pd_params.num_itr];
    float host_error_img[width*height];
    for(uint32_t i = 0; i < host_pd_params.num_itr; i++)
    {

        checkCudaErrors(cudaMemcpy2D(old_primal.ptr,
                                     old_primal.pitch,
                                     primal.ptr,
                                     primal.pitch,
                                     width*sizeof(Primal),
                                     height,
                                     cudaMemcpyDeviceToDevice));

        /* Dual update */
        HuberL2DualPrecondKernel<<<dimGrid, dimBlock>>>(dual,
                                                        dev_pd_params,
                                                        dev_cv_params,
                                                        width,
                                                        height);

        /* Primal update */
        HuberL2PrimalPrecondKernel<<<dimGrid, dimBlock>>>(primal,
                                                          dev_pd_params,
                                                          dev_cv_params,
                                                          width,
                                                          height);

        /* Head primal update */
        HuberL2HeadPrimalKernel<<<dimGrid, dimBlock>>>(head_primal,
                                                       dev_pd_params,
                                                       dev_cv_params,
                                                       width,
                                                       height);

        /* Pixel-wise line search in cost-volume */
        CostVolumePixelWiseSearch<<<dimGrid, dimBlock>>>(aux,
                                                         dev_pd_params,
                                                         dev_cv_params,
                                                         width,
                                                         height);

        host_pd_params.aux_theta = host_pd_params.aux_theta*(1.0 - host_pd_params.aux_theta_gamma*i);
        checkCudaErrors(cudaMemcpy(&dev_pd_params->aux_theta, &host_pd_params.aux_theta, sizeof(float), cudaMemcpyHostToDevice));

        /* Calculate point-wise error */        
        HuberL1CVErrorKernel<<<dimGrid, dimBlock>>>(error_img,
                                                    dev_pd_params,
                                                    dev_cv_params,
                                                    width,
                                                    height);

        checkCudaErrors(cudaMemcpy2D(host_error_img, width*sizeof(float),
                                     error_img.ptr, error_img.pitch,
                                     width*sizeof(float), height, cudaMemcpyDeviceToHost));

        errors[i] = 0.0;
        for(uint32_t e = 0; e < width*height; e++)
            errors[i] += host_error_img[e];

    }

    ////////////////////////////////////////////////////////////////////////////
    //  Copy results back to Matlab gpuArray
    ////////////////////////////////////////////////////////////////////////////
    size_t disp_dim[] = {width, height};
    mxGPUArray* disp_mat = mxGPUCreateGPUArray(2, disp_dim, mxSINGLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    checkCudaErrors(cudaMemcpy2D(mxGPUGetData(disp_mat), width*sizeof(float), min_disp.ptr, min_disp.pitch, width*sizeof(float), height, cudaMemcpyDeviceToDevice));
    plhs[0] = mxGPUCreateMxArrayOnGPU(disp_mat);

    size_t primal_dim[] = {width, height};
    mxGPUArray* primal_mat = mxGPUCreateGPUArray(2, primal_dim, mxSINGLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    checkCudaErrors(cudaMemcpy2D(mxGPUGetData(primal_mat), width*sizeof(Primal), primal.ptr, primal.pitch, width*sizeof(Primal), height, cudaMemcpyDeviceToDevice));
    plhs[1] = mxGPUCreateMxArrayOnGPU(primal_mat);

    size_t dual_dim[] = {2*width, height};
    mxGPUArray* dual_mat = mxGPUCreateGPUArray(2, dual_dim, mxSINGLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    checkCudaErrors(cudaMemcpy2D(mxGPUGetData(dual_mat), width*sizeof(Dual), dual.ptr, dual.pitch, width*sizeof(Dual), height, cudaMemcpyDeviceToDevice));
    plhs[2] = mxGPUCreateMxArrayOnGPU(dual_mat);

    size_t primal_step_dim[] = {width, height};
    mxGPUArray* primal_step_mat = mxGPUCreateGPUArray(2, primal_step_dim, mxSINGLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    checkCudaErrors(cudaMemcpy2D(mxGPUGetData(primal_step_mat), width*sizeof(PrimalStep), primal_step.ptr, primal_step.pitch, width*sizeof(PrimalStep), height, cudaMemcpyDeviceToDevice));
    plhs[3] = mxGPUCreateMxArrayOnGPU(primal_step_mat);

    size_t dual_step_dim[] = {2*width, height};
    mxGPUArray* dual_step_mat = mxGPUCreateGPUArray(2, dual_step_dim, mxSINGLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    checkCudaErrors(cudaMemcpy2D(mxGPUGetData(dual_step_mat), width*sizeof(DualStep), dual_step.ptr, dual_step.pitch, width*sizeof(DualStep), height, cudaMemcpyDeviceToDevice));
    plhs[4] = mxGPUCreateMxArrayOnGPU(dual_step_mat);

    plhs[5] = mxCreateNumericMatrix(host_pd_params.num_itr, 1, mxSINGLE_CLASS, mxREAL);
    copy(errors, errors+host_pd_params.num_itr, (float*)mxGetData(plhs[5]));

    /* Free CUDA memory */
    checkCudaErrors(cudaFree(min_disp.ptr));
    checkCudaErrors(cudaFree(min_disp_cost.ptr));
    checkCudaErrors(cudaFree(max_disp_cost.ptr));

    checkCudaErrors(cudaFree(primal.ptr));
    checkCudaErrors(cudaFree(old_primal.ptr));
    checkCudaErrors(cudaFree(head_primal.ptr));

    checkCudaErrors(cudaFree(dual.ptr));
    checkCudaErrors(cudaFree(aux.ptr));
    checkCudaErrors(cudaFree(diffuse_tensor.ptr));
    checkCudaErrors(cudaFree(error_img.ptr));

    checkCudaErrors(cudaFree(primal_step.ptr));
    checkCudaErrors(cudaFree(dual_step.ptr));

    checkCudaErrors(cudaFreeArray(cost_volume_array));
    checkCudaErrors(cudaFreeArray(left_img_array));
    checkCudaErrors(cudaFreeArray(right_img_array));
}
