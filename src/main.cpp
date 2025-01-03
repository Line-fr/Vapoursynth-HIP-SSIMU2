#include <stdlib.h>
#include <stdio.h>
#include<math.h>
#include "VapourSynth4.h"
#include "VSHelper4.h"
#include<vector>
#include <chrono>
#include <thread>


#ifdef __HIPCC__
    #include<hip/hip_runtime.h>
#elif defined __CUDACC__
    #define LOWLEVEL
    #define hipMemcpyDtoH(x, y, z) cudaMemcpy(x, y, z, cudaMemcpyDeviceToHost)
    #define hipMemcpyHtoD(x, y, z) cudaMemcpy(x, y, z, cudaMemcpyHostToDevice)
    #define hipMemcpyDtoHAsync(x, y, z, w) cudaMemcpyAsync(x, y, z, cudaMemcpyDeviceToHost, w)
    #define hipMemcpyHtoDAsync(x, y, z, w) cudaMemcpyAsync(x, y, z, cudaMemcpyHostToDevice, w)
    #define hipMemcpyPeer cudaMemcpyPeer
    #define hipMemcpyPeerAsync cudaMemcpyPeerAsync
    #define hipMalloc cudaMalloc
    #define hipFree cudaFree
    #define hipDeviceSynchronize cudaDeviceSynchronize
    #define hipSetDevice cudaSetDevice
    #define hipDeviceProp_t cudaDeviceProp
    #define hipGetDeviceCount cudaGetDeviceCount
    #define hipDeviceptr_t void*
    #define hipGetDevice cudaGetDevice
    #define hipGetDeviceProperties cudaGetDeviceProperties
    #define hipError_t cudaError_t
    #define hipGetErrorString cudaGetErrorString
    #define hipStream_t cudaStream_t
    #define hipStreamAddCallback cudaStreamAddCallback
    #define hipDeviceEnablePeerAccess cudaDeviceEnablePeerAccess
    #define hipSuccess cudaSuccess
    #define hipGetLastError cudaGetLastError
    #define hipStreamCreate cudaStreamCreate
    #define hipStreamDestroy cudaStreamDestroy
    #define hipEventCreate cudaEventCreate
    #define hipEventDestroy cudaEventDestroy
    #define hipEventSynchronize cudaEventSynchronize
    #define hipEventRecord cudaEventRecord
    #define hipEvent_t cudaEvent_t
    #define hipEventElapsedTime cudaEventElapsedTime
    #define hipDeviceSetCacheConfig cudaDeviceSetCacheConfig 
    #define hipFuncCachePreferShared cudaFuncCachePreferShared
    #define hipFuncCachePreferNone cudaFuncCachePreferNone
    #define hipFuncCachePreferL1 cudaFuncCachePreferL1
    #define hipFuncCachePreferEqual cudaFuncCachePreferEqual
    #define hipMemGetInfo cudaMemGetInfo
#endif


hipError_t err_hip;

#define GPU_CHECK(x)\
err_hip = (x);\
if (err_hip != hipSuccess)\
{\
   	printf("%s in %s at %d\n", hipGetErrorString(err_hip),  __FILE__, __LINE__);\
}

#define STREAMNUM 30
#define GAUSSIANSIZE 10
#define SIGMA 1.5f
#define PI 3.14159265359

#include "float3operations.hpp"
#include "downsample.hpp"
#include "makeXYB.hpp"
#include "gaussianblur.hpp"
#include "score.hpp"

__global__ void memoryorganizer_kernel(float3* out, const uint8_t *srcp0, const uint8_t *srcp1, const uint8_t *srcp2, int stride, int width, int height){
    size_t x = threadIdx.x + blockIdx.x*blockDim.x;
    if (x > width*height) return;
    int j = x%width;
    int i = x/width;
    out[i*width + j].x = ((float*)(srcp0 + i*stride))[j];
    out[i*width + j].y = ((float*)(srcp1 + i*stride))[j];
    out[i*width + j].z = ((float*)(srcp2 + i*stride))[j];
}

void memoryorganizer(float3* out, const uint8_t *srcp0, const uint8_t *srcp1, const uint8_t *srcp2, int stride, int width, int height, hipStream_t stream){
    int th_x = std::min(256, width*height);
    int bl_x = (width*height-1)/th_x + 1;
    memoryorganizer_kernel<<<dim3(bl_x), dim3(th_x), 0, stream>>>(out, srcp0, srcp1, srcp2, stride, width, height);
}

double ssimu2process(const uint8_t *srcp1[3], const uint8_t *srcp2[3], int stride, int width, int height, float* gaussiankernel, int maxshared, hipStream_t stream){

    int wh = width*height;
    int whs[6] = {wh, ((height-1)/2 + 1)*((width-1)/2 + 1), ((height-1)/4 + 1)*((width-1)/4 + 1), ((height-1)/8 + 1)*((width-1)/8 + 1), ((height-1)/16 + 1)*((width-1)/16 + 1), ((height-1)/32 + 1)*((width-1)/32 + 1)};
    int whs_integral[7];
    whs_integral[0] = 0;
    for (int i = 0; i < 7; i++){
        whs_integral[i+1] = whs_integral[i] + whs[i];
    }
    int totalscalesize = whs_integral[6];

    /*
    float3* srcs = (float3*)malloc(sizeof(float3)*wh * 2);

    for (int i = 0; i < height; i++){
        for (int j = 0; j < width; j++){
            srcs[i*width + j].x = ((float*)(srcp1[0] + i*stride))[j];
            srcs[i*width + j].y = ((float*)(srcp1[1] + i*stride))[j];
            srcs[i*width + j].z = ((float*)(srcp1[2] + i*stride))[j];
            srcs[wh + i*width + j].x = ((float*)(srcp2[0] + i*stride))[j];
            srcs[wh + i*width + j].y = ((float*)(srcp2[1] + i*stride))[j];
            srcs[wh + i*width + j].z = ((float*)(srcp2[2] + i*stride))[j];

            //printf("source is %f, %f, %f in %d, %d\n", srcs[i*width + j].x, srcs[i*width + j].y, srcs[i*width + j].z, j, i);
        }
    }*/

    //big memory allocation, we will try it multiple time if failed to save when too much threads are used
    hipError_t erralloc;
    int tries = 10;

    float3* mem_d;
    erralloc = hipMalloc(&mem_d, sizeof(float3)*totalscalesize*(2 + 6)); //2 base image and 6 working buffers
    while (erralloc != hipSuccess){
        std::this_thread::sleep_for(std::chrono::milliseconds(500)); //0.5s with 10 tries -> shut down after 5 seconds of failing
        erralloc = hipMalloc(&mem_d, sizeof(float3)*totalscalesize*(2 + 6)); //2 base image and 6 working buffers
        tries--;
        if (tries <= 0){
            printf("ERROR, could not allocate VRAM for a frame, try lowering the number of vapoursynth threads\n");
            return -10000.;
        }
    }
    //GPU_CHECK(hipGetLastError());

    float3* src1_d = mem_d; //length totalscalesize
    float3* src2_d = mem_d + totalscalesize;

    float3* temp_d = mem_d + 2*totalscalesize;
    float3* temps11_d = mem_d + 3*totalscalesize;
    float3* temps22_d = mem_d + 4*totalscalesize;
    float3* temps12_d = mem_d + 5*totalscalesize;
    float3* tempb1_d = mem_d + 6*totalscalesize;
    float3* tempb2_d = mem_d + 7*totalscalesize;

    hipEvent_t event_d;
    hipEventCreate(&event_d);

    uint8_t *memory_placeholder[3] = {(uint8_t*)temp_d, (uint8_t*)temp_d+stride*height, (uint8_t*)temp_d+2*stride*height};
    GPU_CHECK(hipMemcpyHtoDAsync(memory_placeholder[0], (void*)srcp1[0], stride * height, stream));
    GPU_CHECK(hipMemcpyHtoDAsync(memory_placeholder[1], (void*)srcp1[1], stride * height, stream));
    GPU_CHECK(hipMemcpyHtoDAsync(memory_placeholder[2], (void*)srcp1[2], stride * height, stream));
    memoryorganizer(src1_d, memory_placeholder[0], memory_placeholder[1], memory_placeholder[2], stride, width, height, stream);

    GPU_CHECK(hipMemcpyHtoDAsync(memory_placeholder[0], (void*)srcp2[0], stride * height, stream));
    GPU_CHECK(hipMemcpyHtoDAsync(memory_placeholder[1], (void*)srcp2[1], stride * height, stream));
    GPU_CHECK(hipMemcpyHtoDAsync(memory_placeholder[2], (void*)srcp2[2], stride * height, stream));
    memoryorganizer(src2_d, memory_placeholder[0], memory_placeholder[1], memory_placeholder[2], stride, width, height, stream);

    //step 1 : fill the downsample part
    int nw = width;
    int nh = height;
    for (int scale = 1; scale <= 5; scale++){
        downsample(src1_d+whs_integral[scale-1], src1_d+whs_integral[scale], nw, nh, stream);
        nw = (nw -1)/2 + 1;
        nh = (nh - 1)/2 + 1;
    }
    nw = width;
    nh = height;
    for (int scale = 1; scale <= 5; scale++){
        downsample(src2_d+whs_integral[scale-1], src2_d+whs_integral[scale], nw, nh, stream);
        nw = (nw -1)/2 + 1;
        nh = (nh - 1)/2 + 1;
    }

    //step 2 : positive XYB transition
    rgb_to_positive_xyb(src1_d, totalscalesize, stream);
    rgb_to_positive_xyb(src2_d, totalscalesize, stream);

    //step 3 : fill buffer s11 s22 and s12 and blur everything
    multarray(src1_d, src1_d, temps11_d, totalscalesize, stream);
    gaussianBlur(temps11_d, temps11_d, temp_d, totalscalesize, width, height, gaussiankernel, stream);

    multarray(src1_d, src2_d, temps12_d, totalscalesize, stream);
    gaussianBlur(temps12_d, temps12_d, temp_d, totalscalesize, width, height, gaussiankernel, stream);

    multarray(src2_d, src2_d, temps22_d, totalscalesize, stream);
    gaussianBlur(temps22_d, temps22_d, temp_d, totalscalesize, width, height, gaussiankernel, stream);

    gaussianBlur(src1_d, tempb1_d, temp_d, totalscalesize, width, height, gaussiankernel, stream);
    gaussianBlur(src2_d, tempb2_d, temp_d, totalscalesize, width, height, gaussiankernel, stream);

    //step 4 : get all scores from our matrices
    std::vector<float3> allscore_res = allscore_map(src1_d, src2_d, tempb1_d, tempb2_d, temps11_d, temps22_d, temps12_d, temp_d, width, height, maxshared, event_d, stream);

    //step 5 : format the vector
    std::vector<float> measure_vec(108);

    for (int plane = 0; plane < 3; plane++){
        for (int scale = 0; scale < 6; scale++){
            for (int n = 0; n < 2; n++){
                for (int i = 0; i < 3; i++){
                    if (plane == 0) measure_vec[plane*6*2*3 + scale*2*3 + n*3 + i] = allscore_res[scale*2*3 + i*2 + n].x;
                    if (plane == 1) measure_vec[plane*6*2*3 + scale*2*3 + n*3 + i] = allscore_res[scale*2*3 + i*2 + n].y;
                    if (plane == 2) measure_vec[plane*6*2*3 + scale*2*3 + n*3 + i] = allscore_res[scale*2*3 + i*2 + n].z;
                }
            }
        }
    }

    //step 6 : enjoy !
    const float ssim = final_score(measure_vec);

    hipEventRecord(event_d, stream); //place an event in the stream at the end of all our operations
    hipEventSynchronize(event_d); //when the event is complete, we know our gpu result is ready!

    hipFree(mem_d);
    hipEventDestroy(event_d);

    return ssim;
}

typedef struct {
    VSNode *reference;
    VSNode *distorted;
    int maxshared;
    float* gaussiankernel_d;
    hipStream_t streams[STREAMNUM];
    int oldthreadnum;
} Ssimulacra2Data;

static const VSFrame *VS_CC ssimulacra2GetFrame(int n, int activationReason, void *instanceData, void **frameData, VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi) {
    Ssimulacra2Data *d = (Ssimulacra2Data *)instanceData;

    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(n, d->reference, frameCtx);
        vsapi->requestFrameFilter(n, d->distorted, frameCtx);
    } else if (activationReason == arAllFramesReady) {
        const VSFrame *src1 = vsapi->getFrameFilter(n, d->reference, frameCtx);
        const VSFrame *src2 = vsapi->getFrameFilter(n, d->distorted, frameCtx);
        
        int height = vsapi->getFrameHeight(src1, 0);
        int width = vsapi->getFrameWidth(src1, 0);
        int stride = vsapi->getStride(src1, 0);

        VSFrame *dst = vsapi->copyFrame(src2, core);

        const uint8_t *srcp1[3] = {
            vsapi->getReadPtr(src1, 0),
            vsapi->getReadPtr(src1, 1),
            vsapi->getReadPtr(src1, 2),
        };

        const uint8_t *srcp2[3] = {
            vsapi->getReadPtr(src2, 0),
            vsapi->getReadPtr(src2, 1),
            vsapi->getReadPtr(src2, 2),
        };

        const double val = ssimu2process(srcp1, srcp2, stride, width, height, d->gaussiankernel_d, d->maxshared, d->streams[n%STREAMNUM]);

        vsapi->mapSetFloat(vsapi->getFramePropertiesRW(dst), "_SSIMULACRA2", val, maReplace);

        // Release the source frame
        vsapi->freeFrame(src1);
        vsapi->freeFrame(src2);

        // A reference is consumed when it is returned, so saving the dst reference somewhere
        // and reusing it is not allowed.
        return dst;
    }

    return NULL;
}

// Free all allocated data on filter destruction
static void VS_CC ssimulacra2Free(void *instanceData, VSCore *core, const VSAPI *vsapi) {
    Ssimulacra2Data *d = (Ssimulacra2Data *)instanceData;
    vsapi->freeNode(d->reference);
    vsapi->freeNode(d->distorted);

    for (int i = 0; i < STREAMNUM; i++){
        hipStreamDestroy(d->streams[i]);
    }
    hipFree(d->gaussiankernel_d);
    //vsapi->setThreadCount(d->oldthreadnum, core);

    free(d);
}

// This function is responsible for validating arguments and creating a new filter  
static void VS_CC ssimulacra2Create(const VSMap *in, VSMap *out, void *userData, VSCore *core, const VSAPI *vsapi) {
    Ssimulacra2Data d;
    Ssimulacra2Data *data;

    // Get a clip reference from the input arguments. This must be freed later.
    d.reference = vsapi->mapGetNode(in, "reference", 0, 0);
    d.distorted = vsapi->mapGetNode(in, "distorted", 0, 0);
    const VSVideoInfo *viref = vsapi->getVideoInfo(d.reference);
    const VSVideoInfo *vidis = vsapi->getVideoInfo(d.distorted);

    if (!(vsh::isSameVideoInfo(viref, vidis))){
        vsapi->mapSetError(out, "SSIMULACRA2: both clips must have the same format and dimensions");
        vsapi->freeNode(d.reference);
        vsapi->freeNode(d.distorted);
        return;
    }

    if ((viref->format.colorFamily != cfRGB) || viref->format.sampleType != stFloat){
        vsapi->mapSetError(out, "SSIMULACRA2: only works with RGBS format");
        vsapi->freeNode(d.reference);
        vsapi->freeNode(d.distorted);
        return;
    }

    int count;
    if (hipGetDeviceCount(&count) != 0){
        vsapi->mapSetError(out, "could not detect devices, check gpu permissions\n");
    };
    if (count == 0){
        vsapi->mapSetError(out, "No GPU was found on the system for a given compilation type. Try switch nvidia/amd binary\n");
    }

    hipDeviceSetCacheConfig(hipFuncCachePreferNone);
    int device;
    hipDeviceProp_t devattr;
    hipGetDevice(&device);
    hipGetDeviceProperties(&devattr, device);

    //int videowidth = viref->width;
    //int videoheight = viref->height;
    //put optimal thread number
    //VSCoreInfo infos;
    //vsapi->getCoreInfo(core, &infos);
    //d.oldthreadnum = infos.numThreads;
    //size_t freemem, totalmem;
    //hipMemGetInfo (&freemem, &totalmem);

    //vsapi->setThreadCount(std::min((int)((float)(freemem - 20*(1llu << 20))/(8*sizeof(float3)*videowidth*videoheight*(1.33333))), d.oldthreadnum), core);

    for (int i = 0; i < STREAMNUM; i++){
        hipStreamCreate(d.streams + i);
    }

    data = (Ssimulacra2Data *)malloc(sizeof(d));
    *data = d;

    for (int i = 0; i < STREAMNUM; i++){
        data->streams[i] = d.streams[i];
    }

    float gaussiankernel[2*GAUSSIANSIZE+1];
    for (int i = 0; i < 2*GAUSSIANSIZE+1; i++){
        gaussiankernel[i] = std::exp(-(GAUSSIANSIZE-i)*(GAUSSIANSIZE-i)/(2*SIGMA*SIGMA))/(std::sqrt(2*PI*SIGMA*SIGMA));
    }

    data->maxshared = devattr.sharedMemPerBlock;    

    hipMalloc(&(data->gaussiankernel_d), sizeof(float)*(2*GAUSSIANSIZE+1));
    hipMemcpyHtoD((hipDeviceptr_t)data->gaussiankernel_d, gaussiankernel, (2*GAUSSIANSIZE+1)*sizeof(float));

    VSFilterDependency deps[] = {{d.reference, rpStrictSpatial}, {d.distorted, rpStrictSpatial}};

    vsapi->createVideoFilter(out, "vship", viref, ssimulacra2GetFrame, ssimulacra2Free, fmParallel, deps, 2, data, core);
}


VS_EXTERNAL_API(void) VapourSynthPluginInit2(VSPlugin *plugin, const VSPLUGINAPI *vspapi) {
    vspapi->configPlugin("com.lumen.vship", "vship", "VapourSynth SSIMULACRA2 on GPU", VS_MAKE_VERSION(3, 0), VAPOURSYNTH_API_VERSION, 0, plugin);
    vspapi->registerFunction("SSIMULACRA2", "reference:vnode;distorted:vnode;", "clip:vnode;", ssimulacra2Create, NULL, plugin);
}
