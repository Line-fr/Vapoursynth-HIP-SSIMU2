#include <stdlib.h>
#include <stdio.h>
#include<math.h>
#include "VapourSynth4.h"
#include "VSHelper4.h"

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
#endif

#define STREAMNUM 30

#include "float3operations.hpp"
#include "downsample.hpp"
#include "makeXYB.hpp"

double ssimu2process(const uint8_t *srcp1[3], const uint8_t *srcp2[3], int stride, int width, int height, hipStream_t stream){

    int wh = width*height;
    int whs[6] = {wh, ((height-1)/2 + 1)*((width-1)/2 + 1), ((height-1)/4 + 1)*((width-1)/4 + 1), ((height-1)/8 + 1)*((width-1)/8 + 1), ((height-1)/16 + 1)*((width-1)/16 + 1), ((height-1)/32 + 1)*((width-1)/32 + 1)};
    int whs_integral[7];
    whs_integral[0] = 0;
    for (int i = 0; i < 7; i++){
        whs_integral[i+1] = whs_integral[i] + whs[i];
    }
    int totalscalesize = whs_integral[6];

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
    }

    float3* mem_d;
    hipMalloc(&mem_d, sizeof(float3)*totalscalesize*(2 + 6)); //2 base image and 6 working buffers
    
    float3* src1_d = mem_d; //length totalscalesize
    float3* src2_d = mem_d + totalscalesize;

    float3* temp_d = mem_d + 2*totalscalesize;
    float3* temps11_d = mem_d + 3*totalscalesize;
    float3* temps22_d = mem_d + 4*totalscalesize;
    float3* temps12_d = mem_d + 5*totalscalesize;
    float3* tempb1_d = mem_d + 6*totalscalesize;
    float3* tempb2_d = mem_d + 7*totalscalesize;


    hipEvent_t event_d, startevent_d;
    hipEventCreate(&event_d);
    hipEventCreate(&startevent_d);
    hipEventRecord(startevent_d, stream);

    hipMemcpyHtoDAsync((hipDeviceptr_t)src1_d, (void*)srcs, sizeof(float3)*wh, stream);
    hipMemcpyHtoDAsync((hipDeviceptr_t)src2_d, (void*)(srcs+wh), sizeof(float3)*wh, stream);
    
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

    //step 3 : fill buffer s11 s22 and s12

    //step 4 : blur everything

    //step 5 : ssim map

    //step 6 : edge diff map    

    //step 7 : get back the score

    //step 8 : enjoy !


    hipEventRecord(event_d, stream); //place an event in the stream at the end of all our operations
    hipEventSynchronize(event_d); //when the event is complete, we know our gpu result is ready!

    float time;
    hipEventElapsedTime (&time, startevent_d, event_d);

    //printf("I took %f ms\n", time);

    free(srcs);
    hipFree(mem_d);
    hipEventDestroy(event_d);
    hipEventDestroy(startevent_d);

    return 0;
}

typedef struct {
    VSNode *reference;
    VSNode *distorted;
    hipStream_t streams[STREAMNUM];
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

        const double val = ssimu2process(srcp1, srcp2, stride, width, height, d->streams[n%STREAMNUM]);

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

    for (int i = 0; i < STREAMNUM; i++){
        hipStreamCreate(d.streams + i);
    }

    data = (Ssimulacra2Data *)malloc(sizeof(d));
    *data = d;

    for (int i = 0; i < STREAMNUM; i++){
        data->streams[i] = d.streams[i];
    }

    VSFilterDependency deps[] = {{d.reference, rpStrictSpatial}, {d.distorted, rpStrictSpatial}};

    vsapi->createVideoFilter(out, "vship", viref, ssimulacra2GetFrame, ssimulacra2Free, fmParallel, deps, 2, data, core);
}


VS_EXTERNAL_API(void) VapourSynthPluginInit2(VSPlugin *plugin, const VSPLUGINAPI *vspapi) {
    vspapi->configPlugin("com.lumen.vship", "vship", "VapourSynth SSIMULACRA2 on GPU", VS_MAKE_VERSION(3, 0), VAPOURSYNTH_API_VERSION, 0, plugin);
    vspapi->registerFunction("SSIMULACRA2", "reference:vnode;distorted:vnode;", "clip:vnode;", ssimulacra2Create, NULL, plugin);
}
