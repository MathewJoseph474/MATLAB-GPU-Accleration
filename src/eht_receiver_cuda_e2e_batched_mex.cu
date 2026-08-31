// eht_receiver_cuda_e2e_batched_mex.cu
//
// MATLAB command API:
//   ticket = eht_receiver_cuda_async_mex('submit', ...
//       iSegments,qSegments,quantScales,csiSegments, ...
//       noiseVariance,llrScale,validationMode[,referencePayloadBits])
//
//   stats = eht_receiver_cuda_async_mex('collect',ticket)
//   ready = eht_receiver_cuda_async_mex('poll',ticket)
//   info  = eht_receiver_cuda_async_mex('status')
//   eht_receiver_cuda_async_mex('reset')
//
// To-do 
// Part 2 development command:
//   rx = eht_receiver_cuda_e2e_batched_mex('reconstruct',RxI,RxQ,RxScale)
//
// Part 3 development command:
//   offset = eht_receiver_cuda_e2e_batched_mex('packetDetect',RxI,RxQ,RxScale)
//
// Part 4 development command:
//   cfoHz = eht_receiver_cuda_e2e_batched_mex('coarseCFO',RxI,RxQ,RxScale)
//
// Part 5 development command:
//   finalOffset = eht_receiver_cuda_e2e_batched_mex(
//       'timingSync',RxI,RxQ,RxScale,lltfReference)
//   fineCFOHz = eht_receiver_cuda_e2e_batched_mex(
//       'fineCFO',RxI,RxQ,RxScale,lltfReference)
//     Runs reconstruction, packet detection, coarse CFO correction, fine
//     timing, then L-LTF fine CFO estimation/correction fully on the GPU.
//   [noiseVar,rxEHTLTF,rxEHTData] = eht_receiver_cuda_e2e_batched_mex(
//       'part7',RxI,RxQ,RxScale,lltfReference)
//     Adds GPU L-LTF noise estimation and EHT field extraction.
//
// Part 8 development command:
//   stats = eht_receiver_cuda_e2e_batched_mex('e2e',...)
//     Runs the raw INT16 waveform through synchronization and the existing
//     CUDA FFT/channel/equalizer/QAM/LDPC/validation backend in one MEX call.
//
// 'submit' stages MATLAB inputs into persistent pinned host memory, enqueues
// H2D copies and all CUDA kernels on one of two nonblocking streams, and
// returns without synchronizing the stream.
//
// 'collect' waits only for the requested ticket, copies no bulk arrays, and
// returns scalar counters/timings.
//
// Fixed configuration:
//   EHT 320 MHz, MCS 13, 4096-QAM, LDPC rate 5/6,


#include "mex.h"
#include "matrix.h"
#include <cuda_runtime.h>
#include <cufft.h>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>
#include <string>
#include <algorithm>

#include "eht_post_equalizer_fixed_config.h"

#define CUFFT_CHECK(call) do { \
    cufftResult err__ = (call); \
    if (err__ != CUFFT_SUCCESS) { \
        mexErrMsgIdAndTxt("eht_receiver_cuda_async:CUFFT", \
            "%s failed with cuFFT error %d",#call,int(err__)); \
    } \
} while (0)

#define CUDA_CHECK(call) do { \
    cudaError_t err__ = (call); \
    if (err__ != cudaSuccess) { \
        mexErrMsgIdAndTxt("eht_receiver_cuda_async:CUDA", \
            "%s failed: %s", #call, cudaGetErrorString(err__)); \
    } \
} while (0)

namespace {

using namespace eht_fixed;

constexpr int Z = 81;
constexpr int BASE_ROWS = 4;
constexpr int BASE_COLS = 24;
constexpr int N = 1944;
constexpr int K = 1620;
constexpr int M = 324;
constexpr int BASE_EDGES = 79;
constexpr int EDGES = BASE_EDGES * Z;
constexpr float SHORTEN_LLR = 1.2676506002282294e30f;
constexpr int kSymbolsPerSegment =
    kTonesPerSegment * kNumOFDMSymbols;
constexpr int kTotalSymbols =
    kNumSegments * kSymbolsPerSegment;
constexpr int kNumSlots = 4;
constexpr int kFFTLength = 4096;
constexpr int kNumDataSymbols = kNumOFDMSymbols;
constexpr int kActiveToneCount = 3984;
constexpr int kDataToneCount = 3920;
constexpr int kLTFInputCount = 4096;
constexpr int kDataInputCount = kNumDataSymbols*kFFTLength;

struct SharedDeviceData {
    uint32_t* encodedSourceIndex = nullptr;
    int* payloadBits = nullptr;
    int* punctureBits = nullptr;
    int* repeatBits = nullptr;

    int* ltfFFTIndices = nullptr;
    int* dataFFTIndices = nullptr;
    int* pilotIndices = nullptr;
    int* dataIndices = nullptr;
    cufftComplex* knownLTF = nullptr;
    cufftComplex* pilotReference = nullptr;

    int pilotCount = 0;
    bool frontendMetadataInitialized = false;
    bool initialized = false;
};

struct Slot {
    cufftComplex* hLTF = nullptr;
    cufftComplex* hData = nullptr;
    int8_t* hReferenceBits = nullptr;
    unsigned long long* hChecksum = nullptr;
    unsigned int* hBitErrors = nullptr;

    cufftComplex* dLTFInput = nullptr;
    cufftComplex* dLTFOutput = nullptr;
    cufftComplex* dDataInput = nullptr;
    cufftComplex* dDataOutput = nullptr;
    cufftComplex* dLTFActive = nullptr;
    cufftComplex* dDataActive = nullptr;
    cufftComplex* dChannelEstimate = nullptr;
    cufftComplex* dPilotRotation = nullptr;
    cufftComplex* dEqualizedActive = nullptr;

    int16_t* dI = nullptr;
    int16_t* dQ = nullptr;
    float* dCSI = nullptr;
    float* dInverseQuantScales = nullptr;
    int8_t* dExternalLLR = nullptr;
    float* dEncoded = nullptr;
    float* dBeliefs = nullptr;
    float* dMessages = nullptr;
    int8_t* dDecoded = nullptr;
    int8_t* dScrambleSequence = nullptr;
    int* dDescrambleEnabled = nullptr;
    unsigned long long* dChecksum = nullptr;
    unsigned int* dBitErrors = nullptr;
    int8_t* dReferenceBits = nullptr;

    cudaStream_t stream = nullptr;
    cufftHandle ltfPlan = 0;
    cufftHandle dataPlan = 0;

    cudaEvent_t start = nullptr;
    cudaEvent_t afterH2D = nullptr;
    cudaEvent_t afterFrontend = nullptr;
    cudaEvent_t afterDemap = nullptr;
    cudaEvent_t afterMap = nullptr;
    cudaEvent_t afterLDPC = nullptr;
    cudaEvent_t afterValidate = nullptr;
    cudaEvent_t done = nullptr;

    bool occupied = false;
    uint64_t ticket = 0;
    int validationMode = 0;
};
// Four asynchronous slots for the complete batched E2E receiver.
// Batch capacity is allocated dynamically from the configured MATLAB batch size.
// Nothing hardcodes packet count or batch size.
struct BatchedSlot {

   
    // Slot state

    bool occupied = false;
    uint64_t ticket = 0;

    mwSize batchCapacity = 0;
    mwSize activeBatchSize = 0;
    mwSize sampleCount = 0;

    
    // CUDa stream
    cudaStream_t stream = nullptr;

    
    // Pinned host staging buffers
    // Required so MATLAB input can be copied into persistent memory and the MEX call can return while the GPU continues processing.
  
    int16_t* hRxI = nullptr;
    int16_t* hRxQ = nullptr;
    float* hInverseScales = nullptr;
    cufftComplex* hLLTFReference = nullptr;

    unsigned long long* hChecksums = nullptr;


    // Raw waveform / front-end buffers

    int16_t* dRxI = nullptr;
    int16_t* dRxQ = nullptr;

    cufftComplex* dWaveform = nullptr;
    cufftComplex* dLLTFReference = nullptr;

    float* dInverseScales = nullptr;

    int* dDetectedOffsets = nullptr;
    float* dCoarseCFO = nullptr;
    float* dTimingMetrics = nullptr;
    int* dTimingCorrection = nullptr;
    int* dFinalOffsets = nullptr;
    float* dFineCFO = nullptr;
    float* dNoiseVariance = nullptr;

  
    // Extracted EHT fields
  
    cufftComplex* dEHTLTF = nullptr;
    cufftComplex* dEHTData = nullptr;


    // FFT input/output
   
   cufftComplex* dLTFInput = nullptr;
    cufftComplex* dLTFOutput = nullptr;

    cufftComplex* dDataInput = nullptr;
    cufftComplex* dDataOutput = nullptr;

    cufftHandle ltfPlan = 0;
    cufftHandle dataPlan = 0;
    mwSize fftPlanBatchSize = 0;

    
    // Channel / equalizer

    cufftComplex* dLTFActive = nullptr;
    cufftComplex* dDataActive = nullptr;
    cufftComplex* dChannelEstimate = nullptr;
    cufftComplex* dPilotRotation = nullptr;
    cufftComplex* dEqualized = nullptr;

    
    // QAM / LLR

    int16_t* dQuantizedI = nullptr;
    int16_t* dQuantizedQ = nullptr;

    float* dCSI = nullptr;
    float* dInverseQuantScale = nullptr;

    int8_t* dExternalLLR = nullptr;
    float* dEncoded = nullptr;

    
    // LDPC

    float* dBeliefs = nullptr;
    float* dMessages = nullptr;
    int8_t* dDecoded = nullptr;

  
    // Descrambling / payload recovery

    int8_t* dScrambleSequence = nullptr;
    int8_t* dDescrambled = nullptr;
    int* dDescrambleEnabled = nullptr;

    unsigned long long* dChecksums = nullptr;

  
    // Per-slot 12-stage CUDA timing events

    cudaEvent_t evPacketDetectStart = nullptr;
    cudaEvent_t evPacketDetectEnd = nullptr;

    cudaEvent_t evCoarseCFOEnd = nullptr;
    cudaEvent_t evTimingSyncEnd = nullptr;
    cudaEvent_t evFineCFOEnd = nullptr;

    cudaEvent_t evOFDMStart = nullptr;
    cudaEvent_t evOFDMEnd = nullptr;

    cudaEvent_t evChannelEstimationEnd = nullptr;
    cudaEvent_t evEqualizationEnd = nullptr;
    cudaEvent_t evQAMDemappingEnd = nullptr;
    cudaEvent_t evLLRGenerationEnd = nullptr;
    cudaEvent_t evLDPCDecodingEnd = nullptr;
    cudaEvent_t evDescramblingEnd = nullptr;
    cudaEvent_t evPayloadRecoveryEnd = nullptr;

    // Marks the point where the complete buffer has finished.
    cudaEvent_t done = nullptr;

   
    // Completed timing values

    float packetDetectionMs = 0.0f;
    float coarseCFOMs = 0.0f;
    float timingSyncMs = 0.0f;
    float fineCFOMs = 0.0f;

    float ofdmMs = 0.0f;
    float channelEstimationMs = 0.0f;
    float equalizationMs = 0.0f;
    float qamDemappingMs = 0.0f;
    float llrGenerationMs = 0.0f;
    float ldpcDecodingMs = 0.0f;
    float descramblingMs = 0.0f;
    float payloadRecoveryMs = 0.0f;
};

BatchedSlot batchedSlots[kNumSlots];
uint64_t nextBatchedTicket = 1;

SharedDeviceData sharedData;
Slot slots[kNumSlots];
bool initialized = false;
uint64_t nextTicket = 1;

void freeSlotResources(Slot& s) {
    if (s.stream && s.occupied)
        cudaStreamSynchronize(s.stream);

    if (s.hLTF) cudaFreeHost(s.hLTF);
    if (s.hData) cudaFreeHost(s.hData);
    if (s.hReferenceBits) cudaFreeHost(s.hReferenceBits);
    if (s.hChecksum) cudaFreeHost(s.hChecksum);
    if (s.hBitErrors) cudaFreeHost(s.hBitErrors);

    if (s.dLTFInput) cudaFree(s.dLTFInput);
    if (s.dLTFOutput) cudaFree(s.dLTFOutput);
    if (s.dDataInput) cudaFree(s.dDataInput);
    if (s.dDataOutput) cudaFree(s.dDataOutput);
    if (s.dLTFActive) cudaFree(s.dLTFActive);
    if (s.dDataActive) cudaFree(s.dDataActive);
    if (s.dChannelEstimate) cudaFree(s.dChannelEstimate);
    if (s.dPilotRotation) cudaFree(s.dPilotRotation);
    if (s.dEqualizedActive) cudaFree(s.dEqualizedActive);

    if (s.dI) cudaFree(s.dI);
    if (s.dQ) cudaFree(s.dQ);
    if (s.dCSI) cudaFree(s.dCSI);
    if (s.dInverseQuantScales) cudaFree(s.dInverseQuantScales);
    if (s.dExternalLLR) cudaFree(s.dExternalLLR);
    if (s.dEncoded) cudaFree(s.dEncoded);
    if (s.dBeliefs) cudaFree(s.dBeliefs);
    if (s.dMessages) cudaFree(s.dMessages);
    if (s.dDecoded) cudaFree(s.dDecoded);
    if (s.dScrambleSequence) cudaFree(s.dScrambleSequence);
    if (s.dDescrambleEnabled) cudaFree(s.dDescrambleEnabled);
    if (s.dChecksum) cudaFree(s.dChecksum);
    if (s.dBitErrors) cudaFree(s.dBitErrors);
    if (s.dReferenceBits) cudaFree(s.dReferenceBits);

    if (s.start) cudaEventDestroy(s.start);
    if (s.afterH2D) cudaEventDestroy(s.afterH2D);
    if (s.afterFrontend) cudaEventDestroy(s.afterFrontend);
    if (s.afterDemap) cudaEventDestroy(s.afterDemap);
    if (s.afterMap) cudaEventDestroy(s.afterMap);
    if (s.afterLDPC) cudaEventDestroy(s.afterLDPC);
    if (s.afterValidate) cudaEventDestroy(s.afterValidate);
    if (s.done) cudaEventDestroy(s.done);
    if (s.ltfPlan) cufftDestroy(s.ltfPlan);
    if (s.dataPlan) cufftDestroy(s.dataPlan);
    if (s.stream) cudaStreamDestroy(s.stream);

    s = Slot{};
}

void freeBatchedSlotResources(BatchedSlot& s)
{
    // Wait only for this slot if it still has work in flight.
    if (s.stream && s.occupied) {
        CUDA_CHECK(cudaStreamSynchronize(s.stream));
    }

    // Pinned host buffers.
    if (s.hRxI) cudaFreeHost(s.hRxI);
    if (s.hRxQ) cudaFreeHost(s.hRxQ);
    if (s.hInverseScales) cudaFreeHost(s.hInverseScales);
    if (s.hLLTFReference) cudaFreeHost(s.hLLTFReference);
    if (s.hChecksums) cudaFreeHost(s.hChecksums);

    // Front-end device buffers.
    if (s.dRxI) cudaFree(s.dRxI);
    if (s.dRxQ) cudaFree(s.dRxQ);
    if (s.dWaveform) cudaFree(s.dWaveform);
    if (s.dLLTFReference) cudaFree(s.dLLTFReference);
    if (s.dInverseScales) cudaFree(s.dInverseScales);
    if (s.dDetectedOffsets) cudaFree(s.dDetectedOffsets);
    if (s.dCoarseCFO) cudaFree(s.dCoarseCFO);
    if (s.dTimingMetrics) cudaFree(s.dTimingMetrics);
    if (s.dTimingCorrection) cudaFree(s.dTimingCorrection);
    if (s.dFinalOffsets) cudaFree(s.dFinalOffsets);
    if (s.dFineCFO) cudaFree(s.dFineCFO);
    if (s.dNoiseVariance) cudaFree(s.dNoiseVariance);

    // EHT fields / FFT.
    if (s.dEHTLTF) cudaFree(s.dEHTLTF);
    if (s.dEHTData) cudaFree(s.dEHTData);
    if (s.dLTFInput) cudaFree(s.dLTFInput);
    if (s.dLTFOutput) cudaFree(s.dLTFOutput);
    if (s.dDataInput) cudaFree(s.dDataInput);
    if (s.dDataOutput) cudaFree(s.dDataOutput);

    // Channel / equalizer.
    if (s.dLTFActive) cudaFree(s.dLTFActive);
    if (s.dDataActive) cudaFree(s.dDataActive);
    if (s.dChannelEstimate) cudaFree(s.dChannelEstimate);
    if (s.dPilotRotation) cudaFree(s.dPilotRotation);
    if (s.dEqualized) cudaFree(s.dEqualized);

    // QAM / LLR.
    if (s.dQuantizedI) cudaFree(s.dQuantizedI);
    if (s.dQuantizedQ) cudaFree(s.dQuantizedQ);
    if (s.dCSI) cudaFree(s.dCSI);
    if (s.dInverseQuantScale) cudaFree(s.dInverseQuantScale);
    if (s.dExternalLLR) cudaFree(s.dExternalLLR);
    if (s.dEncoded) cudaFree(s.dEncoded);

    // LDPC.
    if (s.dBeliefs) cudaFree(s.dBeliefs);
    if (s.dMessages) cudaFree(s.dMessages);
    if (s.dDecoded) cudaFree(s.dDecoded);

    // Descrambling / payload recovery.
    if (s.dScrambleSequence) cudaFree(s.dScrambleSequence);
    if (s.dDescrambled) cudaFree(s.dDescrambled);
    if (s.dDescrambleEnabled) cudaFree(s.dDescrambleEnabled);
    if (s.dChecksums) cudaFree(s.dChecksums);

    // CUDA timing events.
    if (s.evPacketDetectStart) cudaEventDestroy(s.evPacketDetectStart);
    if (s.evPacketDetectEnd) cudaEventDestroy(s.evPacketDetectEnd);
    if (s.evCoarseCFOEnd) cudaEventDestroy(s.evCoarseCFOEnd);
    if (s.evTimingSyncEnd) cudaEventDestroy(s.evTimingSyncEnd);
    if (s.evFineCFOEnd) cudaEventDestroy(s.evFineCFOEnd);
    if (s.evOFDMStart) cudaEventDestroy(s.evOFDMStart);
    if (s.evOFDMEnd) cudaEventDestroy(s.evOFDMEnd);
    if (s.evChannelEstimationEnd) cudaEventDestroy(s.evChannelEstimationEnd);
    if (s.evEqualizationEnd) cudaEventDestroy(s.evEqualizationEnd);
    if (s.evQAMDemappingEnd) cudaEventDestroy(s.evQAMDemappingEnd);
    if (s.evLLRGenerationEnd) cudaEventDestroy(s.evLLRGenerationEnd);
    if (s.evLDPCDecodingEnd) cudaEventDestroy(s.evLDPCDecodingEnd);
    if (s.evDescramblingEnd) cudaEventDestroy(s.evDescramblingEnd);
    if (s.evPayloadRecoveryEnd) cudaEventDestroy(s.evPayloadRecoveryEnd);
    if (s.done) cudaEventDestroy(s.done);

    // cuFFT plans / stream.
    if (s.ltfPlan) cufftDestroy(s.ltfPlan);
    if (s.dataPlan) cufftDestroy(s.dataPlan);
    if (s.stream) cudaStreamDestroy(s.stream);

    s = BatchedSlot{};
}

void allocateBatchedSlot(
    BatchedSlot& s,
    mwSize sampleCount,
    mwSize batchCapacity)
{
    if (sampleCount == 0 || batchCapacity == 0) {
        mexErrMsgIdAndTxt(
            "eht_receiver_cuda_e2e:BatchedSlot",
            "sampleCount and batchCapacity must both be positive.");
    }

    // If this slot was previously allocated, release it first.
    freeBatchedSlotResources(s);

    s.sampleCount = sampleCount;
    s.batchCapacity = batchCapacity;
    s.activeBatchSize = 0;

    const size_t B = size_t(batchCapacity);
    const size_t samples = size_t(sampleCount);
    const size_t totalSamples = samples * B;

    // One independent nonblocking CUDA stream per buffer.
    CUDA_CHECK(cudaStreamCreateWithFlags(&s.stream,cudaStreamNonBlocking));

    // CUDA timing events.
    CUDA_CHECK(cudaEventCreate(&s.evPacketDetectStart));
    CUDA_CHECK(cudaEventCreate(&s.evPacketDetectEnd));
    CUDA_CHECK(cudaEventCreate(&s.evCoarseCFOEnd));
    CUDA_CHECK(cudaEventCreate(&s.evTimingSyncEnd));
    CUDA_CHECK(cudaEventCreate(&s.evFineCFOEnd));
    CUDA_CHECK(cudaEventCreate(&s.evOFDMStart));
    CUDA_CHECK(cudaEventCreate(&s.evOFDMEnd));
    CUDA_CHECK(cudaEventCreate(&s.evChannelEstimationEnd));
    CUDA_CHECK(cudaEventCreate(&s.evEqualizationEnd));
    CUDA_CHECK(cudaEventCreate(&s.evQAMDemappingEnd));
    CUDA_CHECK(cudaEventCreate(&s.evLLRGenerationEnd));
    CUDA_CHECK(cudaEventCreate(&s.evLDPCDecodingEnd));
    CUDA_CHECK(cudaEventCreate(&s.evDescramblingEnd));
    CUDA_CHECK(cudaEventCreate(&s.evPayloadRecoveryEnd));
    CUDA_CHECK(cudaEventCreateWithFlags(&s.done,cudaEventDisableTiming));

    // Pinned host staging buffers.
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&s.hRxI),totalSamples*sizeof(int16_t)));
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&s.hRxQ),totalSamples*sizeof(int16_t)));
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&s.hInverseScales),B*sizeof(float)));
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&s.hLLTFReference),size_t(2560)*sizeof(cufftComplex)));
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&s.hChecksums),B*sizeof(unsigned long long)));

    // Raw input and synchronization.
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&s.dRxI),totalSamples*sizeof(int16_t)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&s.dRxQ),totalSamples*sizeof(int16_t)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&s.dWaveform),totalSamples*sizeof(cufftComplex)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&s.dLLTFReference),size_t(2560)*sizeof(cufftComplex)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&s.dInverseScales),B*sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&s.dDetectedOffsets),B*sizeof(int)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&s.dCoarseCFO),B*sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&s.dTimingMetrics),B*size_t(513)*sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&s.dTimingCorrection),B*sizeof(int)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&s.dFinalOffsets),B*sizeof(int)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&s.dFineCFO),B*sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&s.dNoiseVariance),B*sizeof(float)));

    // Extracted EHT fields.
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&s.dEHTLTF),B*size_t(kEHTLTFLength)*sizeof(cufftComplex)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&s.dEHTData),B*size_t(kEHTDataLength)*sizeof(cufftComplex)));

    // FFT buffers.
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&s.dLTFInput),B*size_t(kFFTLength)*sizeof(cufftComplex)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&s.dLTFOutput),B*size_t(kFFTLength)*sizeof(cufftComplex)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&s.dDataInput),B*size_t(kDataInputCount)*sizeof(cufftComplex)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&s.dDataOutput),B*size_t(kDataInputCount)*sizeof(cufftComplex)));

    // Channel estimation / equalization.
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&s.dLTFActive),B*size_t(kActiveToneCount)*sizeof(cufftComplex)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&s.dDataActive),B*size_t(kActiveToneCount)*size_t(kNumDataSymbols)*sizeof(cufftComplex)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&s.dChannelEstimate),B*size_t(kActiveToneCount)*sizeof(cufftComplex)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&s.dPilotRotation),B*size_t(kNumDataSymbols)*sizeof(cufftComplex)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&s.dEqualized),B*size_t(kActiveToneCount)*size_t(kNumDataSymbols)*sizeof(cufftComplex)));

    // Quantization / QAM / LLR.
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&s.dQuantizedI),B*size_t(kTotalSymbols)*sizeof(int16_t)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&s.dQuantizedQ),B*size_t(kTotalSymbols)*sizeof(int16_t)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&s.dCSI),B*size_t(kDataToneCount)*sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&s.dInverseQuantScale),B*size_t(kNumSegments)*sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&s.dExternalLLR),B*size_t(kTotalSymbols)*size_t(kBitsPerSymbol)*sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&s.dEncoded),B*size_t(kEncodedLength)*sizeof(float)));

    // LDPC.
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&s.dBeliefs),B*size_t(kNumCodewords)*size_t(N)*sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&s.dMessages),B*size_t(kNumCodewords)*size_t(EDGES)*sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&s.dDecoded),B*size_t(kTotalDecodedBits)*sizeof(int8_t)));

    // Descrambling / payload recovery.
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&s.dScrambleSequence),B*size_t(2047)*sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&s.dDescrambled),B*size_t(kTotalDecodedBits)*sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&s.dDescrambleEnabled),B*sizeof(int)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&s.dChecksums),B*sizeof(unsigned long long)));

    // FFT plans are intentionally created later for the active batch size,
    // so partial final batches continue to work correctly.
    s.fftPlanBatchSize = 0;
}

// Ensure that this batched slot owns cuFFT plans sized for the current active batch. The configured maximum batch size is NOT hardcoded.
// A smaller final partial batch gets plans sized for that actual batch.
void ensureBatchedFFTPlans(
    BatchedSlot& s,
    mwSize activeBatchSize)
{
    if (activeBatchSize == 0 || activeBatchSize > s.batchCapacity) {
        mexErrMsgIdAndTxt(
            "eht_receiver_cuda_e2e:BatchedFFT",
            "Invalid active batch size for batched FFT plans.");
    }

    // Plans are only changed while the slot is free.
    if (s.occupied) {
        mexErrMsgIdAndTxt(
            "eht_receiver_cuda_e2e:BatchedFFT",
            "Cannot recreate FFT plans while a batched slot is occupied.");
    }

    // Reuse the existing plans whenever the active batch size is unchanged.
    if (s.ltfPlan != 0 &&
        s.dataPlan != 0 &&
        s.fftPlanBatchSize == activeBatchSize) {
        return;
    }

    if (s.ltfPlan) {
        CUFFT_CHECK(cufftDestroy(s.ltfPlan));
        s.ltfPlan = 0;
    }

    if (s.dataPlan) {
        CUFFT_CHECK(cufftDestroy(s.dataPlan));
        s.dataPlan = 0;
    }

    int n[1] = {kFFTLength};
    int embed[1] = {kFFTLength};

    const int activeBatch = static_cast<int>(activeBatchSize);

    // One 4096-point LTF FFT per packet in the active batch.
    CUFFT_CHECK(cufftPlanMany(
        &s.ltfPlan,
        1,
        n,
        embed,
        1,
        kFFTLength,
        embed,
        1,
        kFFTLength,
        CUFFT_C2C,
        activeBatch));

    // kNumDataSymbols 4096-point data FFTs per packet.
    CUFFT_CHECK(cufftPlanMany(
        &s.dataPlan,
        1,
        n,
        embed,
        1,
        kFFTLength,
        embed,
        1,
        kFFTLength,
        CUFFT_C2C,
        activeBatch * kNumDataSymbols));

    // Each buffer/slot keeps its FFT work on its own nonblocking CUDA stream.
    CUFFT_CHECK(cufftSetStream(s.ltfPlan,s.stream));
    CUFFT_CHECK(cufftSetStream(s.dataPlan,s.stream));

    s.fftPlanBatchSize = activeBatchSize;
}

void cleanup() {
    for (int i = 0; i < kNumSlots; ++i) {
        freeBatchedSlotResources(batchedSlots[i]);
        freeSlotResources(slots[i]);
    }

    if (sharedData.encodedSourceIndex)
        cudaFree(sharedData.encodedSourceIndex);
    if (sharedData.payloadBits)
        cudaFree(sharedData.payloadBits);
    if (sharedData.punctureBits)
        cudaFree(sharedData.punctureBits);
    if (sharedData.repeatBits)
        cudaFree(sharedData.repeatBits);
    if (sharedData.ltfFFTIndices)
        cudaFree(sharedData.ltfFFTIndices);
    if (sharedData.dataFFTIndices)
        cudaFree(sharedData.dataFFTIndices);
    if (sharedData.pilotIndices)
        cudaFree(sharedData.pilotIndices);
    if (sharedData.dataIndices)
        cudaFree(sharedData.dataIndices);
    if (sharedData.knownLTF)
        cudaFree(sharedData.knownLTF);
    if (sharedData.pilotReference)
        cudaFree(sharedData.pilotReference);

    sharedData = SharedDeviceData{};
    initialized = false;
    nextTicket = 1;
    nextBatchedTicket = 1;
}

void allocateSlot(Slot& s) {
    CUDA_CHECK(cudaStreamCreateWithFlags(
        &s.stream,cudaStreamNonBlocking));

    CUDA_CHECK(cudaEventCreate(&s.start));
    CUDA_CHECK(cudaEventCreate(&s.afterH2D));
    CUDA_CHECK(cudaEventCreate(&s.afterFrontend));
    CUDA_CHECK(cudaEventCreate(&s.afterDemap));
    CUDA_CHECK(cudaEventCreate(&s.afterMap));
    CUDA_CHECK(cudaEventCreate(&s.afterLDPC));
    CUDA_CHECK(cudaEventCreate(&s.afterValidate));
    CUDA_CHECK(cudaEventCreateWithFlags(
        &s.done,cudaEventDisableTiming));

    CUDA_CHECK(cudaMallocHost(
        reinterpret_cast<void**>(&s.hLTF),
        size_t(kLTFInputCount)*sizeof(cufftComplex)));
    CUDA_CHECK(cudaMallocHost(
        reinterpret_cast<void**>(&s.hData),
        size_t(kDataInputCount)*sizeof(cufftComplex)));
    CUDA_CHECK(cudaMallocHost(
        reinterpret_cast<void**>(&s.hReferenceBits),
        size_t(kPayloadBits)*sizeof(int8_t)));
    CUDA_CHECK(cudaMallocHost(
        reinterpret_cast<void**>(&s.hChecksum),
        sizeof(unsigned long long)));
    CUDA_CHECK(cudaMallocHost(
        reinterpret_cast<void**>(&s.hBitErrors),
        sizeof(unsigned int)));

    CUFFT_CHECK(cufftPlan1d(
        &s.ltfPlan,kFFTLength,CUFFT_C2C,1));

    int n[1] = {kFFTLength};
    int embed[1] = {kFFTLength};
    CUFFT_CHECK(cufftPlanMany(
        &s.dataPlan,1,n,
        embed,1,kFFTLength,
        embed,1,kFFTLength,
        CUFFT_C2C,kNumDataSymbols));

    CUFFT_CHECK(cufftSetStream(s.ltfPlan,s.stream));
    CUFFT_CHECK(cufftSetStream(s.dataPlan,s.stream));

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&s.dLTFInput),
        size_t(kLTFInputCount)*sizeof(cufftComplex)));
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&s.dLTFOutput),
        size_t(kFFTLength)*sizeof(cufftComplex)));
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&s.dDataInput),
        size_t(kDataInputCount)*sizeof(cufftComplex)));
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&s.dDataOutput),
        size_t(kDataInputCount)*sizeof(cufftComplex)));
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&s.dLTFActive),
        size_t(kActiveToneCount)*sizeof(cufftComplex)));
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&s.dDataActive),
        size_t(kActiveToneCount*kNumDataSymbols)*sizeof(cufftComplex)));
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&s.dChannelEstimate),
        size_t(kActiveToneCount)*sizeof(cufftComplex)));
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&s.dPilotRotation),
        size_t(kNumDataSymbols)*sizeof(cufftComplex)));
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&s.dEqualizedActive),
        size_t(kActiveToneCount*kNumDataSymbols)*sizeof(cufftComplex)));

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&s.dI),
        size_t(kTotalSymbols)*sizeof(int16_t)));
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&s.dQ),
        size_t(kTotalSymbols)*sizeof(int16_t)));
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&s.dCSI),
        size_t(kNumSegments*kTonesPerSegment)*sizeof(float)));
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&s.dInverseQuantScales),
        size_t(kNumSegments)*sizeof(float)));
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&s.dExternalLLR),
        size_t(kTotalExternalElements)*sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&s.dEncoded),
        size_t(kEncodedLength)*sizeof(float)));
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&s.dBeliefs),
        size_t(kNumCodewords*N)*sizeof(float)));
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&s.dMessages),
        size_t(kNumCodewords*EDGES)*sizeof(float)));
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&s.dDecoded),
        size_t(kTotalDecodedBits)*sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&s.dScrambleSequence),
        size_t(2047)*sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&s.dDescrambleEnabled),
        sizeof(int)));
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&s.dChecksum),
        sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&s.dBitErrors),
        sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&s.dReferenceBits),
        size_t(kPayloadBits)*sizeof(int8_t)));
}

void ensureInitialized() {
    if (initialized) return;

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&sharedData.encodedSourceIndex),
        size_t(kEncodedLength)*sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&sharedData.payloadBits),
        size_t(kNumCodewords)*sizeof(int)));
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&sharedData.punctureBits),
        size_t(kNumCodewords)*sizeof(int)));
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&sharedData.repeatBits),
        size_t(kNumCodewords)*sizeof(int)));
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&sharedData.ltfFFTIndices),
        size_t(kActiveToneCount)*sizeof(int)));
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&sharedData.dataFFTIndices),
        size_t(kActiveToneCount)*sizeof(int)));
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&sharedData.dataIndices),
        size_t(kDataToneCount)*sizeof(int)));
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&sharedData.knownLTF),
        size_t(kActiveToneCount)*sizeof(cufftComplex)));

    CUDA_CHECK(cudaMemcpy(
        sharedData.encodedSourceIndex,kEncodedSourceGlobalIndex,
        size_t(kEncodedLength)*sizeof(uint32_t),
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        sharedData.payloadBits,kVecPayloadBits,
        size_t(kNumCodewords)*sizeof(int),
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        sharedData.punctureBits,kVecPunctureBits,
        size_t(kNumCodewords)*sizeof(int),
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        sharedData.repeatBits,kVecRepeatBits,
        size_t(kNumCodewords)*sizeof(int),
        cudaMemcpyHostToDevice));

    for (int i = 0; i < kNumSlots; ++i)
        allocateSlot(slots[i]);

    mexAtExit(cleanup);
    initialized = true;
}


__global__ void reconstructWaveformKernel(
    const int16_t* rxI,
    const int16_t* rxQ,
    cufftComplex* output,
    float inverseScale,
    int count)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= count)
        return;

    output[i].x = float(rxI[i]) * inverseScale;
    output[i].y = float(rxQ[i]) * inverseScale;
}

__global__ void packetDetectKernel(
    const cufftComplex* rx,
    int count,
    int* detectedOffset)
{
    // At 320 MHz, the legacy short-training sequence repeats every
    // 0.8 us = 256 samples.  Compute a normalized one-period
    // autocorrelation metric for each possible starting sample.
    // Match wlanPacketDetect's default coarse-detection threshold (0.5).
    constexpr int lag = 256;
    constexpr float threshold = 0.50f;

    int n = int(blockIdx.x)*blockDim.x + threadIdx.x;
    if (n + 2*lag > count)
        return;

    float corrRe = 0.0f;
    float corrIm = 0.0f;
    float energyA = 0.0f;
    float energyB = 0.0f;

    for (int k = 0; k < lag; ++k) {
        const cufftComplex a = rx[n+k];
        const cufftComplex b = rx[n+lag+k];

        // a * conj(b)
        corrRe += a.x*b.x + a.y*b.y;
        corrIm += a.y*b.x - a.x*b.y;
        energyA += a.x*a.x + a.y*a.y;
        energyB += b.x*b.x + b.y*b.y;
    }

    const float numerator = corrRe*corrRe + corrIm*corrIm;
    const float denominator = energyA*energyB + 1.0e-20f;
    const float metric = numerator/denominator;

    if (metric >= threshold)
        atomicMin(detectedOffset,n);
}

__global__ void coarseCFOCorrelationKernel(
    const cufftComplex* rx,
    int detectedOffset,
    int count,
    float* partialRe,
    float* partialIm)
{
    // L-STF length for CBW320 is 2560 samples.  Its short-training
    // sequence repeats every 0.8 microsec = 256 samples.
    constexpr int lSTFLength = 2560;
    constexpr int lag = 256;
    constexpr int pairCount = lSTFLength-lag;

    int i = int(blockIdx.x)*blockDim.x + threadIdx.x;
    if (i >= pairCount)
        return;

    const int aIndex = detectedOffset+i;
    const int bIndex = aIndex+lag;
    if (bIndex >= count) {
        partialRe[i] = 0.0f;
        partialIm[i] = 0.0f;
        return;
    }

    const cufftComplex a = rx[aIndex];
    const cufftComplex b = rx[bIndex];

    // conj(a)*b. Positive input CFO therefore produces positive phase.
    partialRe[i] = a.x*b.x + a.y*b.y;
    partialIm[i] = a.x*b.y - a.y*b.x;
}

__global__ void coarseCFOCorrectKernel(
    cufftComplex* rx,
    int detectedOffset,
    int count,
    float cfoHz,
    float sampleRateHz)
{
    int i = int(blockIdx.x)*blockDim.x + threadIdx.x;
    const int sample = detectedOffset+i;
    if (sample >= count)
        return;

    // Match applyFrequencyOffsetLocal(...,-coarseCFOHz): rotate the
    // detected waveform by exp(-j*2*pi*cfo*n/Fs), with n=0 at detection.
    constexpr float twoPi = 6.2831853071795864769f;
    const float phase = -twoPi*cfoHz*float(i)/sampleRateHz;
    const float c = cosf(phase);
    const float sn = sinf(phase);
    const cufftComplex x = rx[sample];
    rx[sample].x = x.x*c - x.y*sn;
    rx[sample].y = x.x*sn + x.y*c;
}

// Part 5 GPU-only coarse-CFO reduction used by timingSync. The Part 4
// diagnostic command remains unchanged, but this path avoids returning the
// intermediate CFO to MATLAB before timing synchronization.
__global__ void coarseCFOFinalizeKernel(
    const cufftComplex* rx,
    const int* detectedOffset,
    int count,
    float* cfoHz)
{
    constexpr int lSTFLength = 2560;
    constexpr int lag = 256;
    constexpr int pairCount = lSTFLength-lag;
    constexpr float sampleRateHz = 320.0e6f;
    constexpr float twoPi = 6.2831853071795864769f;

    __shared__ float sumRe[256];
    __shared__ float sumIm[256];

    const int tid = int(threadIdx.x);
    const int offset = *detectedOffset;

    float localRe = 0.0f;
    float localIm = 0.0f;

    if (offset < count && offset+lSTFLength <= count) {
        for (int i = tid; i < pairCount; i += int(blockDim.x)) {
            const cufftComplex a = rx[offset+i];
            const cufftComplex b = rx[offset+i+lag];
            localRe += a.x*b.x + a.y*b.y;
            localIm += a.x*b.y - a.y*b.x;
        }
    }

    sumRe[tid] = localRe;
    sumIm[tid] = localIm;
    __syncthreads();

    for (int stride = int(blockDim.x)/2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sumRe[tid] += sumRe[tid+stride];
            sumIm[tid] += sumIm[tid+stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        if (offset >= count || offset+lSTFLength > count) {
            *cfoHz = 0.0f;
        } else {
            const float phase = atan2f(sumIm[0],sumRe[0]);
            *cfoHz = phase*sampleRateHz/(twoPi*float(lag));
        }
    }
}

__global__ void coarseCFOCorrectPointerKernel(
    cufftComplex* rx,
    const int* detectedOffset,
    int count,
    const float* cfoHz)
{
    const int offset = *detectedOffset;
    const int i = int(blockIdx.x)*blockDim.x + threadIdx.x;
    const int sample = offset+i;
    if (offset >= count || sample >= count)
        return;

    constexpr float sampleRateHz = 320.0e6f;
    constexpr float twoPi = 6.2831853071795864769f;
    const float phase = -twoPi*(*cfoHz)*float(i)/sampleRateHz;
    const float c = cosf(phase);
    const float sn = sinf(phase);
    const cufftComplex x = rx[sample];
    rx[sample].x = x.x*c - x.y*sn;
    rx[sample].y = x.x*sn + x.y*c;
}

__global__ void timingMetricKernel(
    const cufftComplex* rx,
    const cufftComplex* lltfReference,
    const int* detectedOffset,
    int count,
    float referenceEnergy,
    float* metrics)
{
    constexpr int lSTFLength = 2560;
    constexpr int lltfLength = 2560;
    constexpr int maxTimingCorrection = 512;

    const int correction = int(blockIdx.x)*blockDim.x + threadIdx.x;
    if (correction > maxTimingCorrection)
        return;

    const int coarse = *detectedOffset;
    const int start = coarse + correction + lSTFLength;

    if (coarse >= count || start < 0 || start+lltfLength > count) {
        metrics[correction] = -1.0f;
        return;
    }

    float corrRe = 0.0f;
    float corrIm = 0.0f;
    float rxEnergy = 0.0f;

    for (int k = 0; k < lltfLength; ++k) {
        const cufftComplex x = rx[start+k];
        const cufftComplex r = lltfReference[k];

        // conj(r)*x
        corrRe += r.x*x.x + r.y*x.y;
        corrIm += r.x*x.y - r.y*x.x;
        rxEnergy += x.x*x.x + x.y*x.y;
    }

    const float numerator = corrRe*corrRe + corrIm*corrIm;
    const float denominator =
        referenceEnergy*rxEnergy + 1.0e-20f;
    metrics[correction] = numerator/denominator;
}

__global__ void timingArgMaxKernel(
    const float* metrics,
    const int* detectedOffset,
    int* timingCorrection,
    int* finalPacketOffset)
{
    if (blockIdx.x != 0 || threadIdx.x != 0)
        return;

    constexpr int candidateCount = 513;
    float bestMetric = -1.0f;
    int bestCorrection = 0;

    for (int i = 0; i < candidateCount; ++i) {
        const float metric = metrics[i];
        if (metric > bestMetric) {
            bestMetric = metric;
            bestCorrection = i;
        }
    }

    *timingCorrection = bestCorrection;
    *finalPacketOffset = *detectedOffset + bestCorrection;
}

// Part 7 synchronization consistency: after fine timing is known, we should re-reference
// the coarse-CFO correction phase to the final packet boundary. MATLAB first
// extracts the aligned packet and then applies coarse CFO with n=0 at
// that final boundary. The GPU coarse correction originally used n=0 at the
// coarse detector offset, which leaves a constant phase rotation.
__global__ void rebaseCoarseCFOPhaseKernel(
    cufftComplex* rx,
    const int* finalPacketOffset,
    const int* timingCorrection,
    int count,
    const float* coarseCFOHz)
{
    const int packetStart = *finalPacketOffset;
    const int i = int(blockIdx.x)*blockDim.x + threadIdx.x;
    const int sample = packetStart+i;
    if (packetStart < 0 || sample >= count)
        return;

    constexpr float sampleRateHz = 320.0e6f;
    constexpr float twoPi = 6.2831853071795864769f;

    // Undo the constant phase accumulated between the coarse detector start
    // and the final timing boundary.
    const float phase = twoPi*(*coarseCFOHz)*float(*timingCorrection)/sampleRateHz;
    const float c = cosf(phase);
    const float sn = sinf(phase);
    const cufftComplex x = rx[sample];
    rx[sample].x = x.x*c - x.y*sn;
    rx[sample].y = x.x*sn + x.y*c;
}

// Part 6: fine CFO from the two repeated 3.2 microsec L-LTF symbols.
// For CBW320, L-LTF is 2560 samples: 512-sample GI followed by
// two identical 1024-sample long-training symbols.
__global__ void fineCFOFinalizeKernel(
    const cufftComplex* rx,
    const int* finalPacketOffset,
    int count,
    float* fineCFOHz)
{
    constexpr int lltfStart = 2560;
    constexpr int lltfCP = 512;
    constexpr int symbolLength = 1024;
    constexpr float sampleRateHz = 320.0e6f;
    constexpr float twoPi = 6.2831853071795864769f;

    __shared__ float sumRe[256];
    __shared__ float sumIm[256];

    const int tid = int(threadIdx.x);
    const int packetStart = *finalPacketOffset;
    const int first = packetStart + lltfStart + lltfCP;
    const int second = first + symbolLength;

    float localRe = 0.0f;
    float localIm = 0.0f;

    if (packetStart >= 0 && second + symbolLength <= count) {
        for (int i = tid; i < symbolLength; i += int(blockDim.x)) {
            const cufftComplex a = rx[first+i];
            const cufftComplex b = rx[second+i];
            // conj(a)*b: positive residual CFO gives positive phase.
            localRe += a.x*b.x + a.y*b.y;
            localIm += a.x*b.y - a.y*b.x;
        }
    }

    sumRe[tid] = localRe;
    sumIm[tid] = localIm;
    __syncthreads();

    for (int stride = int(blockDim.x)/2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sumRe[tid] += sumRe[tid+stride];
            sumIm[tid] += sumIm[tid+stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        if (packetStart < 0 || second + symbolLength > count) {
            *fineCFOHz = 0.0f;
        } else {
            const float phase = atan2f(sumIm[0],sumRe[0]);
            *fineCFOHz = phase*sampleRateHz/(twoPi*float(symbolLength));
        }
    }
}

__global__ void fineCFOCorrectPointerKernel(
    cufftComplex* rx,
    const int* finalPacketOffset,
    int count,
    const float* fineCFOHz)
{
    const int packetStart = *finalPacketOffset;
    const int i = int(blockIdx.x)*blockDim.x + threadIdx.x;
    const int sample = packetStart+i;
    if (packetStart < 0 || sample >= count)
        return;

    constexpr float sampleRateHz = 320.0e6f;
    constexpr float twoPi = 6.2831853071795864769f;
    const float phase = -twoPi*(*fineCFOHz)*float(i)/sampleRateHz;
    const float c = cosf(phase);
    const float sn = sinf(phase);
    const cufftComplex x = rx[sample];
    rx[sample].x = x.x*c - x.y*sn;
    rx[sample].y = x.x*sn + x.y*c;
}



// Part 7: estimate mean noise power from the two repeated L-LTF symbols.
// The signal component cancels in the symbol-to-symbol difference; dividing
// by two converts the difference power back to per-symbol noise power.
__global__ void noiseVarianceFinalizeKernel(
    const cufftComplex* rx,
    const int* finalPacketOffset,
    int count,
    float* noiseVariance)
{
    constexpr int lltfStart = 2560;
    constexpr int lltfCP = 512;
    constexpr int symbolLength = 1024;
    // The generated WLAN waveform uses a short transition/windowing region
    // at the end of the L-LTF symbol. Comparing that tail directly inflates
    // a time-domain repeat-difference estimate, so ignore the final 16 samples.
    constexpr int usableLength = symbolLength-16;

    __shared__ float sumPower[256];
    const int tid = int(threadIdx.x);
    const int packetStart = *finalPacketOffset;
    const int first = packetStart + lltfStart + lltfCP;
    const int second = first + symbolLength;

    float local = 0.0f;
    if (packetStart >= 0 && second + symbolLength <= count) {
        for (int i = tid; i < usableLength; i += int(blockDim.x)) {
            const cufftComplex a = rx[first+i];
            const cufftComplex b = rx[second+i];
            const float dr = a.x-b.x;
            const float di = a.y-b.y;
            local += dr*dr + di*di;
        }
    }

    sumPower[tid] = local;
    __syncthreads();

    for (int stride = int(blockDim.x)/2; stride > 0; stride >>= 1) {
        if (tid < stride)
            sumPower[tid] += sumPower[tid+stride];
        __syncthreads();
    }

    if (tid == 0) {
        if (packetStart < 0 || second + symbolLength > count)
            *noiseVariance = 0.0f;
        else
            *noiseVariance = sumPower[0]/(2.0f*float(usableLength));
    }
}

__global__ void extractEHTFieldsKernel(
    const cufftComplex* rx,
    const int* finalPacketOffset,
    int count,
    cufftComplex* ehtLTF,
    cufftComplex* ehtData)
{
    constexpr int ehtLTFStart = kEHTLTFStart;
    constexpr int ehtLTFLength = kEHTLTFLength;
    constexpr int ehtDataStart = kEHTDataStart;
    constexpr int ehtDataLength = kEHTDataLength;

    const int i = int(blockIdx.x)*blockDim.x + threadIdx.x;
    const int packetStart = *finalPacketOffset;
    if (packetStart < 0)
        return;

    if (i < ehtLTFLength) {
        const int source = packetStart + ehtLTFStart + i;
        if (source < count)
            ehtLTF[i] = rx[source];
    }
    if (i < ehtDataLength) {
        const int source = packetStart + ehtDataStart + i;
        if (source < count)
            ehtData[i] = rx[source];
    }
}


// Part 10B 1: batched raw-waveform front-end.  Packet is the
// CUDA grid Y dimension; samples/candidates are X.  All intermediate arrays
// are laid out packet-major with a fixed stride.

__global__ void reconstructWaveformBatchKernel(const int16_t* rxI,const int16_t* rxQ,
    cufftComplex* output,const float* inverseScales,int count,int batchSize)
{
    int i=int(blockIdx.x)*blockDim.x+threadIdx.x; int b=int(blockIdx.y);
    if (b>=batchSize || i>=count) return; size_t j=size_t(b)*count+i;
    float sc=inverseScales[b]; output[j].x=float(rxI[j])*sc; output[j].y=float(rxQ[j])*sc;
}

__global__ void packetDetectBatchKernel(const cufftComplex* rx,int count,int* offsets,int batchSize)
{
    constexpr int lag=256; constexpr float threshold=0.50f;
    int n=int(blockIdx.x)*blockDim.x+threadIdx.x; int b=int(blockIdx.y);
    if (b>=batchSize || n+2*lag>count) return;
    const cufftComplex* x=rx+size_t(b)*count;
    float cr=0,ci=0,ea=0,eb=0;
    for(int k=0;k<lag;++k){auto a=x[n+k];auto d=x[n+lag+k];cr+=a.x*d.x+a.y*d.y;ci+=a.y*d.x-a.x*d.y;ea+=a.x*a.x+a.y*a.y;eb+=d.x*d.x+d.y*d.y;}
    float m=(cr*cr+ci*ci)/(ea*eb+1e-20f); if(m>=threshold) atomicMin(offsets+b,n);
}

__global__ void coarseCFOBatchKernel(const cufftComplex* rx,const int* offsets,int count,float* cfo,int batchSize)
{
    constexpr int len=2560,lag=256,pairs=len-lag; constexpr float Fs=320e6f,twoPi=6.2831853071795864769f;
    int b=int(blockIdx.x); if(b>=batchSize) return; int tid=int(threadIdx.x); __shared__ float sr[256],si[256];
    int off=offsets[b]; const cufftComplex* x=rx+size_t(b)*count; float r=0,im=0;
    if(off<count && off+len<=count) for(int i=tid;i<pairs;i+=blockDim.x){auto a=x[off+i];auto d=x[off+i+lag];r+=a.x*d.x+a.y*d.y;im+=a.x*d.y-a.y*d.x;}
    sr[tid]=r;si[tid]=im;__syncthreads(); for(int st=blockDim.x/2;st;st>>=1){if(tid<st){sr[tid]+=sr[tid+st];si[tid]+=si[tid+st];}__syncthreads();}
    if(tid==0)cfo[b]=(off>=count||off+len>count)?0.0f:atan2f(si[0],sr[0])*Fs/(twoPi*lag);
}

__global__ void cfoCorrectBatchKernel(cufftComplex* rx,const int* starts,int count,const float* cfo,int batchSize)
{
    constexpr float Fs=320e6f,twoPi=6.2831853071795864769f; int i=int(blockIdx.x)*blockDim.x+threadIdx.x,b=int(blockIdx.y); if(b>=batchSize)return;
    int off=starts[b],sample=off+i; if(off>=count||sample>=count)return; auto* x=rx+size_t(b)*count; float ph=-twoPi*cfo[b]*float(i)/Fs,c=cosf(ph),sn=sinf(ph); auto v=x[sample];x[sample].x=v.x*c-v.y*sn;x[sample].y=v.x*sn+v.y*c;
}

__global__ void timingMetricBatchKernel(const cufftComplex* rx,const cufftComplex* ref,const int* coarse,int count,float refEnergy,float* metrics,int batchSize)
{
    constexpr int lstf=2560,lltf=2560,maxc=512; int cidx=int(blockIdx.x)*blockDim.x+threadIdx.x,b=int(blockIdx.y); if(b>=batchSize||cidx>maxc)return;
    int off=coarse[b],start=off+cidx+lstf; float* mout=metrics+size_t(b)*513; if(off>=count||start<0||start+lltf>count){mout[cidx]=-1;return;} const auto* x=rx+size_t(b)*count;
    float cr=0,ci=0,e=0;for(int k=0;k<lltf;++k){auto v=x[start+k],r=ref[k];cr+=r.x*v.x+r.y*v.y;ci+=r.x*v.y-r.y*v.x;e+=v.x*v.x+v.y*v.y;}mout[cidx]=(cr*cr+ci*ci)/(refEnergy*e+1e-20f);
}

__global__ void timingArgMaxBatchKernel(const float* metrics,const int* coarse,int* corr,int* finalOff,int batchSize)
{
    int b=int(blockIdx.x);if(b>=batchSize||threadIdx.x)return; const float* m=metrics+size_t(b)*513;float best=-1;int bc=0;for(int i=0;i<513;++i)if(m[i]>best){best=m[i];bc=i;}corr[b]=bc;finalOff[b]=coarse[b]+bc;
}

__global__ void rebaseBatchKernel(cufftComplex* rx,const int* finalOff,const int* corr,int count,const float* coarseCFO,int batchSize)
{
    constexpr float Fs=320e6f,twoPi=6.2831853071795864769f;int i=int(blockIdx.x)*blockDim.x+threadIdx.x,b=int(blockIdx.y);if(b>=batchSize)return;int st=finalOff[b],sample=st+i;if(st<0||sample>=count)return;auto* x=rx+size_t(b)*count;float ph=twoPi*coarseCFO[b]*float(corr[b])/Fs,c=cosf(ph),sn=sinf(ph);auto v=x[sample];x[sample].x=v.x*c-v.y*sn;x[sample].y=v.x*sn+v.y*c;
}

__global__ void fineCFOBatchKernel(const cufftComplex* rx,const int* finalOff,int count,float* fine,int batchSize)
{
    constexpr int ls=2560,cp=512,L=1024;constexpr float Fs=320e6f,twoPi=6.2831853071795864769f;int b=int(blockIdx.x);if(b>=batchSize)return;int tid=threadIdx.x;__shared__ float sr[256],si[256];int st=finalOff[b],a0=st+ls+cp,b0=a0+L;const auto*x=rx+size_t(b)*count;float r=0,im=0;if(st>=0&&b0+L<=count)for(int i=tid;i<L;i+=blockDim.x){auto a=x[a0+i],d=x[b0+i];r+=a.x*d.x+a.y*d.y;im+=a.x*d.y-a.y*d.x;}sr[tid]=r;si[tid]=im;__syncthreads();for(int q=blockDim.x/2;q;q>>=1){if(tid<q){sr[tid]+=sr[tid+q];si[tid]+=si[tid+q];}__syncthreads();}if(tid==0)fine[b]=(st<0||b0+L>count)?0.0f:atan2f(si[0],sr[0])*Fs/(twoPi*L);
}

__global__ void noiseBatchKernel(const cufftComplex* rx,const int* finalOff,int count,float* noise,int batchSize)
{
    constexpr int ls=2560,cp=512,L=1024,U=L-16;int b=blockIdx.x;if(b>=batchSize)return;int tid=threadIdx.x;__shared__ float sp[256];int st=finalOff[b],a0=st+ls+cp,b0=a0+L;const auto*x=rx+size_t(b)*count;float v=0;if(st>=0&&b0+L<=count)for(int i=tid;i<U;i+=blockDim.x){auto a=x[a0+i],d=x[b0+i];float dr=a.x-d.x,di=a.y-d.y;v+=dr*dr+di*di;}sp[tid]=v;__syncthreads();for(int q=blockDim.x/2;q;q>>=1){if(tid<q)sp[tid]+=sp[tid+q];__syncthreads();}if(tid==0)noise[b]=(st<0||b0+L>count)?0.0f:sp[0]/(2.0f*U);
}

__global__ void extractFieldsBatchKernel(const cufftComplex* rx,const int* finalOff,int count,cufftComplex* ltf,cufftComplex* data,int batchSize)
{
    constexpr int lstart=kEHTLTFStart,llen=kEHTLTFLength,dstart=kEHTDataStart,dlen=kEHTDataLength;int i=int(blockIdx.x)*blockDim.x+threadIdx.x,b=int(blockIdx.y);if(b>=batchSize)return;int st=finalOff[b];if(st<0)return;const auto*x=rx+size_t(b)*count;if(i<llen&&st+lstart+i<count)ltf[size_t(b)*llen+i]=x[st+lstart+i];if(i<dlen&&st+dstart+i<count)data[size_t(b)*dlen+i]=x[st+dstart+i];
}

__global__ void extractActiveKernel(
    const cufftComplex* __restrict__ ltfFFT,
    const cufftComplex* __restrict__ dataFFT,
    const int* __restrict__ ltfIndices,
    const int* __restrict__ dataIndices,
    cufftComplex* __restrict__ ltfActive,
    cufftComplex* __restrict__ dataActive)
{
    int i = int(blockIdx.x)*blockDim.x + threadIdx.x;
    if (i >= kActiveToneCount) return;

    const int ltfBin = ltfIndices[i];
    const int dataBin = dataIndices[i];

    ltfActive[i] = ltfFFT[ltfBin];
    for (int sym = 0; sym < kNumDataSymbols; ++sym) {
        dataActive[size_t(sym)*kActiveToneCount+i] =
            dataFFT[size_t(sym)*kFFTLength+dataBin];
    }
}

__global__ void estimateChannelKernel(
    const cufftComplex* __restrict__ ltfActive,
    const cufftComplex* __restrict__ knownLTF,
    cufftComplex* __restrict__ channel)
{
    int i = int(blockIdx.x)*blockDim.x + threadIdx.x;
    if (i >= kActiveToneCount) return;
    const cufftComplex y = ltfActive[i];
    const cufftComplex r = knownLTF[i];
    channel[i].x = y.x*r.x + y.y*r.y;
    channel[i].y = y.y*r.x - y.x*r.y;
}

__global__ void estimatePilotRotationKernel(
    const cufftComplex* __restrict__ dataActive,
    const cufftComplex* __restrict__ channel,
    const int* __restrict__ pilotIndices,
    const cufftComplex* __restrict__ pilotReference,
    int pilotCount,
    cufftComplex* __restrict__ rotation)
{
    int symbol = blockIdx.x;
    extern __shared__ cufftComplex partial[];
    cufftComplex sum{0.0f,0.0f};

    for (int p = threadIdx.x; p < pilotCount; p += blockDim.x) {
        int tone = pilotIndices[p];
        cufftComplex y = dataActive[symbol*kActiveToneCount+tone];
        cufftComplex h = channel[tone];
        cufftComplex r = pilotReference[symbol*pilotCount+p];

        cufftComplex expected;
        expected.x = h.x*r.x-h.y*r.y;
        expected.y = h.x*r.y+h.y*r.x;

        sum.x += y.x*expected.x+y.y*expected.y;
        sum.y += y.y*expected.x-y.x*expected.y;
    }

    partial[threadIdx.x] = sum;
    __syncthreads();

    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x].x += partial[threadIdx.x+stride].x;
            partial[threadIdx.x].y += partial[threadIdx.x+stride].y;
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        float mag = sqrtf(
            partial[0].x*partial[0].x+
            partial[0].y*partial[0].y);
        rotation[symbol] = mag > 0.0f
            ? cufftComplex{partial[0].x/mag,-partial[0].y/mag}
            : cufftComplex{1.0f,0.0f};
    }
}

__global__ void equalizeKernel(
    const cufftComplex* __restrict__ dataActive,
    const cufftComplex* __restrict__ channel,
    const cufftComplex* __restrict__ rotation,
    float noiseVariance,
    cufftComplex* __restrict__ equalized)
{
    int i = int(blockIdx.x)*blockDim.x + threadIdx.x;
    if (i >= kActiveToneCount*kNumDataSymbols) return;

    int tone = i % kActiveToneCount;
    int symbol = i / kActiveToneCount;

    cufftComplex in = dataActive[i];
    cufftComplex rot = rotation[symbol];
    cufftComplex y{
        in.x*rot.x-in.y*rot.y,
        in.x*rot.y+in.y*rot.x};

    cufftComplex h = channel[tone];
    float denom = h.x*h.x+h.y*h.y+noiseVariance;

    equalized[i].x = (y.x*h.x+y.y*h.y)/denom;
    equalized[i].y = (y.y*h.x-y.x*h.y)/denom;
}

__device__ __forceinline__ int16_t matlabRoundInt16(float value)
{
    float rounded = value >= 0.0f
        ? floorf(value+0.5f)
        : ceilf(value-0.5f);
    rounded = fminf(32767.0f,fmaxf(-32768.0f,rounded));
    return static_cast<int16_t>(rounded);
}

__global__ void segmentQuantizeCSIKernel(
    const cufftComplex* __restrict__ equalized,
    const cufftComplex* __restrict__ channel,
    const int* __restrict__ dataIndices,
    float noiseVariance,
    int16_t* __restrict__ iOut,
    int16_t* __restrict__ qOut,
    float* __restrict__ csiOut,
    float* __restrict__ inverseQuantScales)
{
    int segment = blockIdx.x;
    __shared__ float peakValues[256];

    float localPeak = 0.0f;
    for (int linear = threadIdx.x;
         linear < kSymbolsPerSegment;
         linear += blockDim.x) {
        int localTone = linear % kTonesPerSegment;
        int symbol = linear / kTonesPerSegment;
        int rawTone = segment*kTonesPerSegment+localTone;
        int activeTone = dataIndices[rawTone];
        cufftComplex x =
            equalized[symbol*kActiveToneCount+activeTone];
        localPeak = fmaxf(localPeak,fabsf(x.x));
        localPeak = fmaxf(localPeak,fabsf(x.y));
    }

    peakValues[threadIdx.x] = localPeak;
    __syncthreads();
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            peakValues[threadIdx.x] =
                fmaxf(peakValues[threadIdx.x],
                      peakValues[threadIdx.x+stride]);
        __syncthreads();
    }

    constexpr float numerator = 29490.30078125f;
    float scale = peakValues[0] == 0.0f
        ? 1.0f : numerator/peakValues[0];

    if (threadIdx.x == 0)
        inverseQuantScales[segment] = 1.0f/scale;
    __syncthreads();

    for (int outputLinear = threadIdx.x;
         outputLinear < kSymbolsPerSegment;
         outputLinear += blockDim.x) {
        int permutedTone = outputLinear % kTonesPerSegment;
        int symbol = outputLinear / kTonesPerSegment;

        int row49 = permutedTone % 49;
        int col20 = permutedTone / 49;
        int rawLocalTone = col20+20*row49;

        int rawTone = segment*kTonesPerSegment+rawLocalTone;
        int activeTone = dataIndices[rawTone];
        cufftComplex x =
            equalized[symbol*kActiveToneCount+activeTone];

        int outputIndex = segment*kSymbolsPerSegment+outputLinear;
        iOut[outputIndex] = matlabRoundInt16(x.x*scale);
        qOut[outputIndex] = matlabRoundInt16(x.y*scale);

        if (symbol == 0) {
            cufftComplex h = channel[activeTone];
            float power = h.x*h.x+h.y*h.y;
            csiOut[segment*kTonesPerSegment+permutedTone] =
                power/(power+noiseVariance);
        }
    }
}

__device__ __forceinline__ unsigned int grayCode(
    unsigned int value)
{
    return value ^ (value >> 1U);
}

__global__ void qam4096DemapKernel(
    const int16_t* __restrict__ iIn,
    const int16_t* __restrict__ qIn,
    int8_t* __restrict__ llrOut,
    int symbolCount,
    const float* __restrict__ inverseQuantScales,
    float inverseNoiseVariance,
    float llrScale)
{
    int symbol = int(blockIdx.x)*blockDim.x + threadIdx.x;
    if (symbol >= symbolCount) return;

    constexpr float constellationNorm =
        1.0f / 52.24940180716435f;

    int segment = symbol / kSymbolsPerSegment;
    float inverseQuantScale = inverseQuantScales[segment];

    float samples[2] = {
        float(iIn[symbol]) * inverseQuantScale,
        float(qIn[symbol]) * inverseQuantScale
    };

    #pragma unroll
    for (int axis = 0; axis < 2; ++axis) {
        float sample = samples[axis];

        #pragma unroll
        for (int bit = 0; bit < 6; ++bit) {
            float min0 = FLT_MAX;
            float min1 = FLT_MAX;
            int bitPosition = 5-bit;

            #pragma unroll 8
            for (int levelIndex = 0; levelIndex < 64; ++levelIndex) {
                int integerLevel = -63 + 2*levelIndex;
                float level = float(integerLevel)*constellationNorm;
                float delta = sample-level;
                float distance = delta*delta;
                unsigned int label = grayCode(
                    static_cast<unsigned int>(levelIndex));
                unsigned int value =
                    (label >> bitPosition) & 1U;

                if (value == 0U) min0 = fminf(min0,distance);
                else min1 = fminf(min1,distance);
            }

            float llr = (min1-min0)*inverseNoiseVariance;
            float scaled = nearbyintf(llr*llrScale);
            float clipped = fminf(127.0f,fmaxf(-128.0f,scaled));

            llrOut[symbol*kBitsPerSymbol + axis*6 + bit] =
                static_cast<int8_t>(clipped);
        }
    }
}

__global__ void mapWeightKernel(
    const int8_t* __restrict__ externalLLR,
    const float* __restrict__ csi,
    const uint32_t* __restrict__ encodedSourceIndex,
    float inverseLLRScale,
    float* __restrict__ encoded)
{
    int index = int(blockIdx.x)*blockDim.x + threadIdx.x;
    if (index >= kEncodedLength) return;

    uint32_t source = encodedSourceIndex[index];
    int segment = int(source / kExternalElementsPerSegment);
    int local = int(source % kExternalElementsPerSegment);

    int row = local % (kBitsPerSymbol*kTonesPerSegment);
    int bit = row % kBitsPerSymbol;
    int tone = row / kBitsPerSymbol;

    float value = float(externalLLR[source])*inverseLLRScale;

    // The fused max-log demapper already produces MATLAB-compatible
    // polarity for bit position 7. The standalone qam4096 MEX requires a
    // host-side bit-7 sign correction, but applying that correction again
    // here would invert one LLR per QAM symbol.
    value *= csi[segment*kTonesPerSegment + tone];
    encoded[index] = value;
}

// Exact IEEE 802.11 N=1944, rate-5/6 QC shift matrix.
__device__ __constant__ int8_t P[BASE_ROWS * BASE_COLS] = {
  13,48,80,66, 4,74, 7,30,76,52,37,60,-1,49,73,31,74,73,23,-1, 1, 0,-1,-1,
  69,63,74,56,64,77,57,65, 6,16,51,-1,64,-1,68, 9,48,62,54,27,-1, 0, 0,-1,
  51,15, 0,80,24,25,42,54,44,71,71, 9,67,35,-1,58,-1,29,-1,53, 0,-1, 0, 0,
  16,29,36,41,44,56,59,37,50,24,-1,65, 4,65,52,-1, 4,-1,73,52, 1,-1,-1, 0
};

__global__ void reconstructKernel(
    const float* __restrict__ encoded,
    const int* __restrict__ payloadBits,
    const int* __restrict__ punctureBits,
    const int* __restrict__ repeatBits,
    float* __restrict__ beliefs,
    float* __restrict__ messages)
{
    int cw = blockIdx.x;
    int bit = threadIdx.x + blockIdx.y*blockDim.x;
    if (cw >= kNumCodewords || bit >= N) return;

    int pld = payloadBits[cw];
    int sh = K-pld;
    int pun = punctureBits[cw];
    int parityAvailable = M-pun;

    int inputOffset = 0;
    for (int i = 0; i < cw; ++i) {
        inputOffset += payloadBits[i] +
            (M-punctureBits[i]) +
            repeatBits[i];
    }

    float value;
    if (bit < pld) {
        value = encoded[inputOffset+bit];
    } else if (bit < K) {
        value = SHORTEN_LLR;
    } else if (bit < K+parityAvailable) {
        value = encoded[inputOffset+pld+(bit-K)];
    } else {
        value = 0.0f;
    }

    beliefs[cw*N+bit] = value;

    for (int edge = bit; edge < EDGES; edge += N) {
        messages[cw*EDGES+edge] = 0.0f;
    }
}

__global__ void layeredNMSKernel(
    float* __restrict__ beliefs,
    float* __restrict__ messages)
{
    int cw = blockIdx.x;
    int r = threadIdx.x;
    if (cw >= kNumCodewords || r >= Z) return;

    float* L = beliefs + cw*N;
    float* R = messages + cw*EDGES;

    for (int iter = 0; iter < kMaximumIterations; ++iter) {
        int edgeOrdinal = 0;

        for (int br = 0; br < BASE_ROWS; ++br) {
            float min1 = FLT_MAX;
            float min2 = FLT_MAX;
            int minCol = -1;
            int signProduct = 1;
            int layerStart = edgeOrdinal;

            for (int bc = 0; bc < BASE_COLS; ++bc) {
                int shift = int(P[br*BASE_COLS+bc]);
                if (shift < 0) continue;

                int v = bc*Z + ((r+shift)%Z);
                int edge = edgeOrdinal*Z+r;
                float q = L[v]-R[edge];
                float magnitude = fabsf(q);
                int sign = q < 0.0f ? -1 : 1;
                signProduct *= sign;

                if (magnitude < min1) {
                    min2 = min1;
                    min1 = magnitude;
                    minCol = bc;
                } else if (magnitude < min2) {
                    min2 = magnitude;
                }
                ++edgeOrdinal;
            }

            edgeOrdinal = layerStart;

            for (int bc = 0; bc < BASE_COLS; ++bc) {
                int shift = int(P[br*BASE_COLS+bc]);
                if (shift < 0) continue;

                int v = bc*Z + ((r+shift)%Z);
                int edge = edgeOrdinal*Z+r;
                float oldR = R[edge];
                float q = L[v]-oldR;
                int ownSign = q < 0.0f ? -1 : 1;
                float magnitude = bc == minCol ? min2 : min1;
                float newR = kAlpha*magnitude*
                    float(signProduct*ownSign);

                R[edge] = newR;
                L[v] = q+newR;
                ++edgeOrdinal;
            }

            __syncthreads();
        }
    }
}

__global__ void hardDecisionKernel(
    const float* __restrict__ beliefs,
    const int* __restrict__ payloadBits,
    int8_t* __restrict__ decoded)
{
    int cw = blockIdx.x;
    int i = threadIdx.x + blockIdx.y*blockDim.x;
    if (cw >= kNumCodewords ||
        i >= payloadBits[cw]) return;

    int outputOffset = 0;
    for (int c = 0; c < cw; ++c)
        outputOffset += payloadBits[c];

    decoded[outputOffset+i] =
        beliefs[cw*N+i] < 0.0f ? int8_t(1) : int8_t(0);
}

__global__ void deriveScramblerSequenceKernel(
    const int8_t* __restrict__ decoded,
    int8_t* __restrict__ sequence,
    int* __restrict__ descrambleEnabled)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    const int idx[11][7] = {
        {0,1,2,4,6,8,10},
        {0,1,3,5,7,9,-1},
        {1,2,4,6,8,10,-1},
        {0,3,5,7,9,-1,-1},
        {1,4,6,8,10,-1,-1},
        {0,5,7,9,-1,-1,-1},
        {1,6,8,10,-1,-1,-1},
        {0,7,9,-1,-1,-1,-1},
        {1,8,10,-1,-1,-1,-1},
        {0,9,-1,-1,-1,-1,-1},
        {1,10,-1,-1,-1,-1,-1}
    };

    int8_t state[11];
    int any = 0;

    for (int s = 0; s < 11; ++s) {
        int sum = 0;
        for (int j = 0; j < 7; ++j) {
            int index = idx[s][j];
            if (index >= 0) sum += int(decoded[index]);
        }
        state[s] = int8_t(sum & 1);
        any |= int(state[s]);
    }

    *descrambleEnabled = any ? 1 : 0;

    for (int d = 0; d < 2047; ++d) {
        int8_t bit = int8_t(state[0]^state[2]);
        sequence[d] = bit;
        for (int s = 0; s < 10; ++s) state[s] = state[s+1];
        state[10] = bit;
    }
}

__global__ void validatePayloadKernel(
    const int8_t* __restrict__ decoded,
    const int8_t* __restrict__ sequence,
    const int* __restrict__ descrambleEnabled,
    int validationMode,
    const int8_t* __restrict__ referenceTxBits,
    unsigned long long* __restrict__ checksum,
    unsigned int* __restrict__ bitErrors)
{
    int i = int(blockIdx.x)*blockDim.x + threadIdx.x;
    if (i >= kPayloadBits) return;

    constexpr int serviceBits = 16;
    int sourceIndex = serviceBits+i;

    int8_t value = decoded[sourceIndex];
    if (*descrambleEnabled)
        value = int8_t(value ^ sequence[sourceIndex % 2047]);

    if (value)
        atomicAdd(checksum,static_cast<unsigned long long>(i+1));

    if (validationMode == 1 &&
        value != referenceTxBits[i]) {
        atomicAdd(bitErrors,1U);
    }
}




// -------------------------------------------------------------------------
// Part 10B-2: genuinely batched CUDA receiver backend.
// Every packet occupies one contiguous slice of each device array.
// -------------------------------------------------------------------------
__global__ void gatherFFTWindowsBatchKernel(const cufftComplex* ltfField,const cufftComplex* dataField,
    cufftComplex* ltfIn,cufftComplex* dataIn,int ltfStart,int dataStart0,int batch)
{
    int i=int(blockIdx.x)*blockDim.x+threadIdx.x; int b=int(blockIdx.y); if(b>=batch)return;
    if(i<kFFTLength){
        ltfIn[size_t(b)*kFFTLength+i]=ltfField[size_t(b)*kEHTLTFLength+ltfStart+i];
        for(int sym=0;sym<kNumDataSymbols;++sym){
            const int sourceStart=dataStart0+sym*kDataSymbolLength;
            dataIn[size_t(b)*kDataInputCount+size_t(sym)*kFFTLength+i]=
                dataField[size_t(b)*kEHTDataLength+sourceStart+i];
        }
    }
}
__global__ void extractActiveBatchBackendKernel(const cufftComplex* ltfFFT,const cufftComplex* dataFFT,
    const int* li,const int* di,cufftComplex* la,cufftComplex* da,int batch)
{
    int i=int(blockIdx.x)*blockDim.x+threadIdx.x,b=int(blockIdx.y); if(b>=batch||i>=kActiveToneCount)return;
    la[size_t(b)*kActiveToneCount+i]=ltfFFT[size_t(b)*kFFTLength+li[i]];
    size_t doff=size_t(b)*kDataInputCount; size_t aoff=size_t(b)*kActiveToneCount*kNumDataSymbols;
    for(int sym=0;sym<kNumDataSymbols;++sym)
        da[aoff+size_t(sym)*kActiveToneCount+i]=dataFFT[doff+size_t(sym)*kFFTLength+di[i]];
}
__global__ void estimateChannelBatchBackendKernel(const cufftComplex* la,const cufftComplex* known,cufftComplex* ch,int batch)
{
    int i=int(blockIdx.x)*blockDim.x+threadIdx.x,b=int(blockIdx.y);if(b>=batch||i>=kActiveToneCount)return;
    auto y=la[size_t(b)*kActiveToneCount+i],r=known[i]; ch+=size_t(b)*kActiveToneCount;
    ch[i].x=y.x*r.x+y.y*r.y;ch[i].y=y.y*r.x-y.x*r.y;
}
__global__ void pilotRotationBatchBackendKernel(const cufftComplex* da,const cufftComplex* ch,const int* pi,
    const cufftComplex* pref,int pc,cufftComplex* rot,int batch)
{
    int bs=int(blockIdx.x),b=bs/kNumDataSymbols,sym=bs%kNumDataSymbols;if(b>=batch)return;
    extern __shared__ cufftComplex sh[];cufftComplex sum{0,0};
    da+=size_t(b)*kActiveToneCount*kNumDataSymbols;ch+=size_t(b)*kActiveToneCount;
    for(int p=threadIdx.x;p<pc;p+=blockDim.x){int t=pi[p];auto y=da[sym*kActiveToneCount+t],h=ch[t],r=pref[sym*pc+p];
      cufftComplex e{h.x*r.x-h.y*r.y,h.x*r.y+h.y*r.x};sum.x+=y.x*e.x+y.y*e.y;sum.y+=y.y*e.x-y.x*e.y;}
    sh[threadIdx.x]=sum;__syncthreads();for(int q=blockDim.x/2;q;q>>=1){if(threadIdx.x<q){sh[threadIdx.x].x+=sh[threadIdx.x+q].x;sh[threadIdx.x].y+=sh[threadIdx.x+q].y;}__syncthreads();}
    if(threadIdx.x==0){float m=hypotf(sh[0].x,sh[0].y);rot[size_t(b)*kNumDataSymbols+sym]=m>0?cufftComplex{sh[0].x/m,-sh[0].y/m}:cufftComplex{1,0};}
}
__global__ void equalizeBatchBackendKernel(const cufftComplex* da,const cufftComplex* ch,const cufftComplex* rot,
    const float* nv,cufftComplex* eq,int batch)
{
    int i=int(blockIdx.x)*blockDim.x+threadIdx.x,b=int(blockIdx.y);int n=kActiveToneCount*kNumDataSymbols;if(b>=batch||i>=n)return;
    int tone=i%kActiveToneCount,sym=i/kActiveToneCount;size_t ao=size_t(b)*n;auto in=da[ao+i],r=rot[size_t(b)*kNumDataSymbols+sym];
    cufftComplex y{in.x*r.x-in.y*r.y,in.x*r.y+in.y*r.x};auto h=ch[size_t(b)*kActiveToneCount+tone];float d=h.x*h.x+h.y*h.y+nv[b];
    eq[ao+i]={(y.x*h.x+y.y*h.y)/d,(y.y*h.x-y.x*h.y)/d};
}
__global__ void segmentQuantizeBatchBackendKernel(const cufftComplex* eq,const cufftComplex* ch,const int* dataIdx,
    const float* nv,int16_t* io,int16_t* qo,float* csi,float* invq,int batch)
{
    int seg=int(blockIdx.x),b=int(blockIdx.y);if(b>=batch)return;__shared__ float pk[256];float lp=0;
    size_t eo=size_t(b)*kActiveToneCount*kNumDataSymbols,so=size_t(b)*kTotalSymbols,co=size_t(b)*kDataToneCount;
    for(int l=threadIdx.x;l<kSymbolsPerSegment;l+=blockDim.x){int lt=l%kTonesPerSegment,sym=l/kTonesPerSegment,rt=seg*kTonesPerSegment+lt,at=dataIdx[rt];auto x=eq[eo+sym*kActiveToneCount+at];lp=fmaxf(lp,fmaxf(fabsf(x.x),fabsf(x.y)));}
    pk[threadIdx.x]=lp;__syncthreads();for(int q=blockDim.x/2;q;q>>=1){if(threadIdx.x<q)pk[threadIdx.x]=fmaxf(pk[threadIdx.x],pk[threadIdx.x+q]);__syncthreads();}
    float scale=pk[0]==0?1.0f:29490.30078125f/pk[0];if(threadIdx.x==0)invq[size_t(b)*kNumSegments+seg]=1.0f/scale;__syncthreads();
    for(int ol=threadIdx.x;ol<kSymbolsPerSegment;ol+=blockDim.x){int pt=ol%kTonesPerSegment,sym=ol/kTonesPerSegment,row=pt%49,col=pt/49,raw=col+20*row,rt=seg*kTonesPerSegment+raw,at=dataIdx[rt];auto x=eq[eo+sym*kActiveToneCount+at];int oi=seg*kSymbolsPerSegment+ol;io[so+oi]=matlabRoundInt16(x.x*scale);qo[so+oi]=matlabRoundInt16(x.y*scale);if(sym==0){auto h=ch[size_t(b)*kActiveToneCount+at];float pw=h.x*h.x+h.y*h.y;csi[co+seg*kTonesPerSegment+pt]=pw/(pw+nv[b]);}}
}
__global__ void qamBatchBackendKernel(const int16_t* ii,const int16_t* qq,int8_t* out,const float* invq,const float* nv,float ls,int batch)
{
    int lin=int(blockIdx.x)*blockDim.x+threadIdx.x;int total=batch*kTotalSymbols;if(lin>=total)return;int b=lin/kTotalSymbols,s=lin%kTotalSymbols;
    constexpr float cn=1.0f/52.24940180716435f;int seg=s/kSymbolsPerSegment;float iq=invq[size_t(b)*kNumSegments+seg],invn=1.0f/nv[b];float samples[2]={float(ii[lin])*iq,float(qq[lin])*iq};
    size_t oo=size_t(b)*kTotalSymbols*kBitsPerSymbol;
    #pragma unroll
    for(int ax=0;ax<2;++ax){float sample=samples[ax];
      for(int bit=0;bit<6;++bit){float m0=FLT_MAX,m1=FLT_MAX;int bp=5-bit;
        for(int li=0;li<64;++li){int il=-63+2*li;float lv=float(il)*cn,d=sample-lv,dist=d*d;unsigned lab=grayCode((unsigned)li),v=(lab>>bp)&1U;if(v==0)m0=fminf(m0,dist);else m1=fminf(m1,dist);}float llr=(m1-m0)*invn;float sc=nearbyintf(llr*ls);sc=fminf(127.0f,fmaxf(-128.0f,sc));out[oo+size_t(s)*kBitsPerSymbol+ax*6+bit]=(int8_t)sc;}}
}
__global__ void mapWeightBatchBackendKernel(const int8_t* ext,const float* csi,const uint32_t* esi,float invls,float* enc,int batch)
{
 int lin=int(blockIdx.x)*blockDim.x+threadIdx.x,total=batch*kEncodedLength;if(lin>=total)return;int b=lin/kEncodedLength,i=lin%kEncodedLength;uint32_t src=esi[i];int seg=int(src/kExternalElementsPerSegment),local=int(src%kExternalElementsPerSegment),row=local%(kBitsPerSymbol*kTonesPerSegment),tone=row/kBitsPerSymbol;float v=float(ext[size_t(b)*kTotalSymbols*kBitsPerSymbol+src])*invls;v*=csi[size_t(b)*kDataToneCount+seg*kTonesPerSegment+tone];enc[size_t(b)*kEncodedLength+i]=v;
}
__global__ void reconstructBatchBackendKernel(const float* enc,const int* pb,const int* pun,const int* rep,float* bel,float* msg,int batch)
{
 int b=int(blockIdx.z),cw=int(blockIdx.x),bit=int(threadIdx.x)+int(blockIdx.y)*blockDim.x;if(b>=batch||cw>=kNumCodewords||bit>=N)return;int p=pb[cw],sh=K-p,pu=pun[cw],pa=M-pu,off=0;for(int i=0;i<cw;++i)off+=pb[i]+(M-pun[i])+rep[i];float v;if(bit<p)v=enc[size_t(b)*kEncodedLength+off+bit];else if(bit<K)v=SHORTEN_LLR;else if(bit<K+pa)v=enc[size_t(b)*kEncodedLength+off+p+(bit-K)];else v=0;bel[(size_t(b)*kNumCodewords+cw)*N+bit]=v;for(int e=bit;e<EDGES;e+=N)msg[(size_t(b)*kNumCodewords+cw)*EDGES+e]=0;
}
__global__ void layeredNMSBatchBackendKernel(float* bel,float* msg,int batch)
{
 int cw=int(blockIdx.x),b=int(blockIdx.y),r=threadIdx.x;if(b>=batch||cw>=kNumCodewords||r>=Z)return;float*L=bel+(size_t(b)*kNumCodewords+cw)*N;float*R=msg+(size_t(b)*kNumCodewords+cw)*EDGES;
 for(int it=0;it<kMaximumIterations;++it){int eo=0;for(int br=0;br<BASE_ROWS;++br){float m1=FLT_MAX,m2=FLT_MAX;int mc=-1,sp=1,ls=eo;for(int bc=0;bc<BASE_COLS;++bc){int shift=int(P[br*BASE_COLS+bc]);if(shift<0)continue;int v=bc*Z+((r+shift)%Z),e=eo*Z+r;float q=L[v]-R[e],ma=fabsf(q);sp*=q<0?-1:1;if(ma<m1){m2=m1;m1=ma;mc=bc;}else if(ma<m2)m2=ma;++eo;}eo=ls;for(int bc=0;bc<BASE_COLS;++bc){int shift=int(P[br*BASE_COLS+bc]);if(shift<0)continue;int v=bc*Z+((r+shift)%Z),e=eo*Z+r;float old=R[e],q=L[v]-old;int os=q<0?-1:1;float ma=bc==mc?m2:m1,nr=kAlpha*ma*float(sp*os);R[e]=nr;L[v]=q+nr;++eo;}__syncthreads();}}
}
__global__ void hardDecisionBatchBackendKernel(const float* bel,const int* pb,int8_t* dec,int batch)
{
 int b=int(blockIdx.z),cw=int(blockIdx.x),i=threadIdx.x+int(blockIdx.y)*blockDim.x;if(b>=batch||cw>=kNumCodewords||i>=pb[cw])return;int oo=0;for(int c=0;c<cw;++c)oo+=pb[c];dec[size_t(b)*kTotalDecodedBits+oo+i]=bel[(size_t(b)*kNumCodewords+cw)*N+i]<0?1:0;
}
__global__ void deriveScramblerBatchBackendKernel(const int8_t* dec,int8_t* seq,int* en,int batch)
{
 int b=int(blockIdx.x);if(b>=batch||threadIdx.x)return;const int idx[11][7]={{0,1,2,4,6,8,10},{0,1,3,5,7,9,-1},{1,2,4,6,8,10,-1},{0,3,5,7,9,-1,-1},{1,4,6,8,10,-1,-1},{0,5,7,9,-1,-1,-1},{1,6,8,10,-1,-1,-1},{0,7,9,-1,-1,-1,-1},{1,8,10,-1,-1,-1,-1},{0,9,-1,-1,-1,-1,-1},{1,10,-1,-1,-1,-1,-1}};const int8_t*d=dec+size_t(b)*kTotalDecodedBits;int8_t st[11];int any=0;for(int s=0;s<11;++s){int sum=0;for(int j=0;j<7;++j){int x=idx[s][j];if(x>=0)sum+=int(d[x]);}st[s]=int8_t(sum&1);any|=int(st[s]);}en[b]=any?1:0;int8_t*q=seq+size_t(b)*2047;for(int d0=0;d0<2047;++d0){int8_t bit=int8_t(st[0]^st[2]);q[d0]=bit;for(int s=0;s<10;++s)st[s]=st[s+1];st[10]=bit;}
}
// STEP 9 Stage 11: apply the recovered scrambler sequence to the decoded
// LDPC bitstream. Keeping this separate from payload extraction gives an
// independent CUDA-event timing for Descrambling.
__global__ void descrambleDecodedBatchBackendKernel(const int8_t* dec,const int8_t* seq,const int* en,int8_t* out,int batch)
{
 int lin=int(blockIdx.x)*blockDim.x+threadIdx.x,total=batch*kTotalDecodedBits;if(lin>=total)return;int b=lin/kTotalDecodedBits,i=lin%kTotalDecodedBits;int8_t v=dec[size_t(b)*kTotalDecodedBits+i];if(en[b])v=int8_t(v^seq[size_t(b)*2047+(i%2047)]);out[size_t(b)*kTotalDecodedBits+i]=v;
}





// STEP 9 Stage 12: recover the PSDU payload bits from the descrambled
// decoded stream and reduce them to the existing checksum. No bulk payload
// is copied back to MATLAB.



__global__ void checksumRecoveredPayloadBatchBackendKernel(const int8_t* descr,unsigned long long* cs,int batch)
{
 int i=int(blockIdx.x)*blockDim.x+threadIdx.x,b=int(blockIdx.y);if(b>=batch||i>=kPayloadBits)return;int si=16+i;int8_t v=descr[size_t(b)*kTotalDecodedBits+si];if(v)atomicAdd(cs+b,(unsigned long long)(i+1));
}

__global__ void validateBatchBackendKernel(const int8_t* dec,const int8_t* seq,const int* en,const int8_t* ref,unsigned long long* cs,unsigned int* be,int batch)
{
 int i=int(blockIdx.x)*blockDim.x+threadIdx.x,b=int(blockIdx.y);if(b>=batch||i>=kPayloadBits)return;int si=16+i;int8_t v=dec[size_t(b)*kTotalDecodedBits+si];if(en[b])v=int8_t(v^seq[size_t(b)*2047+(si%2047)]);if(v)atomicAdd(cs+b,(unsigned long long)(i+1));if(v!=ref[size_t(b)*kPayloadBits+i])atomicAdd(be+b,1U);
}

__global__ void checksumBatchBackendKernel(const int8_t* dec,const int8_t* seq,const int* en,unsigned long long* cs,int batch)
{
 int i=int(blockIdx.x)*blockDim.x+threadIdx.x,b=int(blockIdx.y);
 if(b>=batch||i>=kPayloadBits)return;
 int si=16+i;
 int8_t v=dec[size_t(b)*kTotalDecodedBits+si];
 if(en[b])v=int8_t(v^seq[size_t(b)*2047+(si%2047)]);
 if(v)atomicAdd(cs+b,(unsigned long long)(i+1));
}

double scalar(const mxArray* value,const char* name) {
    if (!mxIsNumeric(value) || mxIsComplex(value) ||
        mxGetNumberOfElements(value) != 1) {
        mexErrMsgIdAndTxt(
            "eht_receiver_cuda_async:Input",
            "%s must be a real numeric scalar.",name);
    }

    double result = mxGetScalar(value);
    if (!std::isfinite(result)) {
        mexErrMsgIdAndTxt(
            "eht_receiver_cuda_async:Input",
            "%s must be finite.",name);
    }
    return result;
}

int parseDataFFTStart0(const mxArray* value) {
    if (!mxIsNumeric(value) || mxIsComplex(value) ||
        mxGetNumberOfElements(value) != kNumDataSymbols)
        mexErrMsgIdAndTxt(
            "eht_receiver_cuda_e2e:Input",
            "dataFFTStarts must contain %d real values.",kNumDataSymbols);

    int first = -1;
    for (int sym = 0; sym < kNumDataSymbols; ++sym) {
        const double raw = mxIsDouble(value)
            ? mxGetDoubles(value)[sym]
            : double(mxGetSingles(value)[sym]);
        const int start = int(raw)-1;
        if (double(start+1) != raw || start < 0 ||
            start+kFFTLength > kEHTDataLength)
            mexErrMsgIdAndTxt(
                "eht_receiver_cuda_e2e:Input",
                "FFT windows are outside the extracted EHT fields.");
        if (sym == 0) first = start;
        else if (start != first+sym*kDataSymbolLength)
            mexErrMsgIdAndTxt(
                "eht_receiver_cuda_e2e:Input",
                "dataFFTStarts do not match the configured OFDM symbol spacing.");
    }
    return first;
}

std::string commandString(const mxArray* value) {
    if (!mxIsChar(value))
        mexErrMsgIdAndTxt(
            "eht_receiver_cuda_async:Command",
            "First input must be a command string.");

    char* raw = mxArrayToString(value);
    if (!raw)
        mexErrMsgIdAndTxt(
            "eht_receiver_cuda_async:Command",
            "Could not read command string.");

    std::string command(raw);
    mxFree(raw);
    return command;
}

void validateCellCount(const mxArray* value,const char* name) {
    if (!mxIsCell(value) ||
        mxGetNumberOfElements(value) != kNumSegments) {
        mexErrMsgIdAndTxt(
            "eht_receiver_cuda_async:Input",
            "%s must be a four-element cell array.",name);
    }
}


void copyComplexSingleWindow(
    const mxArray* input,
    int firstZeroBased,
    int count,
    cufftComplex* destination,
    const char* name)
{
    if (!mxIsSingle(input) || !mxIsComplex(input))
        mexErrMsgIdAndTxt(
            "eht_receiver_cuda_async:Input",
            "%s must be complex single.",name);

    if (firstZeroBased < 0 ||
        firstZeroBased+count >
            int(mxGetNumberOfElements(input)))
        mexErrMsgIdAndTxt(
            "eht_receiver_cuda_async:Input",
            "%s FFT window is outside the input.",name);

    const mxComplexSingle* p = mxGetComplexSingles(input);
    for (int i = 0; i < count; ++i) {
        destination[i].x = p[firstZeroBased+i].real;
        destination[i].y = p[firstZeroBased+i].imag;
    }
}

std::vector<int> readOneBasedIndices(
    const mxArray* value,
    int expectedCount,
    int maximum,
    const char* name)
{
    if (!mxIsNumeric(value) || mxIsComplex(value) ||
        int(mxGetNumberOfElements(value)) != expectedCount)
        mexErrMsgIdAndTxt(
            "eht_receiver_cuda_async:Input",
            "%s has an invalid size or type.",name);

    std::vector<int> out(expectedCount);
    for (int i = 0; i < expectedCount; ++i) {
        double x = mxIsDouble(value)
            ? mxGetDoubles(value)[i]
            : double(mxGetSingles(value)[i]);
        int v = int(x);
        if (double(v) != x || v < 1 || v > maximum)
            mexErrMsgIdAndTxt(
                "eht_receiver_cuda_async:Input",
                "%s contains an invalid index.",name);
        out[i] = v-1;
    }
    return out;
}

void initializeFrontendMetadata(
    const mxArray* ltfFFTIndices,
    const mxArray* dataFFTIndices,
    const mxArray* knownLTF,
    const mxArray* pilotIndices,
    const mxArray* pilotReference,
    const mxArray* dataIndices)
{
    if (sharedData.frontendMetadataInitialized) return;

    std::vector<int> ltfFFTShifted = readOneBasedIndices(
        ltfFFTIndices,kActiveToneCount,kFFTLength,
        "ltfFFTIndices");
    std::vector<int> dataFFTShifted = readOneBasedIndices(
        dataFFTIndices,kActiveToneCount,kFFTLength,
        "dataFFTIndices");

    // wlanEHTOFDMInfo ActiveFFTIndices address fftshift(fft(...)).
    // cuFFT returns the native unshifted FFT order. Convert each
    // zero-based shifted index to the corresponding native cuFFT bin.
    std::vector<int> ltfFFT(kActiveToneCount);
    std::vector<int> dataFFT(kActiveToneCount);

    for (int i = 0; i < kActiveToneCount; ++i) {
        ltfFFT[i] =
            (ltfFFTShifted[i] + kFFTLength/2) % kFFTLength;
        dataFFT[i] =
            (dataFFTShifted[i] + kFFTLength/2) % kFFTLength;
    }
    std::vector<int> data = readOneBasedIndices(
        dataIndices,kDataToneCount,kActiveToneCount,
        "dataIndices");

    int pilotCount = int(mxGetNumberOfElements(pilotIndices));
    if (pilotCount <= 0)
        mexErrMsgIdAndTxt(
            "eht_receiver_cuda_async:Input",
            "pilotIndices cannot be empty.");

    std::vector<int> pilots = readOneBasedIndices(
        pilotIndices,pilotCount,kActiveToneCount,
        "pilotIndices");

    if (!mxIsSingle(knownLTF) || !mxIsComplex(knownLTF) ||
        int(mxGetNumberOfElements(knownLTF)) != kActiveToneCount)
        mexErrMsgIdAndTxt(
            "eht_receiver_cuda_async:Input",
            "knownLTF must be complex single [3984 x 1].");

    if (!mxIsSingle(pilotReference) ||
        !mxIsComplex(pilotReference) ||
        int(mxGetNumberOfElements(pilotReference)) !=
            pilotCount*kNumDataSymbols)
        mexErrMsgIdAndTxt(
            "eht_receiver_cuda_async:Input",
            "pilotReference must be complex single [P x 2].");

    std::vector<cufftComplex> hKnown(kActiveToneCount);
    std::vector<cufftComplex> hPilot(
        pilotCount*kNumDataSymbols);

    const mxComplexSingle* known = mxGetComplexSingles(knownLTF);
    for (int i = 0; i < kActiveToneCount; ++i)
        hKnown[i] = cufftComplex{known[i].real,known[i].imag};

    const mxComplexSingle* ref =
        mxGetComplexSingles(pilotReference);
    for (int i = 0; i < pilotCount*kNumDataSymbols; ++i)
        hPilot[i] = cufftComplex{ref[i].real,ref[i].imag};

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&sharedData.pilotIndices),
        size_t(pilotCount)*sizeof(int)));
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&sharedData.pilotReference),
        size_t(pilotCount*kNumDataSymbols)*sizeof(cufftComplex)));

    CUDA_CHECK(cudaMemcpy(
        sharedData.ltfFFTIndices,ltfFFT.data(),
        size_t(kActiveToneCount)*sizeof(int),
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        sharedData.dataFFTIndices,dataFFT.data(),
        size_t(kActiveToneCount)*sizeof(int),
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        sharedData.dataIndices,data.data(),
        size_t(kDataToneCount)*sizeof(int),
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        sharedData.pilotIndices,pilots.data(),
        size_t(pilotCount)*sizeof(int),
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        sharedData.knownLTF,hKnown.data(),
        size_t(kActiveToneCount)*sizeof(cufftComplex),
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        sharedData.pilotReference,hPilot.data(),
        size_t(pilotCount*kNumDataSymbols)*sizeof(cufftComplex),
        cudaMemcpyHostToDevice));

    sharedData.pilotCount = pilotCount;
    sharedData.frontendMetadataInitialized = true;
}

Slot* findFreeSlot() {
    for (int i = 0; i < kNumSlots; ++i) {
        if (!slots[i].occupied)
            return &slots[i];
    }
    return nullptr;
}

Slot* findTicket(uint64_t ticket) {
    for (int i = 0; i < kNumSlots; ++i) {
        if (slots[i].occupied && slots[i].ticket == ticket)
            return &slots[i];
    }
    return nullptr;
}

mxArray* makeStats(const Slot& s) {
    float totalMs = 0.0f;
    float h2dMs = 0.0f;
    float frontendMs = 0.0f;
    float demapMs = 0.0f;
    float mapMs = 0.0f;
    float ldpcMs = 0.0f;
    float validateMs = 0.0f;

    CUDA_CHECK(cudaEventElapsedTime(
        &totalMs,s.start,s.afterValidate));
    CUDA_CHECK(cudaEventElapsedTime(
        &h2dMs,s.start,s.afterH2D));
    CUDA_CHECK(cudaEventElapsedTime(
        &frontendMs,s.afterH2D,s.afterFrontend));
    CUDA_CHECK(cudaEventElapsedTime(
        &demapMs,s.afterFrontend,s.afterDemap));
    CUDA_CHECK(cudaEventElapsedTime(
        &mapMs,s.afterDemap,s.afterMap));
    CUDA_CHECK(cudaEventElapsedTime(
        &ldpcMs,s.afterMap,s.afterLDPC));
    CUDA_CHECK(cudaEventElapsedTime(
        &validateMs,s.afterLDPC,s.afterValidate));

    unsigned long long checksum = *s.hChecksum;
    unsigned int bitErrors = *s.hBitErrors;

    const char* fields[] = {
        "Ticket","PacketsProcessed","BitErrors","PacketErrors",
        "PayloadChecksum","ValidationMode",
        "TotalGPUTimeMs","H2DTimeMs","FrontendTimeMs","DemapperTimeMs",
        "MapCSIWeightTimeMs","LDPCDecodeTimeMs",
        "DescrambleValidateTimeMs","ReturnedBulkOutputBytes",
        "Asynchronous","BufferCount"
    };

    mxArray* stats = mxCreateStructMatrix(1,1,16,fields);
    mxSetField(stats,0,"Ticket",
        mxCreateDoubleScalar(double(s.ticket)));
    mxSetField(stats,0,"PacketsProcessed",
        mxCreateDoubleScalar(1));
    mxSetField(stats,0,"BitErrors",
        mxCreateDoubleScalar(double(bitErrors)));
    mxSetField(stats,0,"PacketErrors",
        mxCreateDoubleScalar(bitErrors > 0 ? 1 : 0));
    mxSetField(stats,0,"PayloadChecksum",
        mxCreateDoubleScalar(double(checksum)));
    mxSetField(stats,0,"ValidationMode",
        mxCreateDoubleScalar(s.validationMode));
    mxSetField(stats,0,"TotalGPUTimeMs",
        mxCreateDoubleScalar(totalMs));
    mxSetField(stats,0,"H2DTimeMs",
        mxCreateDoubleScalar(h2dMs));
    mxSetField(stats,0,"FrontendTimeMs",
        mxCreateDoubleScalar(frontendMs));
    mxSetField(stats,0,"DemapperTimeMs",
        mxCreateDoubleScalar(demapMs));
    mxSetField(stats,0,"MapCSIWeightTimeMs",
        mxCreateDoubleScalar(mapMs));
    mxSetField(stats,0,"LDPCDecodeTimeMs",
        mxCreateDoubleScalar(ldpcMs));
    mxSetField(stats,0,"DescrambleValidateTimeMs",
        mxCreateDoubleScalar(validateMs));
    mxSetField(stats,0,"ReturnedBulkOutputBytes",
        mxCreateDoubleScalar(0));
    mxSetField(stats,0,"Asynchronous",
        mxCreateLogicalScalar(true));
    mxSetField(stats,0,"BufferCount",
        mxCreateDoubleScalar(kNumSlots));

    return stats;
}

void reconstructCommand(
    int nlhs,mxArray* plhs[],int nrhs,const mxArray* prhs[])
{
    if (nrhs != 4)
        mexErrMsgIdAndTxt(
            "eht_receiver_cuda_e2e:InputCount",
            "reconstruct expects RxI, RxQ, and RxScale.");

    if (nlhs > 1)
        mexErrMsgIdAndTxt(
            "eht_receiver_cuda_e2e:OutputCount",
            "reconstruct returns one complex single waveform.");

    if (!mxIsInt16(prhs[1]) || mxIsComplex(prhs[1]) ||
        !mxIsInt16(prhs[2]) || mxIsComplex(prhs[2]))
        mexErrMsgIdAndTxt(
            "eht_receiver_cuda_e2e:Input",
            "RxI and RxQ must be real int16 arrays.");

    mwSize count = mxGetNumberOfElements(prhs[1]);
    if (count == 0 || mxGetNumberOfElements(prhs[2]) != count)
        mexErrMsgIdAndTxt(
            "eht_receiver_cuda_e2e:Input",
            "RxI and RxQ must be nonempty and have the same size.");

    double scale = scalar(prhs[3],"RxScale");
    if (!(scale > 0.0))
        mexErrMsgIdAndTxt(
            "eht_receiver_cuda_e2e:Input",
            "RxScale must be positive.");

    const int16_t* hI =
        static_cast<const int16_t*>(mxGetData(prhs[1]));
    const int16_t* hQ =
        static_cast<const int16_t*>(mxGetData(prhs[2]));

    int16_t* dI = nullptr;
    int16_t* dQ = nullptr;
    cufftComplex* dOutput = nullptr;

    size_t iqBytes = size_t(count)*sizeof(int16_t);
    size_t outputBytes = size_t(count)*sizeof(cufftComplex);

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dI),iqBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dQ),iqBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dOutput),outputBytes));

    CUDA_CHECK(cudaMemcpy(dI,hI,iqBytes,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dQ,hQ,iqBytes,cudaMemcpyHostToDevice));

    constexpr int threads = 256;
    int blocks = (int(count)+threads-1)/threads;
    reconstructWaveformKernel<<<blocks,threads>>>(
        dI,dQ,dOutput,float(1.0/scale),int(count));
    CUDA_CHECK(cudaGetLastError());

    std::vector<cufftComplex> hOutput(count);
    CUDA_CHECK(cudaMemcpy(
        hOutput.data(),dOutput,outputBytes,cudaMemcpyDeviceToHost));

    mwSize ndim = mxGetNumberOfDimensions(prhs[1]);
    const mwSize* dims = mxGetDimensions(prhs[1]);
    plhs[0] = mxCreateNumericArray(ndim,dims,mxSINGLE_CLASS,mxCOMPLEX);
    mxComplexSingle* out = mxGetComplexSingles(plhs[0]);

    for (mwSize i = 0; i < count; ++i) {
        out[i].real = hOutput[i].x;
        out[i].imag = hOutput[i].y;
    }

    CUDA_CHECK(cudaFree(dOutput));
    CUDA_CHECK(cudaFree(dQ));
    CUDA_CHECK(cudaFree(dI));
}

void packetDetectCommand(
    int nlhs,mxArray* plhs[],int nrhs,const mxArray* prhs[])
{
    if (nrhs != 4)
        mexErrMsgIdAndTxt(
            "eht_receiver_cuda_e2e:InputCount",
            "packetDetect expects RxI, RxQ, and RxScale.");

    if (nlhs > 1)
        mexErrMsgIdAndTxt(
            "eht_receiver_cuda_e2e:OutputCount",
            "packetDetect returns one coarse packet offset.");

    if (!mxIsInt16(prhs[1]) || mxIsComplex(prhs[1]) ||
        !mxIsInt16(prhs[2]) || mxIsComplex(prhs[2]))
        mexErrMsgIdAndTxt(
            "eht_receiver_cuda_e2e:Input",
            "RxI and RxQ must be real int16 arrays.");

    mwSize count = mxGetNumberOfElements(prhs[1]);
    if (count < 512 || mxGetNumberOfElements(prhs[2]) != count)
        mexErrMsgIdAndTxt(
            "eht_receiver_cuda_e2e:Input",
            "RxI and RxQ must have the same size and contain at least 512 samples.");

    double scale = scalar(prhs[3],"RxScale");
    if (!(scale > 0.0))
        mexErrMsgIdAndTxt(
            "eht_receiver_cuda_e2e:Input",
            "RxScale must be positive.");

    const int16_t* hI =
        static_cast<const int16_t*>(mxGetData(prhs[1]));
    const int16_t* hQ =
        static_cast<const int16_t*>(mxGetData(prhs[2]));

    int16_t* dI = nullptr;
    int16_t* dQ = nullptr;
    cufftComplex* dWaveform = nullptr;
    int* dDetectedOffset = nullptr;

    const size_t iqBytes = size_t(count)*sizeof(int16_t);
    const size_t waveformBytes = size_t(count)*sizeof(cufftComplex);

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dI),iqBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dQ),iqBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dWaveform),waveformBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dDetectedOffset),sizeof(int)));

    CUDA_CHECK(cudaMemcpy(dI,hI,iqBytes,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dQ,hQ,iqBytes,cudaMemcpyHostToDevice));

    constexpr int threads = 256;
    const int sampleBlocks = (int(count)+threads-1)/threads;
    reconstructWaveformKernel<<<sampleBlocks,threads>>>(
        dI,dQ,dWaveform,float(1.0/scale),int(count));
    CUDA_CHECK(cudaGetLastError());

    const int notFound = int(count);
    CUDA_CHECK(cudaMemcpy(
        dDetectedOffset,&notFound,sizeof(int),cudaMemcpyHostToDevice));

    const int candidateCount = int(count)-512+1;
    const int detectBlocks = (candidateCount+threads-1)/threads;
    packetDetectKernel<<<detectBlocks,threads>>>(
        dWaveform,int(count),dDetectedOffset);
    CUDA_CHECK(cudaGetLastError());

    int detectedOffset = notFound;
    CUDA_CHECK(cudaMemcpy(
        &detectedOffset,dDetectedOffset,sizeof(int),cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(dDetectedOffset));
    CUDA_CHECK(cudaFree(dWaveform));
    CUDA_CHECK(cudaFree(dQ));
    CUDA_CHECK(cudaFree(dI));

    if (detectedOffset == notFound)
        plhs[0] = mxCreateDoubleMatrix(0,0,mxREAL);
    else
        plhs[0] = mxCreateDoubleScalar(double(detectedOffset));
}

void coarseCFOCommand(
    int nlhs,mxArray* plhs[],int nrhs,const mxArray* prhs[])
{
    if (nrhs != 4)
        mexErrMsgIdAndTxt(
            "eht_receiver_cuda_e2e:InputCount",
            "coarseCFO expects RxI, RxQ, and RxScale.");

    if (nlhs > 1)
        mexErrMsgIdAndTxt(
            "eht_receiver_cuda_e2e:OutputCount",
            "coarseCFO returns one coarse CFO estimate in Hz.");

    if (!mxIsInt16(prhs[1]) || mxIsComplex(prhs[1]) ||
        !mxIsInt16(prhs[2]) || mxIsComplex(prhs[2]))
        mexErrMsgIdAndTxt(
            "eht_receiver_cuda_e2e:Input",
            "RxI and RxQ must be real int16 arrays.");

    mwSize count = mxGetNumberOfElements(prhs[1]);
    if (count < 2816 || mxGetNumberOfElements(prhs[2]) != count)
        mexErrMsgIdAndTxt(
            "eht_receiver_cuda_e2e:Input",
            "RxI and RxQ must have the same size and contain a complete L-STF.");

    double scale = scalar(prhs[3],"RxScale");
    if (!(scale > 0.0))
        mexErrMsgIdAndTxt(
            "eht_receiver_cuda_e2e:Input",
            "RxScale must be positive.");

    const int16_t* hI =
        static_cast<const int16_t*>(mxGetData(prhs[1]));
    const int16_t* hQ =
        static_cast<const int16_t*>(mxGetData(prhs[2]));

    int16_t* dI = nullptr;
    int16_t* dQ = nullptr;
    cufftComplex* dWaveform = nullptr;
    int* dDetectedOffset = nullptr;
    float* dPartialRe = nullptr;
    float* dPartialIm = nullptr;

    constexpr int lSTFLength = 2560;
    constexpr int lag = 256;
    constexpr int pairCount = lSTFLength-lag;
    constexpr float sampleRateHz = 320.0e6f;
    constexpr float twoPi = 6.2831853071795864769f;

    const size_t iqBytes = size_t(count)*sizeof(int16_t);
    const size_t waveformBytes = size_t(count)*sizeof(cufftComplex);
    const size_t partialBytes = size_t(pairCount)*sizeof(float);

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dI),iqBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dQ),iqBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dWaveform),waveformBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dDetectedOffset),sizeof(int)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dPartialRe),partialBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dPartialIm),partialBytes));

    CUDA_CHECK(cudaMemcpy(dI,hI,iqBytes,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dQ,hQ,iqBytes,cudaMemcpyHostToDevice));

    constexpr int threads = 256;
    const int sampleBlocks = (int(count)+threads-1)/threads;
    reconstructWaveformKernel<<<sampleBlocks,threads>>>(
        dI,dQ,dWaveform,float(1.0/scale),int(count));
    CUDA_CHECK(cudaGetLastError());

    const int notFound = int(count);
    CUDA_CHECK(cudaMemcpy(
        dDetectedOffset,&notFound,sizeof(int),cudaMemcpyHostToDevice));

    const int candidateCount = int(count)-512+1;
    const int detectBlocks = (candidateCount+threads-1)/threads;
    packetDetectKernel<<<detectBlocks,threads>>>(
        dWaveform,int(count),dDetectedOffset);
    CUDA_CHECK(cudaGetLastError());

    int detectedOffset = notFound;
    CUDA_CHECK(cudaMemcpy(
        &detectedOffset,dDetectedOffset,sizeof(int),cudaMemcpyDeviceToHost));

    if (detectedOffset == notFound || detectedOffset+lSTFLength > int(count)) {
        CUDA_CHECK(cudaFree(dPartialIm));
        CUDA_CHECK(cudaFree(dPartialRe));
        CUDA_CHECK(cudaFree(dDetectedOffset));
        CUDA_CHECK(cudaFree(dWaveform));
        CUDA_CHECK(cudaFree(dQ));
        CUDA_CHECK(cudaFree(dI));
        plhs[0] = mxCreateDoubleMatrix(0,0,mxREAL);
        return;
    }

    const int corrBlocks = (pairCount+threads-1)/threads;
    coarseCFOCorrelationKernel<<<corrBlocks,threads>>>(
        dWaveform,detectedOffset,int(count),dPartialRe,dPartialIm);
    CUDA_CHECK(cudaGetLastError());

    std::vector<float> hPartialRe(pairCount);
    std::vector<float> hPartialIm(pairCount);
    CUDA_CHECK(cudaMemcpy(
        hPartialRe.data(),dPartialRe,partialBytes,cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        hPartialIm.data(),dPartialIm,partialBytes,cudaMemcpyDeviceToHost));

    double corrRe = 0.0;
    double corrIm = 0.0;
    for (int i = 0; i < pairCount; ++i) {
        corrRe += double(hPartialRe[i]);
        corrIm += double(hPartialIm[i]);
    }

    const double phase = std::atan2(corrIm,corrRe);
    const double cfoHz = phase*double(sampleRateHz)/(2.0*3.14159265358979323846*double(lag));



    // Apply the correction on the GPU now so the corrected detected waveform
    // is already the natural input to Part 5.  Part 4 returns only the scalar
    // CFO estimate to keep the temporary API small.



    const int correctedCount = int(count)-detectedOffset;
    const int correctionBlocks = (correctedCount+threads-1)/threads;
    coarseCFOCorrectKernel<<<correctionBlocks,threads>>>(
        dWaveform,detectedOffset,int(count),float(cfoHz),sampleRateHz);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaFree(dPartialIm));
    CUDA_CHECK(cudaFree(dPartialRe));
    CUDA_CHECK(cudaFree(dDetectedOffset));
    CUDA_CHECK(cudaFree(dWaveform));
    CUDA_CHECK(cudaFree(dQ));
    CUDA_CHECK(cudaFree(dI));

    plhs[0] = mxCreateDoubleScalar(cfoHz);
}


void timingSyncCommand(
    int nlhs,mxArray* plhs[],int nrhs,const mxArray* prhs[])
{
    if (nrhs != 5)
        mexErrMsgIdAndTxt(
            "eht_receiver_cuda_e2e:InputCount",
            "timingSync expects RxI, RxQ, RxScale, and an L-LTF reference.");

    if (nlhs > 1)
        mexErrMsgIdAndTxt(
            "eht_receiver_cuda_e2e:OutputCount",
            "timingSync returns one final packet offset.");

    if (!mxIsInt16(prhs[1]) || mxIsComplex(prhs[1]) ||
        !mxIsInt16(prhs[2]) || mxIsComplex(prhs[2]))
        mexErrMsgIdAndTxt(
            "eht_receiver_cuda_e2e:Input",
            "RxI and RxQ must be real int16 arrays.");

    const mwSize count = mxGetNumberOfElements(prhs[1]);
    if (count < 6400 || mxGetNumberOfElements(prhs[2]) != count)
        mexErrMsgIdAndTxt(
            "eht_receiver_cuda_e2e:Input",
            "RxI and RxQ must have the same size and contain the legacy preamble.");

    const double scale = scalar(prhs[3],"RxScale");
    if (!(scale > 0.0))
        mexErrMsgIdAndTxt(
            "eht_receiver_cuda_e2e:Input",
            "RxScale must be positive.");

    constexpr int lltfLength = 2560;
    if (!mxIsSingle(prhs[4]) || !mxIsComplex(prhs[4]) ||
        int(mxGetNumberOfElements(prhs[4])) != lltfLength)
        mexErrMsgIdAndTxt(
            "eht_receiver_cuda_e2e:Input",
            "L-LTF reference must be complex single with 2560 samples.");

    const int16_t* hI =
        static_cast<const int16_t*>(mxGetData(prhs[1]));
    const int16_t* hQ =
        static_cast<const int16_t*>(mxGetData(prhs[2]));
    const mxComplexSingle* hReference = mxGetComplexSingles(prhs[4]);

    std::vector<cufftComplex> reference(lltfLength);
    double referenceEnergyDouble = 0.0;
    for (int i = 0; i < lltfLength; ++i) {
        reference[i].x = hReference[i].real;
        reference[i].y = hReference[i].imag;
        referenceEnergyDouble +=
            double(reference[i].x)*double(reference[i].x) +
            double(reference[i].y)*double(reference[i].y);
    }
    const float referenceEnergy = float(referenceEnergyDouble);
    if (!(referenceEnergy > 0.0f))
        mexErrMsgIdAndTxt(
            "eht_receiver_cuda_e2e:Input",
            "L-LTF reference energy must be positive.");

    int16_t* dI = nullptr;
    int16_t* dQ = nullptr;
    cufftComplex* dWaveform = nullptr;
    cufftComplex* dReference = nullptr;
    int* dDetectedOffset = nullptr;
    float* dCFOHz = nullptr;
    float* dMetrics = nullptr;
    int* dTimingCorrection = nullptr;
    int* dFinalPacketOffset = nullptr;

    constexpr int candidateCount = 513;
    const size_t iqBytes = size_t(count)*sizeof(int16_t);
    const size_t waveformBytes = size_t(count)*sizeof(cufftComplex);
    const size_t referenceBytes = size_t(lltfLength)*sizeof(cufftComplex);

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dI),iqBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dQ),iqBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dWaveform),waveformBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dReference),referenceBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dDetectedOffset),sizeof(int)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dCFOHz),sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dMetrics),
        size_t(candidateCount)*sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dTimingCorrection),sizeof(int)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dFinalPacketOffset),sizeof(int)));

    CUDA_CHECK(cudaMemcpy(dI,hI,iqBytes,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dQ,hQ,iqBytes,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        dReference,reference.data(),referenceBytes,cudaMemcpyHostToDevice));

    constexpr int threads = 256;
    const int sampleBlocks = (int(count)+threads-1)/threads;
    reconstructWaveformKernel<<<sampleBlocks,threads>>>(
        dI,dQ,dWaveform,float(1.0/scale),int(count));
    CUDA_CHECK(cudaGetLastError());

    const int notFound = int(count);
    CUDA_CHECK(cudaMemcpy(
        dDetectedOffset,&notFound,sizeof(int),cudaMemcpyHostToDevice));

    const int detectCandidates = int(count)-512+1;
    const int detectBlocks = (detectCandidates+threads-1)/threads;
    packetDetectKernel<<<detectBlocks,threads>>>(
        dWaveform,int(count),dDetectedOffset);
    CUDA_CHECK(cudaGetLastError());

    coarseCFOFinalizeKernel<<<1,threads>>>(
        dWaveform,dDetectedOffset,int(count),dCFOHz);
    CUDA_CHECK(cudaGetLastError());

    coarseCFOCorrectPointerKernel<<<sampleBlocks,threads>>>(
        dWaveform,dDetectedOffset,int(count),dCFOHz);
    CUDA_CHECK(cudaGetLastError());

    const int timingBlocks = (candidateCount+threads-1)/threads;
    timingMetricKernel<<<timingBlocks,threads>>>(
        dWaveform,dReference,dDetectedOffset,int(count),
        referenceEnergy,dMetrics);
    CUDA_CHECK(cudaGetLastError());

    timingArgMaxKernel<<<1,1>>>(
        dMetrics,dDetectedOffset,dTimingCorrection,dFinalPacketOffset);
    CUDA_CHECK(cudaGetLastError());

    int finalPacketOffset = notFound;
    CUDA_CHECK(cudaMemcpy(
        &finalPacketOffset,dFinalPacketOffset,sizeof(int),
        cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(dFinalPacketOffset));
    CUDA_CHECK(cudaFree(dTimingCorrection));
    CUDA_CHECK(cudaFree(dMetrics));
    CUDA_CHECK(cudaFree(dCFOHz));
    CUDA_CHECK(cudaFree(dDetectedOffset));
    CUDA_CHECK(cudaFree(dReference));
    CUDA_CHECK(cudaFree(dWaveform));
    CUDA_CHECK(cudaFree(dQ));
    CUDA_CHECK(cudaFree(dI));

    if (finalPacketOffset >= int(count))
        plhs[0] = mxCreateDoubleMatrix(0,0,mxREAL);
    else
        plhs[0] = mxCreateDoubleScalar(double(finalPacketOffset));
}

void fineCFOCommand(
    int nlhs,mxArray* plhs[],int nrhs,const mxArray* prhs[])
{
    if (nrhs != 5)
        mexErrMsgIdAndTxt(
            "eht_receiver_cuda_e2e:InputCount",
            "fineCFO expects RxI, RxQ, RxScale, and an L-LTF reference.");

    if (nlhs > 1)
        mexErrMsgIdAndTxt(
            "eht_receiver_cuda_e2e:OutputCount",
            "fineCFO returns one fine CFO estimate in Hz.");

    if (!mxIsInt16(prhs[1]) || mxIsComplex(prhs[1]) ||
        !mxIsInt16(prhs[2]) || mxIsComplex(prhs[2]))
        mexErrMsgIdAndTxt(
            "eht_receiver_cuda_e2e:Input",
            "RxI and RxQ must be real int16 arrays.");

    const mwSize count = mxGetNumberOfElements(prhs[1]);
    if (count < 6400 || mxGetNumberOfElements(prhs[2]) != count)
        mexErrMsgIdAndTxt(
            "eht_receiver_cuda_e2e:Input",
            "RxI and RxQ must have the same size and contain the legacy preamble.");

    const double scale = scalar(prhs[3],"RxScale");
    if (!(scale > 0.0))
        mexErrMsgIdAndTxt(
            "eht_receiver_cuda_e2e:Input",
            "RxScale must be positive.");

    constexpr int lltfLength = 2560;
    if (!mxIsSingle(prhs[4]) || !mxIsComplex(prhs[4]) ||
        int(mxGetNumberOfElements(prhs[4])) != lltfLength)
        mexErrMsgIdAndTxt(
            "eht_receiver_cuda_e2e:Input",
            "L-LTF reference must be complex single with 2560 samples.");

    const int16_t* hI =
        static_cast<const int16_t*>(mxGetData(prhs[1]));
    const int16_t* hQ =
        static_cast<const int16_t*>(mxGetData(prhs[2]));
    const mxComplexSingle* hReference = mxGetComplexSingles(prhs[4]);

    std::vector<cufftComplex> reference(lltfLength);
    double referenceEnergyDouble = 0.0;
    for (int i = 0; i < lltfLength; ++i) {
        reference[i].x = hReference[i].real;
        reference[i].y = hReference[i].imag;
        referenceEnergyDouble +=
            double(reference[i].x)*double(reference[i].x) +
            double(reference[i].y)*double(reference[i].y);
    }
    const float referenceEnergy = float(referenceEnergyDouble);
    if (!(referenceEnergy > 0.0f))
        mexErrMsgIdAndTxt(
            "eht_receiver_cuda_e2e:Input",
            "L-LTF reference energy must be positive.");

    int16_t* dI = nullptr;
    int16_t* dQ = nullptr;
    cufftComplex* dWaveform = nullptr;
    cufftComplex* dReference = nullptr;
    int* dDetectedOffset = nullptr;
    float* dCoarseCFOHz = nullptr;
    float* dMetrics = nullptr;
    int* dTimingCorrection = nullptr;
    int* dFinalPacketOffset = nullptr;
    float* dFineCFOHz = nullptr;

    constexpr int candidateCount = 513;
    const size_t iqBytes = size_t(count)*sizeof(int16_t);
    const size_t waveformBytes = size_t(count)*sizeof(cufftComplex);
    const size_t referenceBytes = size_t(lltfLength)*sizeof(cufftComplex);

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dI),iqBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dQ),iqBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dWaveform),waveformBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dReference),referenceBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dDetectedOffset),sizeof(int)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dCoarseCFOHz),sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dMetrics),
        size_t(candidateCount)*sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dTimingCorrection),sizeof(int)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dFinalPacketOffset),sizeof(int)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dFineCFOHz),sizeof(float)));

    CUDA_CHECK(cudaMemcpy(dI,hI,iqBytes,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dQ,hQ,iqBytes,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        dReference,reference.data(),referenceBytes,cudaMemcpyHostToDevice));

    constexpr int threads = 256;
    const int sampleBlocks = (int(count)+threads-1)/threads;
    reconstructWaveformKernel<<<sampleBlocks,threads>>>(
        dI,dQ,dWaveform,float(1.0/scale),int(count));
    CUDA_CHECK(cudaGetLastError());

    const int notFound = int(count);
    CUDA_CHECK(cudaMemcpy(
        dDetectedOffset,&notFound,sizeof(int),cudaMemcpyHostToDevice));

    const int detectCandidates = int(count)-512+1;
    const int detectBlocks = (detectCandidates+threads-1)/threads;
    packetDetectKernel<<<detectBlocks,threads>>>(
        dWaveform,int(count),dDetectedOffset);
    CUDA_CHECK(cudaGetLastError());

    coarseCFOFinalizeKernel<<<1,threads>>>(
        dWaveform,dDetectedOffset,int(count),dCoarseCFOHz);
    CUDA_CHECK(cudaGetLastError());

    coarseCFOCorrectPointerKernel<<<sampleBlocks,threads>>>(
        dWaveform,dDetectedOffset,int(count),dCoarseCFOHz);
    CUDA_CHECK(cudaGetLastError());

    const int timingBlocks = (candidateCount+threads-1)/threads;
    timingMetricKernel<<<timingBlocks,threads>>>(
        dWaveform,dReference,dDetectedOffset,int(count),
        referenceEnergy,dMetrics);
    CUDA_CHECK(cudaGetLastError());

    timingArgMaxKernel<<<1,1>>>(
        dMetrics,dDetectedOffset,dTimingCorrection,dFinalPacketOffset);
    CUDA_CHECK(cudaGetLastError());

    fineCFOFinalizeKernel<<<1,threads>>>(
        dWaveform,dFinalPacketOffset,int(count),dFineCFOHz);
    CUDA_CHECK(cudaGetLastError());

    // Apply the residual correction now so this development path is ready
    // for Part 7 without a host round trip.
    fineCFOCorrectPointerKernel<<<sampleBlocks,threads>>>(
        dWaveform,dFinalPacketOffset,int(count),dFineCFOHz);
    CUDA_CHECK(cudaGetLastError());

    float fineCFOHz = 0.0f;
    int finalPacketOffset = notFound;
    CUDA_CHECK(cudaMemcpy(
        &finalPacketOffset,dFinalPacketOffset,sizeof(int),cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        &fineCFOHz,dFineCFOHz,sizeof(float),cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(dFineCFOHz));
    CUDA_CHECK(cudaFree(dFinalPacketOffset));
    CUDA_CHECK(cudaFree(dTimingCorrection));
    CUDA_CHECK(cudaFree(dMetrics));
    CUDA_CHECK(cudaFree(dCoarseCFOHz));
    CUDA_CHECK(cudaFree(dDetectedOffset));
    CUDA_CHECK(cudaFree(dReference));
    CUDA_CHECK(cudaFree(dWaveform));
    CUDA_CHECK(cudaFree(dQ));
    CUDA_CHECK(cudaFree(dI));

    if (finalPacketOffset >= int(count))
        plhs[0] = mxCreateDoubleMatrix(0,0,mxREAL);
    else
        plhs[0] = mxCreateDoubleScalar(double(fineCFOHz));
}


void part7Command(
    int nlhs,mxArray* plhs[],int nrhs,const mxArray* prhs[])
{
    if (nrhs != 5)
        mexErrMsgIdAndTxt(
            "eht_receiver_cuda_e2e:InputCount",
            "part7 expects RxI, RxQ, RxScale, and an L-LTF reference.");

    if (nlhs > 3)
        mexErrMsgIdAndTxt(
            "eht_receiver_cuda_e2e:OutputCount",
            "part7 returns noiseVariance, EHT-LTF, and EHT-Data.");

    if (!mxIsInt16(prhs[1]) || mxIsComplex(prhs[1]) ||
        !mxIsInt16(prhs[2]) || mxIsComplex(prhs[2]))
        mexErrMsgIdAndTxt(
            "eht_receiver_cuda_e2e:Input",
            "RxI and RxQ must be real int16 arrays.");

    const mwSize count = mxGetNumberOfElements(prhs[1]);
    if (count < kPacketLengthSamples || mxGetNumberOfElements(prhs[2]) != count)
        mexErrMsgIdAndTxt(
            "eht_receiver_cuda_e2e:Input",
            "RxI and RxQ must contain the complete stored packet waveform.");

    const double scale = scalar(prhs[3],"RxScale");
    if (!(scale > 0.0))
        mexErrMsgIdAndTxt(
            "eht_receiver_cuda_e2e:Input",
            "RxScale must be positive.");

    constexpr int lltfLength = 2560;
    constexpr int ehtLTFLength = kEHTLTFLength;
    constexpr int ehtDataLength = kEHTDataLength;
    if (!mxIsSingle(prhs[4]) || !mxIsComplex(prhs[4]) ||
        int(mxGetNumberOfElements(prhs[4])) != lltfLength)
        mexErrMsgIdAndTxt(
            "eht_receiver_cuda_e2e:Input",
            "L-LTF reference must be complex single with 2560 samples.");

    const int16_t* hI = static_cast<const int16_t*>(mxGetData(prhs[1]));
    const int16_t* hQ = static_cast<const int16_t*>(mxGetData(prhs[2]));
    const mxComplexSingle* hReference = mxGetComplexSingles(prhs[4]);

    std::vector<cufftComplex> reference(lltfLength);
    double referenceEnergyDouble = 0.0;
    for (int i = 0; i < lltfLength; ++i) {
        reference[i].x = hReference[i].real;
        reference[i].y = hReference[i].imag;
        referenceEnergyDouble +=
            double(reference[i].x)*double(reference[i].x) +
            double(reference[i].y)*double(reference[i].y);
    }
    const float referenceEnergy = float(referenceEnergyDouble);
    if (!(referenceEnergy > 0.0f))
        mexErrMsgIdAndTxt(
            "eht_receiver_cuda_e2e:Input",
            "L-LTF reference energy must be positive.");

    int16_t* dI = nullptr;
    int16_t* dQ = nullptr;
    cufftComplex* dWaveform = nullptr;
    cufftComplex* dReference = nullptr;
    int* dDetectedOffset = nullptr;
    float* dCoarseCFOHz = nullptr;
    float* dMetrics = nullptr;
    int* dTimingCorrection = nullptr;
    int* dFinalPacketOffset = nullptr;
    float* dFineCFOHz = nullptr;
    float* dNoiseVariance = nullptr;
    cufftComplex* dEHTLTF = nullptr;
    cufftComplex* dEHTData = nullptr;

    constexpr int candidateCount = 513;
    const size_t iqBytes = size_t(count)*sizeof(int16_t);
    const size_t waveformBytes = size_t(count)*sizeof(cufftComplex);
    const size_t referenceBytes = size_t(lltfLength)*sizeof(cufftComplex);

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dI),iqBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dQ),iqBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dWaveform),waveformBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dReference),referenceBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dDetectedOffset),sizeof(int)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dCoarseCFOHz),sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dMetrics),
        size_t(candidateCount)*sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dTimingCorrection),sizeof(int)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dFinalPacketOffset),sizeof(int)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dFineCFOHz),sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dNoiseVariance),sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dEHTLTF),
        size_t(ehtLTFLength)*sizeof(cufftComplex)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dEHTData),
        size_t(ehtDataLength)*sizeof(cufftComplex)));

    CUDA_CHECK(cudaMemcpy(dI,hI,iqBytes,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dQ,hQ,iqBytes,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        dReference,reference.data(),referenceBytes,cudaMemcpyHostToDevice));

    constexpr int threads = 256;
    const int sampleBlocks = (int(count)+threads-1)/threads;
    reconstructWaveformKernel<<<sampleBlocks,threads>>>(
        dI,dQ,dWaveform,float(1.0/scale),int(count));
    CUDA_CHECK(cudaGetLastError());

    const int notFound = int(count);
    CUDA_CHECK(cudaMemcpy(
        dDetectedOffset,&notFound,sizeof(int),cudaMemcpyHostToDevice));

    const int detectCandidates = int(count)-512+1;
    const int detectBlocks = (detectCandidates+threads-1)/threads;
    packetDetectKernel<<<detectBlocks,threads>>>(
        dWaveform,int(count),dDetectedOffset);
    CUDA_CHECK(cudaGetLastError());

    coarseCFOFinalizeKernel<<<1,threads>>>(
        dWaveform,dDetectedOffset,int(count),dCoarseCFOHz);
    CUDA_CHECK(cudaGetLastError());

    coarseCFOCorrectPointerKernel<<<sampleBlocks,threads>>>(
        dWaveform,dDetectedOffset,int(count),dCoarseCFOHz);
    CUDA_CHECK(cudaGetLastError());

    const int timingBlocks = (candidateCount+threads-1)/threads;
    timingMetricKernel<<<timingBlocks,threads>>>(
        dWaveform,dReference,dDetectedOffset,int(count),
        referenceEnergy,dMetrics);
    CUDA_CHECK(cudaGetLastError());

    timingArgMaxKernel<<<1,1>>>(
        dMetrics,dDetectedOffset,dTimingCorrection,dFinalPacketOffset);
    CUDA_CHECK(cudaGetLastError());

    // Match the MATLAB synchronization phase convention before fine CFO.


    rebaseCoarseCFOPhaseKernel<<<sampleBlocks,threads>>>(
        dWaveform,dFinalPacketOffset,dTimingCorrection,int(count),dCoarseCFOHz);
    CUDA_CHECK(cudaGetLastError());

    fineCFOFinalizeKernel<<<1,threads>>>(
        dWaveform,dFinalPacketOffset,int(count),dFineCFOHz);
    CUDA_CHECK(cudaGetLastError());

    fineCFOCorrectPointerKernel<<<sampleBlocks,threads>>>(
        dWaveform,dFinalPacketOffset,int(count),dFineCFOHz);
    CUDA_CHECK(cudaGetLastError());

    noiseVarianceFinalizeKernel<<<1,threads>>>(
        dWaveform,dFinalPacketOffset,int(count),dNoiseVariance);
    CUDA_CHECK(cudaGetLastError());

    const int extractBlocks = (ehtDataLength+threads-1)/threads;
    extractEHTFieldsKernel<<<extractBlocks,threads>>>(
        dWaveform,dFinalPacketOffset,int(count),dEHTLTF,dEHTData);
    CUDA_CHECK(cudaGetLastError());

    float noiseVariance = 0.0f;
    int finalPacketOffset = notFound;
    CUDA_CHECK(cudaMemcpy(
        &finalPacketOffset,dFinalPacketOffset,sizeof(int),cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        &noiseVariance,dNoiseVariance,sizeof(float),cudaMemcpyDeviceToHost));

    std::vector<cufftComplex> hEHTLTF(ehtLTFLength);
    std::vector<cufftComplex> hEHTData(ehtDataLength);
    CUDA_CHECK(cudaMemcpy(hEHTLTF.data(),dEHTLTF,
        size_t(ehtLTFLength)*sizeof(cufftComplex),cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hEHTData.data(),dEHTData,
        size_t(ehtDataLength)*sizeof(cufftComplex),cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(dEHTData));
    CUDA_CHECK(cudaFree(dEHTLTF));
    CUDA_CHECK(cudaFree(dNoiseVariance));
    CUDA_CHECK(cudaFree(dFineCFOHz));
    CUDA_CHECK(cudaFree(dFinalPacketOffset));
    CUDA_CHECK(cudaFree(dTimingCorrection));
    CUDA_CHECK(cudaFree(dMetrics));
    CUDA_CHECK(cudaFree(dCoarseCFOHz));
    CUDA_CHECK(cudaFree(dDetectedOffset));
    CUDA_CHECK(cudaFree(dReference));
    CUDA_CHECK(cudaFree(dWaveform));
    CUDA_CHECK(cudaFree(dQ));
    CUDA_CHECK(cudaFree(dI));

    if (finalPacketOffset >= int(count))
        mexErrMsgIdAndTxt(
            "eht_receiver_cuda_e2e:Sync",
            "GPU synchronization failed before Part 7 extraction.");

    if (nlhs >= 1)
        plhs[0] = mxCreateDoubleScalar(double(noiseVariance));

    if (nlhs >= 2) {
        plhs[1] = mxCreateNumericMatrix(
            ehtLTFLength,1,mxSINGLE_CLASS,mxCOMPLEX);
        mxComplexSingle* out = mxGetComplexSingles(plhs[1]);
        for (int i = 0; i < ehtLTFLength; ++i) {
            out[i].real = hEHTLTF[i].x;
            out[i].imag = hEHTLTF[i].y;
        }
    }

    if (nlhs >= 3) {
        plhs[2] = mxCreateNumericMatrix(
            ehtDataLength,1,mxSINGLE_CLASS,mxCOMPLEX);
        mxComplexSingle* out = mxGetComplexSingles(plhs[2]);
        for (int i = 0; i < ehtDataLength; ++i) {
            out[i].real = hEHTData[i].x;
            out[i].imag = hEHTData[i].y;
        }
    }
}


void e2eCommand(
    int nlhs,mxArray* plhs[],int nrhs,const mxArray* prhs[])
{
    // command + 14 inputs:
    // RxI,RxQ,RxScale,llrScale,lltfReference,
    // ltfFFTIndices,dataFFTIndices,ltfFFTStart,dataFFTStarts,
    // knownLTF,pilotIndices,pilotReference,dataIndices,referencePayloadBits
    if (nrhs != 15)
        mexErrMsgIdAndTxt(
            "eht_receiver_cuda_e2e:InputCount",
            "e2e expects command plus 14 inputs.");
    if (nlhs > 1)
        mexErrMsgIdAndTxt(
            "eht_receiver_cuda_e2e:OutputCount",
            "e2e returns one validation statistics structure.");

    ensureInitialized();

    if (!mxIsInt16(prhs[1]) || mxIsComplex(prhs[1]) ||
        !mxIsInt16(prhs[2]) || mxIsComplex(prhs[2]))
        mexErrMsgIdAndTxt(
            "eht_receiver_cuda_e2e:Input",
            "RxI and RxQ must be real int16 arrays.");

    const mwSize count = mxGetNumberOfElements(prhs[1]);
    if (count < kPacketLengthSamples || mxGetNumberOfElements(prhs[2]) != count)
        mexErrMsgIdAndTxt(
            "eht_receiver_cuda_e2e:Input",
            "RxI and RxQ must contain one complete stored packet waveform.");

    const double scale = scalar(prhs[3],"RxScale");
    const double llrScale = scalar(prhs[4],"llrScale");
    if (!(scale > 0.0) || !(llrScale > 0.0))
        mexErrMsgIdAndTxt(
            "eht_receiver_cuda_e2e:Input",
            "RxScale and llrScale must be positive.");

    constexpr int lltfLength = 2560;
    constexpr int ehtLTFLength = kEHTLTFLength;
    constexpr int ehtDataLength = kEHTDataLength;
    if (!mxIsSingle(prhs[5]) || !mxIsComplex(prhs[5]) ||
        int(mxGetNumberOfElements(prhs[5])) != lltfLength)
        mexErrMsgIdAndTxt(
            "eht_receiver_cuda_e2e:Input",
            "L-LTF reference must be complex single with 2560 samples.");

    const int ltfFFTStart =
        static_cast<int>(scalar(prhs[8],"ltfFFTStart"));
    if (!mxIsNumeric(prhs[9]) || mxIsComplex(prhs[9]) ||
        mxGetNumberOfElements(prhs[9]) != kNumDataSymbols)
        mexErrMsgIdAndTxt(
            "eht_receiver_cuda_e2e:Input",
            "dataFFTStarts must contain %d real values.",kNumDataSymbols);

    int dataStarts[kNumDataSymbols];
    for (int sym = 0; sym < kNumDataSymbols; ++sym) {
        const double raw = mxIsDouble(prhs[9])
            ? mxGetDoubles(prhs[9])[sym]
            : double(mxGetSingles(prhs[9])[sym]);
        dataStarts[sym] = int(raw);
        if (double(dataStarts[sym]) != raw || dataStarts[sym] < 1 ||
            dataStarts[sym]-1+kFFTLength > ehtDataLength)
            mexErrMsgIdAndTxt(
                "eht_receiver_cuda_e2e:Input",
                "FFT windows are outside the extracted EHT fields.");
    }
    if (ltfFFTStart < 1 || ltfFFTStart-1+kFFTLength > ehtLTFLength)
        mexErrMsgIdAndTxt(
            "eht_receiver_cuda_e2e:Input",
            "FFT windows are outside the extracted EHT fields.");

    initializeFrontendMetadata(
        prhs[6],  // ltfFFTIndices
        prhs[7],  // dataFFTIndices
        prhs[10], // knownLTF
        prhs[11], // pilotIndices
        prhs[12], // pilotReference
        prhs[13]  // dataIndices
    );

    const mxArray* referenceBits = prhs[14];
    if (mxIsComplex(referenceBits) ||
        mxGetNumberOfElements(referenceBits) != kPayloadBits ||
        !(mxIsInt8(referenceBits) || mxIsLogical(referenceBits)))
        mexErrMsgIdAndTxt(
            "eht_receiver_cuda_e2e:Input",
            "referencePayloadBits must be int8/logical with %d elements.",kPayloadBits);

    Slot* slot = findFreeSlot();
    if (!slot)
        mexErrMsgIdAndTxt(
            "eht_receiver_cuda_e2e:NoFreeSlot",
            "No free CUDA receiver slot is available.");

    if (mxIsInt8(referenceBits)) {
        const int8_t* p =
            static_cast<const int8_t*>(mxGetData(referenceBits));
        for (int i = 0; i < kPayloadBits; ++i) {
            if (p[i] != 0 && p[i] != 1)
                mexErrMsgIdAndTxt(
                    "eht_receiver_cuda_e2e:Input",
                    "referencePayloadBits must contain only 0 or 1.");
            slot->hReferenceBits[i] = p[i];
        }
    } else {
        const mxLogical* p = mxGetLogicals(referenceBits);
        for (int i = 0; i < kPayloadBits; ++i)
            slot->hReferenceBits[i] = p[i] ? int8_t(1) : int8_t(0);
    }

    const int16_t* hI = static_cast<const int16_t*>(mxGetData(prhs[1]));
    const int16_t* hQ = static_cast<const int16_t*>(mxGetData(prhs[2]));
    const mxComplexSingle* hReference = mxGetComplexSingles(prhs[5]);

    std::vector<cufftComplex> reference(lltfLength);
    double referenceEnergyDouble = 0.0;
    for (int i = 0; i < lltfLength; ++i) {
        reference[i].x = hReference[i].real;
        reference[i].y = hReference[i].imag;
        referenceEnergyDouble +=
            double(reference[i].x)*double(reference[i].x) +
            double(reference[i].y)*double(reference[i].y);
    }
    const float referenceEnergy = float(referenceEnergyDouble);
    if (!(referenceEnergy > 0.0f))
        mexErrMsgIdAndTxt(
            "eht_receiver_cuda_e2e:Input",
            "L-LTF reference energy must be positive.");

    int16_t* dRawI = nullptr;
    int16_t* dRawQ = nullptr;
    cufftComplex* dWaveform = nullptr;
    cufftComplex* dTimingReference = nullptr;
    int* dDetectedOffset = nullptr;
    float* dCoarseCFOHz = nullptr;
    float* dMetrics = nullptr;
    int* dTimingCorrection = nullptr;
    int* dFinalPacketOffset = nullptr;
    float* dFineCFOHz = nullptr;
    float* dNoiseVariance = nullptr;
    cufftComplex* dEHTLTF = nullptr;
    cufftComplex* dEHTData = nullptr;

    constexpr int candidateCount = 513;
    const size_t iqBytes = size_t(count)*sizeof(int16_t);
    const size_t waveformBytes = size_t(count)*sizeof(cufftComplex);
    const size_t referenceBytes = size_t(lltfLength)*sizeof(cufftComplex);

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dRawI),iqBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dRawQ),iqBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dWaveform),waveformBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dTimingReference),referenceBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dDetectedOffset),sizeof(int)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dCoarseCFOHz),sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dMetrics),
        size_t(candidateCount)*sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dTimingCorrection),sizeof(int)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dFinalPacketOffset),sizeof(int)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dFineCFOHz),sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dNoiseVariance),sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dEHTLTF),
        size_t(ehtLTFLength)*sizeof(cufftComplex)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dEHTData),
        size_t(ehtDataLength)*sizeof(cufftComplex)));

    slot->ticket = nextTicket++;
    slot->validationMode = 1;
    slot->occupied = true;
    *slot->hChecksum = 0;
    *slot->hBitErrors = 0;

    CUDA_CHECK(cudaEventRecord(slot->start,slot->stream));
    CUDA_CHECK(cudaMemcpyAsync(
        dRawI,hI,iqBytes,cudaMemcpyHostToDevice,slot->stream));
    CUDA_CHECK(cudaMemcpyAsync(
        dRawQ,hQ,iqBytes,cudaMemcpyHostToDevice,slot->stream));
    CUDA_CHECK(cudaMemcpyAsync(
        dTimingReference,reference.data(),referenceBytes,
        cudaMemcpyHostToDevice,slot->stream));
    CUDA_CHECK(cudaMemcpyAsync(
        slot->dReferenceBits,slot->hReferenceBits,
        size_t(kPayloadBits)*sizeof(int8_t),
        cudaMemcpyHostToDevice,slot->stream));
    CUDA_CHECK(cudaMemsetAsync(
        slot->dChecksum,0,sizeof(unsigned long long),slot->stream));
    CUDA_CHECK(cudaMemsetAsync(
        slot->dBitErrors,0,sizeof(unsigned int),slot->stream));
    CUDA_CHECK(cudaEventRecord(slot->afterH2D,slot->stream));

    constexpr int threads = 256;
    const int sampleBlocks = (int(count)+threads-1)/threads;
    reconstructWaveformKernel<<<sampleBlocks,threads,0,slot->stream>>>(
        dRawI,dRawQ,dWaveform,float(1.0/scale),int(count));
    CUDA_CHECK(cudaGetLastError());

    const int notFound = int(count);
    CUDA_CHECK(cudaMemcpyAsync(
        dDetectedOffset,&notFound,sizeof(int),
        cudaMemcpyHostToDevice,slot->stream));

    const int detectCandidates = int(count)-512+1;
    const int detectBlocks = (detectCandidates+threads-1)/threads;
    packetDetectKernel<<<detectBlocks,threads,0,slot->stream>>>(
        dWaveform,int(count),dDetectedOffset);
    CUDA_CHECK(cudaGetLastError());

    coarseCFOFinalizeKernel<<<1,threads,0,slot->stream>>>(
        dWaveform,dDetectedOffset,int(count),dCoarseCFOHz);
    CUDA_CHECK(cudaGetLastError());

    coarseCFOCorrectPointerKernel<<<sampleBlocks,threads,0,slot->stream>>>(
        dWaveform,dDetectedOffset,int(count),dCoarseCFOHz);
    CUDA_CHECK(cudaGetLastError());

    const int timingBlocks = (candidateCount+threads-1)/threads;
    timingMetricKernel<<<timingBlocks,threads,0,slot->stream>>>(
        dWaveform,dTimingReference,dDetectedOffset,int(count),
        referenceEnergy,dMetrics);
    CUDA_CHECK(cudaGetLastError());

    timingArgMaxKernel<<<1,1,0,slot->stream>>>(
        dMetrics,dDetectedOffset,dTimingCorrection,dFinalPacketOffset);
    CUDA_CHECK(cudaGetLastError());

    rebaseCoarseCFOPhaseKernel<<<sampleBlocks,threads,0,slot->stream>>>(
        dWaveform,dFinalPacketOffset,dTimingCorrection,int(count),dCoarseCFOHz);
    CUDA_CHECK(cudaGetLastError());

    fineCFOFinalizeKernel<<<1,threads,0,slot->stream>>>(
        dWaveform,dFinalPacketOffset,int(count),dFineCFOHz);
    CUDA_CHECK(cudaGetLastError());

    fineCFOCorrectPointerKernel<<<sampleBlocks,threads,0,slot->stream>>>(
        dWaveform,dFinalPacketOffset,int(count),dFineCFOHz);
    CUDA_CHECK(cudaGetLastError());

    noiseVarianceFinalizeKernel<<<1,threads,0,slot->stream>>>(
        dWaveform,dFinalPacketOffset,int(count),dNoiseVariance);
    CUDA_CHECK(cudaGetLastError());

    const int extractBlocks = (ehtDataLength+threads-1)/threads;
    extractEHTFieldsKernel<<<extractBlocks,threads,0,slot->stream>>>(
        dWaveform,dFinalPacketOffset,int(count),dEHTLTF,dEHTData);
    CUDA_CHECK(cudaGetLastError());

    // Only one scalar is staged to host control memory here. The bulk
    // EHT-LTF and EHT-Data arrays stay resident on the GPU and are copied
    // device-to-device into the existing backend FFT input buffers.


    float noiseVariance = 0.0f;
    int finalPacketOffset = notFound;
    CUDA_CHECK(cudaMemcpyAsync(
        &noiseVariance,dNoiseVariance,sizeof(float),
        cudaMemcpyDeviceToHost,slot->stream));
    CUDA_CHECK(cudaMemcpyAsync(
        &finalPacketOffset,dFinalPacketOffset,sizeof(int),
        cudaMemcpyDeviceToHost,slot->stream));
    CUDA_CHECK(cudaStreamSynchronize(slot->stream));

    if (finalPacketOffset >= int(count) || !(noiseVariance > 0.0f)) {
        slot->occupied = false;
        slot->ticket = 0;
        slot->validationMode = 0;
        mexErrMsgIdAndTxt(
            "eht_receiver_cuda_e2e:Sync",
            "GPU synchronization/noise estimation failed before backend integration.");
    }

    CUDA_CHECK(cudaMemcpyAsync(
        slot->dLTFInput,dEHTLTF+(ltfFFTStart-1),
        size_t(kLTFInputCount)*sizeof(cufftComplex),
        cudaMemcpyDeviceToDevice,slot->stream));
    for (int sym = 0; sym < kNumDataSymbols; ++sym) {
        CUDA_CHECK(cudaMemcpyAsync(
            slot->dDataInput+size_t(sym)*kFFTLength,
            dEHTData+(dataStarts[sym]-1),
            size_t(kFFTLength)*sizeof(cufftComplex),
            cudaMemcpyDeviceToDevice,slot->stream));
    }

    CUFFT_CHECK(cufftExecC2C(
        slot->ltfPlan,slot->dLTFInput,
        slot->dLTFOutput,CUFFT_FORWARD));
    CUFFT_CHECK(cufftExecC2C(
        slot->dataPlan,slot->dDataInput,
        slot->dDataOutput,CUFFT_FORWARD));

    int blocks = (kActiveToneCount+threads-1)/threads;
    extractActiveKernel<<<blocks,threads,0,slot->stream>>>(
        slot->dLTFOutput,slot->dDataOutput,
        sharedData.ltfFFTIndices,sharedData.dataFFTIndices,
        slot->dLTFActive,slot->dDataActive);
    CUDA_CHECK(cudaGetLastError());

    estimateChannelKernel<<<blocks,threads,0,slot->stream>>>(
        slot->dLTFActive,sharedData.knownLTF,
        slot->dChannelEstimate);
    CUDA_CHECK(cudaGetLastError());

    estimatePilotRotationKernel<<<
        kNumDataSymbols,256,256*sizeof(cufftComplex),slot->stream>>>(
            slot->dDataActive,slot->dChannelEstimate,
            sharedData.pilotIndices,sharedData.pilotReference,
            sharedData.pilotCount,slot->dPilotRotation);
    CUDA_CHECK(cudaGetLastError());

    blocks = (kActiveToneCount*kNumDataSymbols+threads-1)/threads;
    equalizeKernel<<<blocks,threads,0,slot->stream>>>(
        slot->dDataActive,slot->dChannelEstimate,
        slot->dPilotRotation,noiseVariance,
        slot->dEqualizedActive);
    CUDA_CHECK(cudaGetLastError());

    segmentQuantizeCSIKernel<<<4,256,0,slot->stream>>>(
        slot->dEqualizedActive,slot->dChannelEstimate,
        sharedData.dataIndices,noiseVariance,
        slot->dI,slot->dQ,slot->dCSI,
        slot->dInverseQuantScales);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(slot->afterFrontend,slot->stream));

    blocks = (kTotalSymbols+threads-1)/threads;
    qam4096DemapKernel<<<blocks,threads,0,slot->stream>>>(
        slot->dI,slot->dQ,slot->dExternalLLR,kTotalSymbols,
        slot->dInverseQuantScales,float(1.0/noiseVariance),
        float(llrScale));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(slot->afterDemap,slot->stream));

    blocks = (kEncodedLength+threads-1)/threads;
    mapWeightKernel<<<blocks,threads,0,slot->stream>>>(
        slot->dExternalLLR,slot->dCSI,
        sharedData.encodedSourceIndex,float(1.0/llrScale),
        slot->dEncoded);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(slot->afterMap,slot->stream));

    dim3 reconBlock(256);
    dim3 reconGrid(kNumCodewords,(N+reconBlock.x-1)/reconBlock.x);
    reconstructKernel<<<reconGrid,reconBlock,0,slot->stream>>>(
        slot->dEncoded,sharedData.payloadBits,
        sharedData.punctureBits,sharedData.repeatBits,
        slot->dBeliefs,slot->dMessages);
    CUDA_CHECK(cudaGetLastError());

    layeredNMSKernel<<<kNumCodewords,128,0,slot->stream>>>(
        slot->dBeliefs,slot->dMessages);
    CUDA_CHECK(cudaGetLastError());

    dim3 decisionBlock(256);
    dim3 decisionGrid(kNumCodewords,(K+decisionBlock.x-1)/decisionBlock.x);
    hardDecisionKernel<<<decisionGrid,decisionBlock,0,slot->stream>>>(
        slot->dBeliefs,sharedData.payloadBits,slot->dDecoded);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(slot->afterLDPC,slot->stream));

    deriveScramblerSequenceKernel<<<1,1,0,slot->stream>>>(
        slot->dDecoded,slot->dScrambleSequence,
        slot->dDescrambleEnabled);
    CUDA_CHECK(cudaGetLastError());

    blocks = (kPayloadBits+threads-1)/threads;
    validatePayloadKernel<<<blocks,threads,0,slot->stream>>>(
        slot->dDecoded,slot->dScrambleSequence,
        slot->dDescrambleEnabled,1,
        slot->dReferenceBits,slot->dChecksum,slot->dBitErrors);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(slot->afterValidate,slot->stream));

    CUDA_CHECK(cudaMemcpyAsync(
        slot->hChecksum,slot->dChecksum,
        sizeof(unsigned long long),
        cudaMemcpyDeviceToHost,slot->stream));
    CUDA_CHECK(cudaMemcpyAsync(
        slot->hBitErrors,slot->dBitErrors,
        sizeof(unsigned int),
        cudaMemcpyDeviceToHost,slot->stream));
    CUDA_CHECK(cudaEventRecord(slot->done,slot->stream));
    CUDA_CHECK(cudaEventSynchronize(slot->done));

    CUDA_CHECK(cudaFree(dEHTData));
    CUDA_CHECK(cudaFree(dEHTLTF));
    CUDA_CHECK(cudaFree(dNoiseVariance));
    CUDA_CHECK(cudaFree(dFineCFOHz));
    CUDA_CHECK(cudaFree(dFinalPacketOffset));
    CUDA_CHECK(cudaFree(dTimingCorrection));
    CUDA_CHECK(cudaFree(dMetrics));
    CUDA_CHECK(cudaFree(dCoarseCFOHz));
    CUDA_CHECK(cudaFree(dDetectedOffset));
    CUDA_CHECK(cudaFree(dTimingReference));
    CUDA_CHECK(cudaFree(dWaveform));
    CUDA_CHECK(cudaFree(dRawQ));
    CUDA_CHECK(cudaFree(dRawI));

    if (nlhs == 1)
        plhs[0] = makeStats(*slot);

    slot->occupied = false;
    slot->ticket = 0;
    slot->validationMode = 0;
}



void batchFrontendCommand(int nlhs,mxArray* plhs[],int nrhs,const mxArray* prhs[])
{
    if(nrhs!=5) mexErrMsgIdAndTxt("eht_receiver_cuda_e2e:InputCount","batchFrontend expects RxI,RxQ,RxScale,lltfReference.");
    if(nlhs>1) mexErrMsgIdAndTxt("eht_receiver_cuda_e2e:OutputCount","batchFrontend returns one struct.");
    if(!mxIsInt16(prhs[1])||mxIsComplex(prhs[1])||!mxIsInt16(prhs[2])||mxIsComplex(prhs[2])) mexErrMsgIdAndTxt("eht_receiver_cuda_e2e:Input","RxI/RxQ must be real int16 matrices.");
    mwSize count=mxGetM(prhs[1]),batch=mxGetN(prhs[1]); if(count<kPacketLengthSamples||mxGetM(prhs[2])!=count||mxGetN(prhs[2])!=batch||batch<1) mexErrMsgIdAndTxt("eht_receiver_cuda_e2e:Input","RxI/RxQ must be [samples x batch].");
    if(!mxIsSingle(prhs[4])||!mxIsComplex(prhs[4])||mxGetNumberOfElements(prhs[4])!=2560) mexErrMsgIdAndTxt("eht_receiver_cuda_e2e:Input","L-LTF reference must be complex single, 2560 samples.");
    mwSize scn=mxGetNumberOfElements(prhs[3]); if(scn!=1&&scn!=batch) mexErrMsgIdAndTxt("eht_receiver_cuda_e2e:Input","RxScale must be scalar or one value per packet.");
    std::vector<float> inv(batch); for(mwSize b=0;b<batch;++b){double v;if(mxIsDouble(prhs[3]))v=mxGetDoubles(prhs[3])[scn==1?0:b];else if(mxIsSingle(prhs[3]))v=mxGetSingles(prhs[3])[scn==1?0:b];else v=scalar(prhs[3],"RxScale");if(!(v>0))mexErrMsgIdAndTxt("eht_receiver_cuda_e2e:Input","RxScale must be positive.");inv[b]=float(1.0/v);}
    const mxComplexSingle* hr=mxGetComplexSingles(prhs[4]);std::vector<cufftComplex> ref(2560);double re=0;for(int i=0;i<2560;++i){ref[i].x=hr[i].real;ref[i].y=hr[i].imag;re+=double(ref[i].x)*ref[i].x+double(ref[i].y)*ref[i].y;}float refE=float(re);
    size_t total=size_t(count)*batch;int16_t *dI=nullptr,*dQ=nullptr;cufftComplex *dW=nullptr,*dRef=nullptr,*dL=nullptr,*dD=nullptr;float *dInv=nullptr,*dCC=nullptr,*dMet=nullptr,*dFC=nullptr,*dN=nullptr;int *dOff=nullptr,*dCorr=nullptr,*dFinal=nullptr;
    CUDA_CHECK(cudaMalloc((void**)&dI,total*sizeof(int16_t)));CUDA_CHECK(cudaMalloc((void**)&dQ,total*sizeof(int16_t)));CUDA_CHECK(cudaMalloc((void**)&dW,total*sizeof(cufftComplex)));CUDA_CHECK(cudaMalloc((void**)&dRef,2560*sizeof(cufftComplex)));CUDA_CHECK(cudaMalloc((void**)&dInv,batch*sizeof(float)));CUDA_CHECK(cudaMalloc((void**)&dOff,batch*sizeof(int)));CUDA_CHECK(cudaMalloc((void**)&dCC,batch*sizeof(float)));CUDA_CHECK(cudaMalloc((void**)&dMet,size_t(batch)*513*sizeof(float)));CUDA_CHECK(cudaMalloc((void**)&dCorr,batch*sizeof(int)));CUDA_CHECK(cudaMalloc((void**)&dFinal,batch*sizeof(int)));CUDA_CHECK(cudaMalloc((void**)&dFC,batch*sizeof(float)));CUDA_CHECK(cudaMalloc((void**)&dN,batch*sizeof(float)));CUDA_CHECK(cudaMalloc((void**)&dL,size_t(batch)*kEHTLTFLength*sizeof(cufftComplex)));CUDA_CHECK(cudaMalloc((void**)&dD,size_t(batch)*kEHTDataLength*sizeof(cufftComplex)));
    CUDA_CHECK(cudaMemcpy(dI,mxGetData(prhs[1]),total*sizeof(int16_t),cudaMemcpyHostToDevice));CUDA_CHECK(cudaMemcpy(dQ,mxGetData(prhs[2]),total*sizeof(int16_t),cudaMemcpyHostToDevice));CUDA_CHECK(cudaMemcpy(dRef,ref.data(),2560*sizeof(cufftComplex),cudaMemcpyHostToDevice));CUDA_CHECK(cudaMemcpy(dInv,inv.data(),batch*sizeof(float),cudaMemcpyHostToDevice));
    std::vector<int> init(batch,int(count));CUDA_CHECK(cudaMemcpy(dOff,init.data(),batch*sizeof(int),cudaMemcpyHostToDevice));
    int th=256;dim3 sampleGrid((int(count)+th-1)/th,(unsigned)batch);reconstructWaveformBatchKernel<<<sampleGrid,th>>>(dI,dQ,dW,dInv,int(count),int(batch));CUDA_CHECK(cudaGetLastError());dim3 detectGrid((int(count)-512+1+th-1)/th,(unsigned)batch);packetDetectBatchKernel<<<detectGrid,th>>>(dW,int(count),dOff,int(batch));CUDA_CHECK(cudaGetLastError());coarseCFOBatchKernel<<<int(batch),th>>>(dW,dOff,int(count),dCC,int(batch));CUDA_CHECK(cudaGetLastError());cfoCorrectBatchKernel<<<sampleGrid,th>>>(dW,dOff,int(count),dCC,int(batch));CUDA_CHECK(cudaGetLastError());dim3 tg((513+th-1)/th,(unsigned)batch);timingMetricBatchKernel<<<tg,th>>>(dW,dRef,dOff,int(count),refE,dMet,int(batch));CUDA_CHECK(cudaGetLastError());timingArgMaxBatchKernel<<<int(batch),1>>>(dMet,dOff,dCorr,dFinal,int(batch));CUDA_CHECK(cudaGetLastError());rebaseBatchKernel<<<sampleGrid,th>>>(dW,dFinal,dCorr,int(count),dCC,int(batch));CUDA_CHECK(cudaGetLastError());fineCFOBatchKernel<<<int(batch),th>>>(dW,dFinal,int(count),dFC,int(batch));CUDA_CHECK(cudaGetLastError());cfoCorrectBatchKernel<<<sampleGrid,th>>>(dW,dFinal,int(count),dFC,int(batch));CUDA_CHECK(cudaGetLastError());noiseBatchKernel<<<int(batch),th>>>(dW,dFinal,int(count),dN,int(batch));CUDA_CHECK(cudaGetLastError());dim3 eg((kEHTDataLength+th-1)/th,(unsigned)batch);extractFieldsBatchKernel<<<eg,th>>>(dW,dFinal,int(count),dL,dD,int(batch));CUDA_CHECK(cudaGetLastError());
    std::vector<int> off(batch),fin(batch);std::vector<float> cc(batch),fc(batch),nv(batch);CUDA_CHECK(cudaMemcpy(off.data(),dOff,batch*sizeof(int),cudaMemcpyDeviceToHost));CUDA_CHECK(cudaMemcpy(fin.data(),dFinal,batch*sizeof(int),cudaMemcpyDeviceToHost));CUDA_CHECK(cudaMemcpy(cc.data(),dCC,batch*sizeof(float),cudaMemcpyDeviceToHost));CUDA_CHECK(cudaMemcpy(fc.data(),dFC,batch*sizeof(float),cudaMemcpyDeviceToHost));CUDA_CHECK(cudaMemcpy(nv.data(),dN,batch*sizeof(float),cudaMemcpyDeviceToHost));
    const char* f[]={"BatchSize","CoarseOffsets","FinalOffsets","CoarseCFOHz","FineCFOHz","NoiseVariance","FieldsRemainOnGPU"};mxArray*out=mxCreateStructMatrix(1,1,7,f);auto mk=[&](auto &v){mxArray*a=mxCreateDoubleMatrix(batch,1,mxREAL);double*p=mxGetPr(a);for(mwSize i=0;i<batch;++i)p[i]=double(v[i]);return a;};mxSetField(out,0,"BatchSize",mxCreateDoubleScalar(double(batch)));mxSetField(out,0,"CoarseOffsets",mk(off));mxSetField(out,0,"FinalOffsets",mk(fin));mxSetField(out,0,"CoarseCFOHz",mk(cc));mxSetField(out,0,"FineCFOHz",mk(fc));mxSetField(out,0,"NoiseVariance",mk(nv));mxSetField(out,0,"FieldsRemainOnGPU",mxCreateLogicalScalar(true));
    cudaFree(dD);cudaFree(dL);cudaFree(dN);cudaFree(dFC);cudaFree(dFinal);cudaFree(dCorr);cudaFree(dMet);cudaFree(dCC);cudaFree(dOff);cudaFree(dInv);cudaFree(dRef);cudaFree(dW);cudaFree(dQ);cudaFree(dI);if(nlhs==1)plhs[0]=out;else mxDestroyArray(out);
}


void e2eBatchCommand(int nlhs,mxArray* plhs[],int nrhs,const mxArray* prhs[])
{
    if(nrhs!=15) mexErrMsgIdAndTxt("eht_receiver_cuda_e2e:InputCount","e2eBatch expects command plus 14 inputs.");
    if(nlhs>1) mexErrMsgIdAndTxt("eht_receiver_cuda_e2e:OutputCount","e2eBatch returns one batch validation structure.");
    if(!mxIsInt16(prhs[1])||mxIsComplex(prhs[1])||!mxIsInt16(prhs[2])||mxIsComplex(prhs[2])) mexErrMsgIdAndTxt("eht_receiver_cuda_e2e:Input","RxI/RxQ must be real int16 matrices.");
    mwSize count=mxGetM(prhs[1]),batch=mxGetN(prhs[1]);if(count<kPacketLengthSamples||batch<1||mxGetM(prhs[2])!=count||mxGetN(prhs[2])!=batch) mexErrMsgIdAndTxt("eht_receiver_cuda_e2e:Input","RxI/RxQ must be [samples x batch].");
    mwSize scn=mxGetNumberOfElements(prhs[3]);if(scn!=1&&scn!=batch) mexErrMsgIdAndTxt("eht_receiver_cuda_e2e:Input","RxScale must be scalar or one value per packet.");
    double llrScale=scalar(prhs[4],"llrScale");if(!(llrScale>0))mexErrMsgIdAndTxt("eht_receiver_cuda_e2e:Input","llrScale must be positive.");
    if(!mxIsSingle(prhs[5])||!mxIsComplex(prhs[5])||mxGetNumberOfElements(prhs[5])!=2560) mexErrMsgIdAndTxt("eht_receiver_cuda_e2e:Input","L-LTF reference must be complex single, 2560 samples.");
    int ltfStart=int(scalar(prhs[8],"ltfFFTStart"))-1;if(ltfStart<0||ltfStart+kFFTLength>kEHTLTFLength)mexErrMsgIdAndTxt("eht_receiver_cuda_e2e:Input","Invalid ltfFFTStart.");
    int d0=parseDataFFTStart0(prhs[9]);
    if(mxGetM(prhs[14])!=kPayloadBits||(mxGetN(prhs[14])!=1&&mxGetN(prhs[14])!=batch)||!(mxIsInt8(prhs[14])||mxIsLogical(prhs[14]))) mexErrMsgIdAndTxt("eht_receiver_cuda_e2e:Input","referencePayloadBits must be [%d x 1] or [%d x batch].",kPayloadBits,kPayloadBits);
    ensureInitialized();initializeFrontendMetadata(prhs[6],prhs[7],prhs[10],prhs[11],prhs[12],prhs[13]);
    std::vector<float> inv(batch);for(mwSize b=0;b<batch;++b){double v=mxIsDouble(prhs[3])?mxGetDoubles(prhs[3])[scn==1?0:b]:mxIsSingle(prhs[3])?double(mxGetSingles(prhs[3])[scn==1?0:b]):scalar(prhs[3],"RxScale");if(!(v>0))mexErrMsgIdAndTxt("eht_receiver_cuda_e2e:Input","RxScale must be positive.");inv[b]=float(1.0/v);}
    const mxComplexSingle* hr=mxGetComplexSingles(prhs[5]);std::vector<cufftComplex> tref(2560);double er=0;for(int i=0;i<2560;++i){tref[i]={hr[i].real,hr[i].imag};er+=double(hr[i].real)*hr[i].real+double(hr[i].imag)*hr[i].imag;}float refE=float(er);
    std::vector<int8_t> href(size_t(batch)*kPayloadBits);mwSize rc=mxGetN(prhs[14]);if(mxIsInt8(prhs[14])){const int8_t*p=(const int8_t*)mxGetData(prhs[14]);for(mwSize b=0;b<batch;++b)std::memcpy(href.data()+size_t(b)*kPayloadBits,p+size_t(rc==1?0:b)*kPayloadBits,kPayloadBits);}else{const mxLogical*p=mxGetLogicals(prhs[14]);for(mwSize b=0;b<batch;++b)for(int i=0;i<kPayloadBits;++i)href[size_t(b)*kPayloadBits+i]=p[size_t(rc==1?0:b)*kPayloadBits+i]?1:0;}
    size_t total=size_t(count)*batch;int16_t *ri=nullptr,*rq=nullptr,*qi=nullptr,*qq=nullptr;cufftComplex *w=nullptr,*tr=nullptr,*lf=nullptr,*df=nullptr,*li=nullptr,*lo=nullptr,*di=nullptr,*doo=nullptr,*la=nullptr,*da=nullptr,*ch=nullptr,*rot=nullptr,*eq=nullptr;float *is=nullptr,*cc=nullptr,*met=nullptr,*fc=nullptr,*nv=nullptr,*csi=nullptr,*invq=nullptr,*enc=nullptr,*bel=nullptr,*msg=nullptr;int *off=nullptr,*corr=nullptr,*fin=nullptr,*den=nullptr;int8_t *ellr=nullptr,*dec=nullptr,*seq=nullptr,*dref=nullptr;unsigned long long*cs=nullptr;unsigned int*be=nullptr;cufftHandle lp=0,dp=0;
    auto cm=[&](void**p,size_t n){CUDA_CHECK(cudaMalloc(p,n));};cm((void**)&ri,total*sizeof(int16_t));cm((void**)&rq,total*sizeof(int16_t));cm((void**)&w,total*sizeof(cufftComplex));cm((void**)&tr,2560*sizeof(cufftComplex));cm((void**)&is,batch*sizeof(float));cm((void**)&off,batch*sizeof(int));cm((void**)&cc,batch*sizeof(float));cm((void**)&met,size_t(batch)*513*sizeof(float));cm((void**)&corr,batch*sizeof(int));cm((void**)&fin,batch*sizeof(int));cm((void**)&fc,batch*sizeof(float));cm((void**)&nv,batch*sizeof(float));cm((void**)&lf,size_t(batch)*kEHTLTFLength*sizeof(cufftComplex));cm((void**)&df,size_t(batch)*kEHTDataLength*sizeof(cufftComplex));cm((void**)&li,size_t(batch)*kFFTLength*sizeof(cufftComplex));cm((void**)&lo,size_t(batch)*kFFTLength*sizeof(cufftComplex));cm((void**)&di,size_t(batch)*kDataInputCount*sizeof(cufftComplex));cm((void**)&doo,size_t(batch)*kDataInputCount*sizeof(cufftComplex));cm((void**)&la,size_t(batch)*kActiveToneCount*sizeof(cufftComplex));cm((void**)&da,size_t(batch)*kActiveToneCount*kNumDataSymbols*sizeof(cufftComplex));cm((void**)&ch,size_t(batch)*kActiveToneCount*sizeof(cufftComplex));cm((void**)&rot,size_t(batch)*kNumDataSymbols*sizeof(cufftComplex));cm((void**)&eq,size_t(batch)*kActiveToneCount*kNumDataSymbols*sizeof(cufftComplex));cm((void**)&qi,size_t(batch)*kTotalSymbols*sizeof(int16_t));cm((void**)&qq,size_t(batch)*kTotalSymbols*sizeof(int16_t));cm((void**)&csi,size_t(batch)*kDataToneCount*sizeof(float));cm((void**)&invq,size_t(batch)*kNumSegments*sizeof(float));cm((void**)&ellr,size_t(batch)*kTotalSymbols*kBitsPerSymbol*sizeof(int8_t));cm((void**)&enc,size_t(batch)*kEncodedLength*sizeof(float));cm((void**)&bel,size_t(batch)*kNumCodewords*N*sizeof(float));cm((void**)&msg,size_t(batch)*kNumCodewords*EDGES*sizeof(float));cm((void**)&dec,size_t(batch)*kTotalDecodedBits*sizeof(int8_t));cm((void**)&seq,size_t(batch)*2047*sizeof(int8_t));cm((void**)&den,batch*sizeof(int));cm((void**)&dref,size_t(batch)*kPayloadBits*sizeof(int8_t));cm((void**)&cs,batch*sizeof(unsigned long long));cm((void**)&be,batch*sizeof(unsigned int));
    CUDA_CHECK(cudaMemcpy(ri,mxGetData(prhs[1]),total*sizeof(int16_t),cudaMemcpyHostToDevice));CUDA_CHECK(cudaMemcpy(rq,mxGetData(prhs[2]),total*sizeof(int16_t),cudaMemcpyHostToDevice));CUDA_CHECK(cudaMemcpy(tr,tref.data(),2560*sizeof(cufftComplex),cudaMemcpyHostToDevice));CUDA_CHECK(cudaMemcpy(is,inv.data(),batch*sizeof(float),cudaMemcpyHostToDevice));CUDA_CHECK(cudaMemcpy(dref,href.data(),size_t(batch)*kPayloadBits*sizeof(int8_t),cudaMemcpyHostToDevice));std::vector<int> init(batch,int(count));CUDA_CHECK(cudaMemcpy(off,init.data(),batch*sizeof(int),cudaMemcpyHostToDevice));CUDA_CHECK(cudaMemset(cs,0,batch*sizeof(unsigned long long)));CUDA_CHECK(cudaMemset(be,0,batch*sizeof(unsigned int)));
   
   
   
   // STEP 7: fine-grained GPU front-end profiling using CUDA events.
    // These events are recorded in the same CUDA stream (default stream here),
    // so elapsed times measure actual device execution rather than host launch time.


    cudaEvent_t evPacketDetectStart=nullptr, evPacketDetectEnd=nullptr;
    cudaEvent_t evCoarseCFOEnd=nullptr, evTimingSyncEnd=nullptr, evFineCFOEnd=nullptr;
    CUDA_CHECK(cudaEventCreate(&evPacketDetectStart));
    CUDA_CHECK(cudaEventCreate(&evPacketDetectEnd));
    CUDA_CHECK(cudaEventCreate(&evCoarseCFOEnd));
    CUDA_CHECK(cudaEventCreate(&evTimingSyncEnd));
    CUDA_CHECK(cudaEventCreate(&evFineCFOEnd));

    int th=256;
    dim3 sg((int(count)+th-1)/th,(unsigned)batch);
    reconstructWaveformBatchKernel<<<sg,th>>>(ri,rq,w,is,int(count),int(batch));
    CUDA_CHECK(cudaGetLastError());

    // Stage 1: Packet Detection. INT16->FP32 reconstruction is intentionally
    // outside the locked 12 receiver stages.


    CUDA_CHECK(cudaEventRecord(evPacketDetectStart));
    dim3 dg((int(count)-512+1+th-1)/th,(unsigned)batch);
    packetDetectBatchKernel<<<dg,th>>>(w,int(count),off,int(batch));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(evPacketDetectEnd));

    // Stage 2: Coarse CFO = estimation + correction.
    coarseCFOBatchKernel<<<int(batch),th>>>(w,off,int(count),cc,int(batch));
    CUDA_CHECK(cudaGetLastError());
    cfoCorrectBatchKernel<<<sg,th>>>(w,off,int(count),cc,int(batch));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(evCoarseCFOEnd));

    // Stage 3: Timing Synchronization = timing metric/argmax + rebase to the
    // final packet boundary.
    dim3 tg((513+th-1)/th,(unsigned)batch);
    timingMetricBatchKernel<<<tg,th>>>(w,tr,off,int(count),refE,met,int(batch));
    CUDA_CHECK(cudaGetLastError());
    timingArgMaxBatchKernel<<<int(batch),1>>>(met,off,corr,fin,int(batch));
    CUDA_CHECK(cudaGetLastError());
    rebaseBatchKernel<<<sg,th>>>(w,fin,corr,int(count),cc,int(batch));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(evTimingSyncEnd));

    // Stage 4: Fine CFO = estimation + correction.
    fineCFOBatchKernel<<<int(batch),th>>>(w,fin,int(count),fc,int(batch));
    CUDA_CHECK(cudaGetLastError());
    cfoCorrectBatchKernel<<<sg,th>>>(w,fin,int(count),fc,int(batch));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(evFineCFOEnd));

    // Noise estimation and field extraction are currently outside the locked
    // four Step-7 timing categories.
    noiseBatchKernel<<<int(batch),th>>>(w,fin,int(count),nv,int(batch));
    CUDA_CHECK(cudaGetLastError());
    dim3 eg((kEHTDataLength+th-1)/th,(unsigned)batch);
    extractFieldsBatchKernel<<<eg,th>>>(w,fin,int(count),lf,df,int(batch));
    CUDA_CHECK(cudaGetLastError());
    dim3 gg((kFFTLength+th-1)/th,(unsigned)batch);gatherFFTWindowsBatchKernel<<<gg,th>>>(lf,df,li,di,ltfStart,d0,int(batch));CUDA_CHECK(cudaGetLastError());int n[1]={kFFTLength},emb[1]={kFFTLength};CUFFT_CHECK(cufftPlanMany(&lp,1,n,emb,1,kFFTLength,emb,1,kFFTLength,CUFFT_C2C,int(batch)));CUFFT_CHECK(cufftPlanMany(&dp,1,n,emb,1,kFFTLength,emb,1,kFFTLength,CUFFT_C2C,int(batch)*kNumDataSymbols));CUFFT_CHECK(cufftExecC2C(lp,li,lo,CUFFT_FORWARD));CUFFT_CHECK(cufftExecC2C(dp,di,doo,CUFFT_FORWARD));
    dim3 ag((kActiveToneCount+th-1)/th,(unsigned)batch);extractActiveBatchBackendKernel<<<ag,th>>>(lo,doo,sharedData.ltfFFTIndices,sharedData.dataFFTIndices,la,da,int(batch));estimateChannelBatchBackendKernel<<<ag,th>>>(la,sharedData.knownLTF,ch,int(batch));pilotRotationBatchBackendKernel<<<int(batch)*kNumDataSymbols,256,256*sizeof(cufftComplex)>>>(da,ch,sharedData.pilotIndices,sharedData.pilotReference,sharedData.pilotCount,rot,int(batch));dim3 eqg((kActiveToneCount*kNumDataSymbols+th-1)/th,(unsigned)batch);equalizeBatchBackendKernel<<<eqg,th>>>(da,ch,rot,nv,eq,int(batch));dim3 sqg(kNumSegments,(unsigned)batch);segmentQuantizeBatchBackendKernel<<<sqg,256>>>(eq,ch,sharedData.dataIndices,nv,qi,qq,csi,invq,int(batch));int bt=int(batch)*kTotalSymbols; qamBatchBackendKernel<<<(bt+th-1)/th,th>>>(qi,qq,ellr,invq,nv,float(llrScale),int(batch));int beN=int(batch)*kEncodedLength;mapWeightBatchBackendKernel<<<(beN+th-1)/th,th>>>(ellr,csi,sharedData.encodedSourceIndex,float(1.0/llrScale),enc,int(batch));dim3 rg(kNumCodewords,(N+th-1)/th,(unsigned)batch);reconstructBatchBackendKernel<<<rg,th>>>(enc,sharedData.payloadBits,sharedData.punctureBits,sharedData.repeatBits,bel,msg,int(batch));dim3 ldg(kNumCodewords,(unsigned)batch);layeredNMSBatchBackendKernel<<<ldg,128>>>(bel,msg,int(batch));dim3 hdg(kNumCodewords,(K+th-1)/th,(unsigned)batch);hardDecisionBatchBackendKernel<<<hdg,th>>>(bel,sharedData.payloadBits,dec,int(batch));deriveScramblerBatchBackendKernel<<<int(batch),1>>>(dec,seq,den,int(batch));dim3 vg((kPayloadBits+th-1)/th,(unsigned)batch);validateBatchBackendKernel<<<vg,th>>>(dec,seq,den,dref,cs,be,int(batch));CUDA_CHECK(cudaGetLastError());CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<unsigned long long> hcs(batch);std::vector<unsigned int> hbe(batch);CUDA_CHECK(cudaMemcpy(hcs.data(),cs,batch*sizeof(unsigned long long),cudaMemcpyDeviceToHost));CUDA_CHECK(cudaMemcpy(hbe.data(),be,batch*sizeof(unsigned int),cudaMemcpyDeviceToHost));
    const char* fs[]={"BatchSize","PacketsProcessed","TotalBitErrors","FailedPackets","Checksums","BitErrorsPerPacket","PacketErrorsPerPacket","ReturnedBulkOutputBytes","Pass"};mxArray*out=mxCreateStructMatrix(1,1,9,fs);mxArray*mc=mxCreateDoubleMatrix(batch,1,mxREAL),*mb=mxCreateDoubleMatrix(batch,1,mxREAL),*mp=mxCreateDoubleMatrix(batch,1,mxREAL);double*pc=mxGetPr(mc),*pb=mxGetPr(mb),*pp=mxGetPr(mp);double tbe=0,fail=0;for(mwSize b=0;b<batch;++b){pc[b]=double(hcs[b]);pb[b]=double(hbe[b]);pp[b]=hbe[b]?1.0:0.0;tbe+=hbe[b];fail+=pp[b];}mxSetField(out,0,"BatchSize",mxCreateDoubleScalar(double(batch)));mxSetField(out,0,"PacketsProcessed",mxCreateDoubleScalar(double(batch)));mxSetField(out,0,"TotalBitErrors",mxCreateDoubleScalar(tbe));mxSetField(out,0,"FailedPackets",mxCreateDoubleScalar(fail));mxSetField(out,0,"Checksums",mc);mxSetField(out,0,"BitErrorsPerPacket",mb);mxSetField(out,0,"PacketErrorsPerPacket",mp);mxSetField(out,0,"ReturnedBulkOutputBytes",mxCreateDoubleScalar(0));mxSetField(out,0,"Pass",mxCreateLogicalScalar(tbe==0&&fail==0));
    if(lp)cufftDestroy(lp);if(dp)cufftDestroy(dp);cudaFree(be);cudaFree(cs);cudaFree(dref);cudaFree(den);cudaFree(seq);cudaFree(dec);cudaFree(msg);cudaFree(bel);cudaFree(enc);cudaFree(ellr);cudaFree(invq);cudaFree(csi);cudaFree(qq);cudaFree(qi);cudaFree(eq);cudaFree(rot);cudaFree(ch);cudaFree(da);cudaFree(la);cudaFree(doo);cudaFree(di);cudaFree(lo);cudaFree(li);cudaFree(df);cudaFree(lf);cudaFree(nv);cudaFree(fc);cudaFree(fin);cudaFree(corr);cudaFree(met);cudaFree(cc);cudaFree(off);cudaFree(is);cudaFree(tr);cudaFree(w);cudaFree(rq);cudaFree(ri);if(nlhs==1)plhs[0]=out;else mxDestroyArray(out);
}



BatchedSlot* findFreeBatchedSlot()
{
    for (int i = 0; i < kNumSlots; ++i) {
        if (!batchedSlots[i].occupied)
            return &batchedSlots[i];
    }
    return nullptr;
}

BatchedSlot* findBatchedTicket(uint64_t ticket)
{
    for (int i = 0; i < kNumSlots; ++i) {
        if (batchedSlots[i].occupied &&
            batchedSlots[i].ticket == ticket)
            return &batchedSlots[i];
    }
    return nullptr;
}

int batchedSlotIndex(const BatchedSlot* slot)
{
    return int(slot - &batchedSlots[0]);
}

// Allocate all four complete E2E batch buffers before the timed benchmark.
// sampleCount and batchCapacity are supplied by MATLAB; only the number of
// buffers (kNumSlots = 4) is fixed.

void e2eBatchPrepareCommand(
    int nlhs,mxArray* plhs[],int nrhs,const mxArray* prhs[])
{
    if (nrhs != 3)
        mexErrMsgIdAndTxt(
            "eht_receiver_cuda_e2e:InputCount",
            "e2eBatchPrepare expects sampleCount and batchCapacity.");

    if (nlhs > 1)
        mexErrMsgIdAndTxt(
            "eht_receiver_cuda_e2e:OutputCount",
            "e2eBatchPrepare returns at most one information structure.");

    ensureInitialized();

    const double sampleCountRaw = scalar(prhs[1],"sampleCount");
    const double batchCapacityRaw = scalar(prhs[2],"batchCapacity");

    if (sampleCountRaw < 1.0 ||
        batchCapacityRaw < 1.0 ||
        floor(sampleCountRaw) != sampleCountRaw ||
        floor(batchCapacityRaw) != batchCapacityRaw)
        mexErrMsgIdAndTxt(
            "eht_receiver_cuda_e2e:Input",
            "sampleCount and batchCapacity must be positive integers.");

    const mwSize sampleCount = mwSize(sampleCountRaw);
    const mwSize batchCapacity = mwSize(batchCapacityRaw);

    for (int i = 0; i < kNumSlots; ++i) {
        BatchedSlot& s = batchedSlots[i];

        if (s.occupied)
            mexErrMsgIdAndTxt(
                "eht_receiver_cuda_e2e:Busy",
                "Cannot prepare batched buffers while work is in flight.");

        if (!s.stream ||
            s.sampleCount != sampleCount ||
            s.batchCapacity < batchCapacity) {
            allocateBatchedSlot(s,sampleCount,batchCapacity);
        }

        ensureBatchedFFTPlans(s,batchCapacity);
    }

    if (nlhs == 1) {
        const char* fields[] = {
            "BufferCount","SampleCount","BatchCapacity",
            "Asynchronous","UsesPinnedHostMemory"
        };
        mxArray* out = mxCreateStructMatrix(1,1,5,fields);
        mxSetField(out,0,"BufferCount",
            mxCreateDoubleScalar(double(kNumSlots)));
        mxSetField(out,0,"SampleCount",
            mxCreateDoubleScalar(double(sampleCount)));
        mxSetField(out,0,"BatchCapacity",
            mxCreateDoubleScalar(double(batchCapacity)));
        mxSetField(out,0,"Asynchronous",
            mxCreateLogicalScalar(true));
        mxSetField(out,0,"UsesPinnedHostMemory",
            mxCreateLogicalScalar(true));
        plhs[0] = out;
    }
}

// Submit one complete receiver batch to one of four persistent CUDA streams.
// This command returns after staging and enqueueing the work. It does not wait
// for the GPU batch to finish.

void e2eBatchSubmitCommand(
    int nlhs,mxArray* plhs[],int nrhs,const mxArray* prhs[])
{
    if (nrhs != 14)
        mexErrMsgIdAndTxt(
            "eht_receiver_cuda_e2e:InputCount",
            "e2eBatchSubmit expects command plus 13 inputs.");

    if (nlhs > 1)
        mexErrMsgIdAndTxt(
            "eht_receiver_cuda_e2e:OutputCount",
            "e2eBatchSubmit returns one ticket.");

    if (!mxIsInt16(prhs[1]) || mxIsComplex(prhs[1]) ||
        !mxIsInt16(prhs[2]) || mxIsComplex(prhs[2]))
        mexErrMsgIdAndTxt(
            "eht_receiver_cuda_e2e:Input",
            "RxI/RxQ must be real int16 matrices.");

    const mwSize count = mxGetM(prhs[1]);
    const mwSize batch = mxGetN(prhs[1]);

    if (count < kPacketLengthSamples ||
        batch < 1 ||
        mxGetM(prhs[2]) != count ||
        mxGetN(prhs[2]) != batch)
        mexErrMsgIdAndTxt(
            "eht_receiver_cuda_e2e:Input",
            "RxI/RxQ must be [samples x batch].");

    const mwSize scn = mxGetNumberOfElements(prhs[3]);
    if (scn != 1 && scn != batch)
        mexErrMsgIdAndTxt(
            "eht_receiver_cuda_e2e:Input",
            "RxScale must be scalar or one value per packet.");

    const double llrScale = scalar(prhs[4],"llrScale");
    if (!(llrScale > 0.0))
        mexErrMsgIdAndTxt(
            "eht_receiver_cuda_e2e:Input",
            "llrScale must be positive.");

    if (!mxIsSingle(prhs[5]) ||
        !mxIsComplex(prhs[5]) ||
        mxGetNumberOfElements(prhs[5]) != 2560)
        mexErrMsgIdAndTxt(
            "eht_receiver_cuda_e2e:Input",
            "L-LTF reference must be complex single, 2560 samples.");

    const int ltfStart =
        int(scalar(prhs[8],"ltfFFTStart")) - 1;
    if (ltfStart < 0 ||
        ltfStart + kFFTLength > kEHTLTFLength)
        mexErrMsgIdAndTxt(
            "eht_receiver_cuda_e2e:Input",
            "Invalid ltfFFTStart.");

    const int d0 = parseDataFFTStart0(prhs[9]);

    ensureInitialized();
    initializeFrontendMetadata(
        prhs[6],prhs[7],prhs[10],
        prhs[11],prhs[12],prhs[13]);

    BatchedSlot* slot = findFreeBatchedSlot();
    if (!slot)
        mexErrMsgIdAndTxt(
            "eht_receiver_cuda_e2e:NoFreeBatchedSlot",
            "All four asynchronous batched GPU buffers are occupied.");

    // Allocate/reallocate only while the slot is free. In the normal benchmark
    // e2eBatchPrepare performs this before timing, so no allocation occurs here.

    if (!slot->stream ||
        slot->sampleCount != count ||
        slot->batchCapacity < batch) {
        allocateBatchedSlot(*slot,count,batch);
    }

    ensureBatchedFFTPlans(*slot,batch);

    slot->activeBatchSize = batch;

    const size_t total = size_t(count) * size_t(batch);
    const size_t iqBytes = total * sizeof(int16_t);

    // Copy MATLAB-owned memory into persistent pinned host staging memory
    // before returning control to MATLAB.
    std::memcpy(slot->hRxI,mxGetData(prhs[1]),iqBytes);
    std::memcpy(slot->hRxQ,mxGetData(prhs[2]),iqBytes);

    for (mwSize b = 0; b < batch; ++b) {
        double v;
        if (mxIsDouble(prhs[3]))
            v = mxGetDoubles(prhs[3])[scn == 1 ? 0 : b];
        else if (mxIsSingle(prhs[3]))
            v = double(mxGetSingles(prhs[3])[scn == 1 ? 0 : b]);
        else
            v = scalar(prhs[3],"RxScale");

        if (!(v > 0.0))
            mexErrMsgIdAndTxt(
                "eht_receiver_cuda_e2e:Input",
                "RxScale must be positive.");

        slot->hInverseScales[b] = float(1.0/v);
    }

    const mxComplexSingle* hr = mxGetComplexSingles(prhs[5]);
    double referenceEnergy = 0.0;
    for (int i = 0; i < 2560; ++i) {
        slot->hLLTFReference[i] =
            cufftComplex{hr[i].real,hr[i].imag};
        referenceEnergy +=
            double(hr[i].real)*double(hr[i].real) +
            double(hr[i].imag)*double(hr[i].imag);
    }
    const float refE = float(referenceEnergy);

    slot->ticket = nextBatchedTicket++;
    slot->occupied = true;

    const int B = int(batch);
    const int C = int(count);
    const int th = 256;
    cudaStream_t stream = slot->stream;

    // Asynchronous H2D. RxI/RxQ/scales/reference are sourced from pinned
    // persistent host buffers owned by this slot.
    CUDA_CHECK(cudaMemcpyAsync(
        slot->dRxI,slot->hRxI,iqBytes,
        cudaMemcpyHostToDevice,stream));

    CUDA_CHECK(cudaMemcpyAsync(
        slot->dRxQ,slot->hRxQ,iqBytes,
        cudaMemcpyHostToDevice,stream));

    CUDA_CHECK(cudaMemcpyAsync(
        slot->dInverseScales,slot->hInverseScales,
        size_t(batch)*sizeof(float),
        cudaMemcpyHostToDevice,stream));

    CUDA_CHECK(cudaMemcpyAsync(
        slot->dLLTFReference,slot->hLLTFReference,
        size_t(2560)*sizeof(cufftComplex),
        cudaMemcpyHostToDevice,stream));

    // Initial packet-detection offset must be larger than every valid offset.
    CUDA_CHECK(cudaMemsetAsync(
        slot->dDetectedOffsets,0x7f,
        size_t(batch)*sizeof(int),stream));

    CUDA_CHECK(cudaMemsetAsync(
        slot->dChecksums,0,
        size_t(batch)*sizeof(unsigned long long),stream));

    dim3 sg((C+th-1)/th,(unsigned)B);

    // INT16 -> FP32 reconstruction remains outside the 12 stages.
    reconstructWaveformBatchKernel<<<sg,th,0,stream>>>(
        slot->dRxI,slot->dRxQ,
        slot->dWaveform,slot->dInverseScales,C,B);
    CUDA_CHECK(cudaGetLastError());

    // Stage 1: Packet Detection.
    CUDA_CHECK(cudaEventRecord(slot->evPacketDetectStart,stream));
    dim3 dg((C-512+1+th-1)/th,(unsigned)B);
    packetDetectBatchKernel<<<dg,th,0,stream>>>(
        slot->dWaveform,C,slot->dDetectedOffsets,B);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(slot->evPacketDetectEnd,stream));

    // Stage 2: Coarse CFO.
    coarseCFOBatchKernel<<<B,th,0,stream>>>(
        slot->dWaveform,slot->dDetectedOffsets,
        C,slot->dCoarseCFO,B);
    CUDA_CHECK(cudaGetLastError());

    cfoCorrectBatchKernel<<<sg,th,0,stream>>>(
        slot->dWaveform,slot->dDetectedOffsets,
        C,slot->dCoarseCFO,B);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(slot->evCoarseCFOEnd,stream));

    // Stage 3: Timing Synchronization.
    dim3 tg((513+th-1)/th,(unsigned)B);
    timingMetricBatchKernel<<<tg,th,0,stream>>>(
        slot->dWaveform,slot->dLLTFReference,
        slot->dDetectedOffsets,C,refE,
        slot->dTimingMetrics,B);
    CUDA_CHECK(cudaGetLastError());

    timingArgMaxBatchKernel<<<B,1,0,stream>>>(
        slot->dTimingMetrics,slot->dDetectedOffsets,
        slot->dTimingCorrection,slot->dFinalOffsets,B);
    CUDA_CHECK(cudaGetLastError());

    rebaseBatchKernel<<<sg,th,0,stream>>>(
        slot->dWaveform,slot->dFinalOffsets,
        slot->dTimingCorrection,C,slot->dCoarseCFO,B);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(slot->evTimingSyncEnd,stream));

    // Stage 4: Fine CFO.
    fineCFOBatchKernel<<<B,th,0,stream>>>(
        slot->dWaveform,slot->dFinalOffsets,
        C,slot->dFineCFO,B);
    CUDA_CHECK(cudaGetLastError());

    cfoCorrectBatchKernel<<<sg,th,0,stream>>>(
        slot->dWaveform,slot->dFinalOffsets,
        C,slot->dFineCFO,B);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(slot->evFineCFOEnd,stream));

    // Noise estimation remains outside the locked 12 stages.
    noiseBatchKernel<<<B,th,0,stream>>>(
        slot->dWaveform,slot->dFinalOffsets,
        C,slot->dNoiseVariance,B);
    CUDA_CHECK(cudaGetLastError());

    // Stage 5: OFDM Demodulation.
    CUDA_CHECK(cudaEventRecord(slot->evOFDMStart,stream));

    dim3 eg((kEHTDataLength+th-1)/th,(unsigned)B);
    extractFieldsBatchKernel<<<eg,th,0,stream>>>(
        slot->dWaveform,slot->dFinalOffsets,C,
        slot->dEHTLTF,slot->dEHTData,B);
    CUDA_CHECK(cudaGetLastError());

    dim3 gg((kFFTLength+th-1)/th,(unsigned)B);
    gatherFFTWindowsBatchKernel<<<gg,th,0,stream>>>(
        slot->dEHTLTF,slot->dEHTData,
        slot->dLTFInput,slot->dDataInput,
        ltfStart,d0,B);
    CUDA_CHECK(cudaGetLastError());

    CUFFT_CHECK(cufftExecC2C(
        slot->ltfPlan,
        slot->dLTFInput,
        slot->dLTFOutput,
        CUFFT_FORWARD));

    CUFFT_CHECK(cufftExecC2C(
        slot->dataPlan,
        slot->dDataInput,
        slot->dDataOutput,
        CUFFT_FORWARD));

    dim3 ag((kActiveToneCount+th-1)/th,(unsigned)B);
    extractActiveBatchBackendKernel<<<ag,th,0,stream>>>(
        slot->dLTFOutput,slot->dDataOutput,
        sharedData.ltfFFTIndices,sharedData.dataFFTIndices,
        slot->dLTFActive,slot->dDataActive,B);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(slot->evOFDMEnd,stream));

    // Stage 6: Channel Estimation.
    estimateChannelBatchBackendKernel<<<ag,th,0,stream>>>(
        slot->dLTFActive,sharedData.knownLTF,
        slot->dChannelEstimate,B);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(
        slot->evChannelEstimationEnd,stream));

    // Stage 7: Equalization.
    pilotRotationBatchBackendKernel
        <<<B*kNumDataSymbols,256,
           256*sizeof(cufftComplex),stream>>>(
            slot->dDataActive,
            slot->dChannelEstimate,
            sharedData.pilotIndices,
            sharedData.pilotReference,
            sharedData.pilotCount,
            slot->dPilotRotation,
            B);
    CUDA_CHECK(cudaGetLastError());

    dim3 eqg(
        (kActiveToneCount*kNumDataSymbols+th-1)/th,
        (unsigned)B);
    equalizeBatchBackendKernel<<<eqg,th,0,stream>>>(
        slot->dDataActive,
        slot->dChannelEstimate,
        slot->dPilotRotation,
        slot->dNoiseVariance,
        slot->dEqualized,
        B);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(slot->evEqualizationEnd,stream));

    // Stage 8: 4096-QAM Demapping.
    dim3 sqg(kNumSegments,(unsigned)B);
    segmentQuantizeBatchBackendKernel<<<sqg,256,0,stream>>>(
        slot->dEqualized,
        slot->dChannelEstimate,
        sharedData.dataIndices,
        slot->dNoiseVariance,
        slot->dQuantizedI,
        slot->dQuantizedQ,
        slot->dCSI,
        slot->dInverseQuantScale,
        B);
    CUDA_CHECK(cudaGetLastError());

    const int bt = B*kTotalSymbols;
    qamBatchBackendKernel<<<(bt+th-1)/th,th,0,stream>>>(
        slot->dQuantizedI,
        slot->dQuantizedQ,
        slot->dExternalLLR,
        slot->dInverseQuantScale,
        slot->dNoiseVariance,
        float(llrScale),
        B);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(slot->evQAMDemappingEnd,stream));

    // Stage 9: LLR Generation.
    const int beN = B*kEncodedLength;
    mapWeightBatchBackendKernel
        <<<(beN+th-1)/th,th,0,stream>>>(
            slot->dExternalLLR,
            slot->dCSI,
            sharedData.encodedSourceIndex,
            float(1.0/llrScale),
            slot->dEncoded,
            B);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(slot->evLLRGenerationEnd,stream));

    // Stage 10: LDPC Decoding.
    dim3 rg(kNumCodewords,(N+th-1)/th,(unsigned)B);
    reconstructBatchBackendKernel<<<rg,th,0,stream>>>(
        slot->dEncoded,
        sharedData.payloadBits,
        sharedData.punctureBits,
        sharedData.repeatBits,
        slot->dBeliefs,
        slot->dMessages,
        B);
    CUDA_CHECK(cudaGetLastError());

    dim3 ldg(kNumCodewords,(unsigned)B);
    layeredNMSBatchBackendKernel<<<ldg,128,0,stream>>>(
        slot->dBeliefs,slot->dMessages,B);
    CUDA_CHECK(cudaGetLastError());

    dim3 hdg(kNumCodewords,(K+th-1)/th,(unsigned)B);
    hardDecisionBatchBackendKernel<<<hdg,th,0,stream>>>(
        slot->dBeliefs,
        sharedData.payloadBits,
        slot->dDecoded,
        B);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(slot->evLDPCDecodingEnd,stream));

    // Stage 11: Descrambling.
    deriveScramblerBatchBackendKernel<<<B,1,0,stream>>>(
        slot->dDecoded,
        slot->dScrambleSequence,
        slot->dDescrambleEnabled,
        B);
    CUDA_CHECK(cudaGetLastError());

    const int descrN = B*kTotalDecodedBits;
    descrambleDecodedBatchBackendKernel
        <<<(descrN+th-1)/th,th,0,stream>>>(
            slot->dDecoded,
            slot->dScrambleSequence,
            slot->dDescrambleEnabled,
            slot->dDescrambled,
            B);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(slot->evDescramblingEnd,stream));

    // Stage 12: Payload Recovery.
    dim3 vg((kPayloadBits+th-1)/th,(unsigned)B);
    checksumRecoveredPayloadBatchBackendKernel<<<vg,th,0,stream>>>(
        slot->dDescrambled,
        slot->dChecksums,
        B);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(slot->evPayloadRecoveryEnd,stream));

    // Only the tiny checksum vector returns to host memory.
    CUDA_CHECK(cudaMemcpyAsync(
        slot->hChecksums,
        slot->dChecksums,
        size_t(batch)*sizeof(unsigned long long),
        cudaMemcpyDeviceToHost,
        stream));

    // Completion event comes after D2H, so collect() knows both compute and
    // the scalar checksum results are complete.
    CUDA_CHECK(cudaEventRecord(slot->done,stream));

    if (nlhs == 1)
        plhs[0] = mxCreateDoubleScalar(double(slot->ticket));
}

mxArray* makeBatchedStats(const BatchedSlot& s)
{
    float packetDetectionMs=0.0f;
    float coarseCFOMs=0.0f;
    float timingSyncMs=0.0f;
    float fineCFOMs=0.0f;
    float ofdmMs=0.0f;
    float channelEstimationMs=0.0f;
    float equalizationMs=0.0f;
    float qamDemappingMs=0.0f;
    float llrGenerationMs=0.0f;
    float ldpcDecodingMs=0.0f;
    float descramblingMs=0.0f;
    float payloadRecoveryMs=0.0f;

    CUDA_CHECK(cudaEventElapsedTime(
        &packetDetectionMs,
        s.evPacketDetectStart,s.evPacketDetectEnd));
    CUDA_CHECK(cudaEventElapsedTime(
        &coarseCFOMs,
        s.evPacketDetectEnd,s.evCoarseCFOEnd));
    CUDA_CHECK(cudaEventElapsedTime(
        &timingSyncMs,
        s.evCoarseCFOEnd,s.evTimingSyncEnd));
    CUDA_CHECK(cudaEventElapsedTime(
        &fineCFOMs,
        s.evTimingSyncEnd,s.evFineCFOEnd));
    CUDA_CHECK(cudaEventElapsedTime(
        &ofdmMs,
        s.evOFDMStart,s.evOFDMEnd));
    CUDA_CHECK(cudaEventElapsedTime(
        &channelEstimationMs,
        s.evOFDMEnd,s.evChannelEstimationEnd));
    CUDA_CHECK(cudaEventElapsedTime(
        &equalizationMs,
        s.evChannelEstimationEnd,s.evEqualizationEnd));
    CUDA_CHECK(cudaEventElapsedTime(
        &qamDemappingMs,
        s.evEqualizationEnd,s.evQAMDemappingEnd));
    CUDA_CHECK(cudaEventElapsedTime(
        &llrGenerationMs,
        s.evQAMDemappingEnd,s.evLLRGenerationEnd));
    CUDA_CHECK(cudaEventElapsedTime(
        &ldpcDecodingMs,
        s.evLLRGenerationEnd,s.evLDPCDecodingEnd));
    CUDA_CHECK(cudaEventElapsedTime(
        &descramblingMs,
        s.evLDPCDecodingEnd,s.evDescramblingEnd));
    CUDA_CHECK(cudaEventElapsedTime(
        &payloadRecoveryMs,
        s.evDescramblingEnd,s.evPayloadRecoveryEnd));

    const char* fields[] = {
        "BatchSize","PacketsProcessed","Checksums",
        "ReturnedBulkOutputBytes","Pass",
        "PacketDetectionTimeMs","CoarseCFOTimeMs",
        "TimingSynchronizationTimeMs","FineCFOTimeMs",
        "OFDMDemodulationTimeMs","ChannelEstimationTimeMs",
        "EqualizationTimeMs","QAMDemappingTimeMs",
        "LLRGenerationTimeMs","LDPCDecodingTimeMs",
        "DescramblingTimeMs","PayloadRecoveryTimeMs",
        "Ticket","BufferIndex","Asynchronous","BufferCount"
    };

    mxArray* out = mxCreateStructMatrix(1,1,21,fields);

    mxArray* mc =
        mxCreateDoubleMatrix(s.activeBatchSize,1,mxREAL);
    double* pc = mxGetPr(mc);

    for (mwSize b = 0; b < s.activeBatchSize; ++b)
        pc[b] = double(s.hChecksums[b]);

    mxSetField(out,0,"BatchSize",
        mxCreateDoubleScalar(double(s.activeBatchSize)));
    mxSetField(out,0,"PacketsProcessed",
        mxCreateDoubleScalar(double(s.activeBatchSize)));
    mxSetField(out,0,"Checksums",mc);
    mxSetField(out,0,"ReturnedBulkOutputBytes",
        mxCreateDoubleScalar(0));
    mxSetField(out,0,"Pass",
        mxCreateLogicalScalar(true));

    mxSetField(out,0,"PacketDetectionTimeMs",
        mxCreateDoubleScalar(double(packetDetectionMs)));
    mxSetField(out,0,"CoarseCFOTimeMs",
        mxCreateDoubleScalar(double(coarseCFOMs)));
    mxSetField(out,0,"TimingSynchronizationTimeMs",
        mxCreateDoubleScalar(double(timingSyncMs)));
    mxSetField(out,0,"FineCFOTimeMs",
        mxCreateDoubleScalar(double(fineCFOMs)));
    mxSetField(out,0,"OFDMDemodulationTimeMs",
        mxCreateDoubleScalar(double(ofdmMs)));
    mxSetField(out,0,"ChannelEstimationTimeMs",
        mxCreateDoubleScalar(double(channelEstimationMs)));
    mxSetField(out,0,"EqualizationTimeMs",
        mxCreateDoubleScalar(double(equalizationMs)));
    mxSetField(out,0,"QAMDemappingTimeMs",
        mxCreateDoubleScalar(double(qamDemappingMs)));
    mxSetField(out,0,"LLRGenerationTimeMs",
        mxCreateDoubleScalar(double(llrGenerationMs)));
    mxSetField(out,0,"LDPCDecodingTimeMs",
        mxCreateDoubleScalar(double(ldpcDecodingMs)));
    mxSetField(out,0,"DescramblingTimeMs",
        mxCreateDoubleScalar(double(descramblingMs)));
    mxSetField(out,0,"PayloadRecoveryTimeMs",
        mxCreateDoubleScalar(double(payloadRecoveryMs)));

    mxSetField(out,0,"Ticket",
        mxCreateDoubleScalar(double(s.ticket)));
    mxSetField(out,0,"BufferIndex",
        mxCreateDoubleScalar(double(
            batchedSlotIndex(&s)+1)));
    mxSetField(out,0,"Asynchronous",
        mxCreateLogicalScalar(true));
    mxSetField(out,0,"BufferCount",
        mxCreateDoubleScalar(double(kNumSlots)));

    return out;
}

void e2eBatchCollectCommand(
    int nlhs,mxArray* plhs[],int nrhs,const mxArray* prhs[])
{
    if (nrhs != 2)
        mexErrMsgIdAndTxt(
            "eht_receiver_cuda_e2e:InputCount",
            "e2eBatchCollect expects command and ticket.");

    if (nlhs > 1)
        mexErrMsgIdAndTxt(
            "eht_receiver_cuda_e2e:OutputCount",
            "e2eBatchCollect returns one result structure.");

    ensureInitialized();

    const uint64_t ticket =
        static_cast<uint64_t>(scalar(prhs[1],"ticket"));

    BatchedSlot* slot = findBatchedTicket(ticket);
    if (!slot)
        mexErrMsgIdAndTxt(
            "eht_receiver_cuda_e2e:UnknownBatchedTicket",
            "Ticket is not active or has already been collected.");

    // Wait only for this buffer. Other CUDA streams remain independent.
    CUDA_CHECK(cudaEventSynchronize(slot->done));

    if (nlhs == 1)
        plhs[0] = makeBatchedStats(*slot);

    slot->occupied = false;
    slot->ticket = 0;
    slot->activeBatchSize = 0;
}

void e2eBatchPollCommand(
    int nlhs,mxArray* plhs[],int nrhs,const mxArray* prhs[])
{
    if (nrhs != 2)
        mexErrMsgIdAndTxt(
            "eht_receiver_cuda_e2e:InputCount",
            "e2eBatchPoll expects command and ticket.");

    if (nlhs > 1)
        mexErrMsgIdAndTxt(
            "eht_receiver_cuda_e2e:OutputCount",
            "e2eBatchPoll returns one logical scalar.");

    ensureInitialized();

    const uint64_t ticket =
        static_cast<uint64_t>(scalar(prhs[1],"ticket"));

    BatchedSlot* slot = findBatchedTicket(ticket);
    if (!slot)
        mexErrMsgIdAndTxt(
            "eht_receiver_cuda_e2e:UnknownBatchedTicket",
            "Ticket is not active or has already been collected.");

    const cudaError_t status = cudaEventQuery(slot->done);
    if (status != cudaSuccess && status != cudaErrorNotReady)
        CUDA_CHECK(status);

    if (nlhs == 1)
        plhs[0] = mxCreateLogicalScalar(
            status == cudaSuccess);
}

void e2eBatchResetCommand()
{
    ensureInitialized();

    for (int i = 0; i < kNumSlots; ++i) {
        BatchedSlot& s = batchedSlots[i];
        if (s.occupied) {
            CUDA_CHECK(cudaEventSynchronize(s.done));
            s.occupied = false;
            s.ticket = 0;
            s.activeBatchSize = 0;
        }
    }
}

void e2eBatchRunCommand(int nlhs,mxArray* plhs[],int nrhs,const mxArray* prhs[])
{
    if(nrhs!=14) mexErrMsgIdAndTxt("eht_receiver_cuda_e2e:InputCount","e2eBatchRun expects command plus 13 inputs.");
    if(nlhs>1) mexErrMsgIdAndTxt("eht_receiver_cuda_e2e:OutputCount","e2eBatchRun returns one batch result structure.");
    if(!mxIsInt16(prhs[1])||mxIsComplex(prhs[1])||!mxIsInt16(prhs[2])||mxIsComplex(prhs[2])) mexErrMsgIdAndTxt("eht_receiver_cuda_e2e:Input","RxI/RxQ must be real int16 matrices.");
    mwSize count=mxGetM(prhs[1]),batch=mxGetN(prhs[1]);if(count<kPacketLengthSamples||batch<1||mxGetM(prhs[2])!=count||mxGetN(prhs[2])!=batch) mexErrMsgIdAndTxt("eht_receiver_cuda_e2e:Input","RxI/RxQ must be [samples x batch].");
    mwSize scn=mxGetNumberOfElements(prhs[3]);if(scn!=1&&scn!=batch) mexErrMsgIdAndTxt("eht_receiver_cuda_e2e:Input","RxScale must be scalar or one value per packet.");
    double llrScale=scalar(prhs[4],"llrScale");if(!(llrScale>0))mexErrMsgIdAndTxt("eht_receiver_cuda_e2e:Input","llrScale must be positive.");
    if(!mxIsSingle(prhs[5])||!mxIsComplex(prhs[5])||mxGetNumberOfElements(prhs[5])!=2560) mexErrMsgIdAndTxt("eht_receiver_cuda_e2e:Input","L-LTF reference must be complex single, 2560 samples.");
    int ltfStart=int(scalar(prhs[8],"ltfFFTStart"))-1;if(ltfStart<0||ltfStart+kFFTLength>kEHTLTFLength)mexErrMsgIdAndTxt("eht_receiver_cuda_e2e:Input","Invalid ltfFFTStart.");
    int d0=parseDataFFTStart0(prhs[9]);
    ensureInitialized();initializeFrontendMetadata(prhs[6],prhs[7],prhs[10],prhs[11],prhs[12],prhs[13]);
    std::vector<float> inv(batch);for(mwSize b=0;b<batch;++b){double v=mxIsDouble(prhs[3])?mxGetDoubles(prhs[3])[scn==1?0:b]:mxIsSingle(prhs[3])?double(mxGetSingles(prhs[3])[scn==1?0:b]):scalar(prhs[3],"RxScale");if(!(v>0))mexErrMsgIdAndTxt("eht_receiver_cuda_e2e:Input","RxScale must be positive.");inv[b]=float(1.0/v);}
    const mxComplexSingle* hr=mxGetComplexSingles(prhs[5]);std::vector<cufftComplex> tref(2560);double er=0;for(int i=0;i<2560;++i){tref[i]={hr[i].real,hr[i].imag};er+=double(hr[i].real)*hr[i].real+double(hr[i].imag)*hr[i].imag;}float refE=float(er);
    size_t total=size_t(count)*batch;int16_t *ri=nullptr,*rq=nullptr,*qi=nullptr,*qq=nullptr;cufftComplex *w=nullptr,*tr=nullptr,*lf=nullptr,*df=nullptr,*li=nullptr,*lo=nullptr,*di=nullptr,*doo=nullptr,*la=nullptr,*da=nullptr,*ch=nullptr,*rot=nullptr,*eq=nullptr;float *is=nullptr,*cc=nullptr,*met=nullptr,*fc=nullptr,*nv=nullptr,*csi=nullptr,*invq=nullptr,*enc=nullptr,*bel=nullptr,*msg=nullptr;int *off=nullptr,*corr=nullptr,*fin=nullptr,*den=nullptr;int8_t *ellr=nullptr,*dec=nullptr,*seq=nullptr,*descr=nullptr;unsigned long long*cs=nullptr;cufftHandle lp=0,dp=0;
    auto cm=[&](void**p,size_t n){CUDA_CHECK(cudaMalloc(p,n));};cm((void**)&ri,total*sizeof(int16_t));cm((void**)&rq,total*sizeof(int16_t));cm((void**)&w,total*sizeof(cufftComplex));cm((void**)&tr,2560*sizeof(cufftComplex));cm((void**)&is,batch*sizeof(float));cm((void**)&off,batch*sizeof(int));cm((void**)&cc,batch*sizeof(float));cm((void**)&met,size_t(batch)*513*sizeof(float));cm((void**)&corr,batch*sizeof(int));cm((void**)&fin,batch*sizeof(int));cm((void**)&fc,batch*sizeof(float));cm((void**)&nv,batch*sizeof(float));cm((void**)&lf,size_t(batch)*kEHTLTFLength*sizeof(cufftComplex));cm((void**)&df,size_t(batch)*kEHTDataLength*sizeof(cufftComplex));cm((void**)&li,size_t(batch)*kFFTLength*sizeof(cufftComplex));cm((void**)&lo,size_t(batch)*kFFTLength*sizeof(cufftComplex));cm((void**)&di,size_t(batch)*kDataInputCount*sizeof(cufftComplex));cm((void**)&doo,size_t(batch)*kDataInputCount*sizeof(cufftComplex));cm((void**)&la,size_t(batch)*kActiveToneCount*sizeof(cufftComplex));cm((void**)&da,size_t(batch)*kActiveToneCount*kNumDataSymbols*sizeof(cufftComplex));cm((void**)&ch,size_t(batch)*kActiveToneCount*sizeof(cufftComplex));cm((void**)&rot,size_t(batch)*kNumDataSymbols*sizeof(cufftComplex));cm((void**)&eq,size_t(batch)*kActiveToneCount*kNumDataSymbols*sizeof(cufftComplex));cm((void**)&qi,size_t(batch)*kTotalSymbols*sizeof(int16_t));cm((void**)&qq,size_t(batch)*kTotalSymbols*sizeof(int16_t));cm((void**)&csi,size_t(batch)*kDataToneCount*sizeof(float));cm((void**)&invq,size_t(batch)*kNumSegments*sizeof(float));cm((void**)&ellr,size_t(batch)*kTotalSymbols*kBitsPerSymbol*sizeof(int8_t));cm((void**)&enc,size_t(batch)*kEncodedLength*sizeof(float));cm((void**)&bel,size_t(batch)*kNumCodewords*N*sizeof(float));cm((void**)&msg,size_t(batch)*kNumCodewords*EDGES*sizeof(float));cm((void**)&dec,size_t(batch)*kTotalDecodedBits*sizeof(int8_t));cm((void**)&seq,size_t(batch)*2047*sizeof(int8_t));cm((void**)&descr,size_t(batch)*kTotalDecodedBits*sizeof(int8_t));cm((void**)&den,batch*sizeof(int));cm((void**)&cs,batch*sizeof(unsigned long long));
    CUDA_CHECK(cudaMemcpy(ri,mxGetData(prhs[1]),total*sizeof(int16_t),cudaMemcpyHostToDevice));CUDA_CHECK(cudaMemcpy(rq,mxGetData(prhs[2]),total*sizeof(int16_t),cudaMemcpyHostToDevice));CUDA_CHECK(cudaMemcpy(tr,tref.data(),2560*sizeof(cufftComplex),cudaMemcpyHostToDevice));CUDA_CHECK(cudaMemcpy(is,inv.data(),batch*sizeof(float),cudaMemcpyHostToDevice));std::vector<int> init(batch,int(count));CUDA_CHECK(cudaMemcpy(off,init.data(),batch*sizeof(int),cudaMemcpyHostToDevice));CUDA_CHECK(cudaMemset(cs,0,batch*sizeof(unsigned long long)));
    
    
    
    // STEP 7: fine-grained GPU front-end profiling for the e2eBatchRun path.
    // The benchmark uses e2eBatchRun, so the CUDA events must be declared and
    
    // recorded in this function's scope (not only in e2eBatchCommand).
    
    cudaEvent_t evPacketDetectStart=nullptr, evPacketDetectEnd=nullptr;
    cudaEvent_t evCoarseCFOEnd=nullptr, evTimingSyncEnd=nullptr, evFineCFOEnd=nullptr;
    cudaEvent_t evOFDMStart=nullptr, evOFDMEnd=nullptr;
    cudaEvent_t evChannelEstimationEnd=nullptr, evEqualizationEnd=nullptr;
    cudaEvent_t evQAMDemappingEnd=nullptr, evLLRGenerationEnd=nullptr;
    cudaEvent_t evLDPCDecodingEnd=nullptr, evDescramblingEnd=nullptr, evPayloadRecoveryEnd=nullptr;
    CUDA_CHECK(cudaEventCreate(&evPacketDetectStart));
    CUDA_CHECK(cudaEventCreate(&evPacketDetectEnd));
    CUDA_CHECK(cudaEventCreate(&evCoarseCFOEnd));
    CUDA_CHECK(cudaEventCreate(&evTimingSyncEnd));
    CUDA_CHECK(cudaEventCreate(&evFineCFOEnd));
    CUDA_CHECK(cudaEventCreate(&evOFDMStart));
    CUDA_CHECK(cudaEventCreate(&evOFDMEnd));
    CUDA_CHECK(cudaEventCreate(&evChannelEstimationEnd));
    CUDA_CHECK(cudaEventCreate(&evEqualizationEnd));
    CUDA_CHECK(cudaEventCreate(&evQAMDemappingEnd));
    CUDA_CHECK(cudaEventCreate(&evLLRGenerationEnd));
    CUDA_CHECK(cudaEventCreate(&evLDPCDecodingEnd));
    CUDA_CHECK(cudaEventCreate(&evDescramblingEnd));
    CUDA_CHECK(cudaEventCreate(&evPayloadRecoveryEnd));

    int th=256;
    dim3 sg((int(count)+th-1)/th,(unsigned)batch);

    // INT16 -> FP32 waveform reconstruction is intentionally outside the
    // locked 12 receiver-stage timings.
    reconstructWaveformBatchKernel<<<sg,th>>>(ri,rq,w,is,int(count),int(batch));
    CUDA_CHECK(cudaGetLastError());

    // Stage 1: Packet Detection.
    CUDA_CHECK(cudaEventRecord(evPacketDetectStart));
    dim3 dg((int(count)-512+1+th-1)/th,(unsigned)batch);
    packetDetectBatchKernel<<<dg,th>>>(w,int(count),off,int(batch));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(evPacketDetectEnd));

    // Stage 2: Coarse CFO = estimation + correction.
    coarseCFOBatchKernel<<<int(batch),th>>>(w,off,int(count),cc,int(batch));
    CUDA_CHECK(cudaGetLastError());
    cfoCorrectBatchKernel<<<sg,th>>>(w,off,int(count),cc,int(batch));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(evCoarseCFOEnd));

    // Stage 3: Timing Synchronization.
    dim3 tg((513+th-1)/th,(unsigned)batch);
    timingMetricBatchKernel<<<tg,th>>>(w,tr,off,int(count),refE,met,int(batch));
    CUDA_CHECK(cudaGetLastError());
    timingArgMaxBatchKernel<<<int(batch),1>>>(met,off,corr,fin,int(batch));
    CUDA_CHECK(cudaGetLastError());
    rebaseBatchKernel<<<sg,th>>>(w,fin,corr,int(count),cc,int(batch));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(evTimingSyncEnd));

    // Stage 4: Fine CFO = estimation + correction.
    fineCFOBatchKernel<<<int(batch),th>>>(w,fin,int(count),fc,int(batch));
    CUDA_CHECK(cudaGetLastError());
    cfoCorrectBatchKernel<<<sg,th>>>(w,fin,int(count),fc,int(batch));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(evFineCFOEnd));

    // Noise estimation is outside the locked 12 stage timings.
    noiseBatchKernel<<<int(batch),th>>>(w,fin,int(count),nv,int(batch));
    CUDA_CHECK(cudaGetLastError());

    // Stage 5: OFDM Demodulation = EHT field extraction + FFT window gather
    // + cuFFT execution + active-tone extraction.
    CUDA_CHECK(cudaEventRecord(evOFDMStart));
    dim3 eg((kEHTDataLength+th-1)/th,(unsigned)batch);
    extractFieldsBatchKernel<<<eg,th>>>(w,fin,int(count),lf,df,int(batch));
    CUDA_CHECK(cudaGetLastError());
    dim3 gg((kFFTLength+th-1)/th,(unsigned)batch);
    gatherFFTWindowsBatchKernel<<<gg,th>>>(lf,df,li,di,ltfStart,d0,int(batch));
    CUDA_CHECK(cudaGetLastError());
    int n[1]={kFFTLength},emb[1]={kFFTLength};
    CUFFT_CHECK(cufftPlanMany(&lp,1,n,emb,1,kFFTLength,emb,1,kFFTLength,CUFFT_C2C,int(batch)));
    CUFFT_CHECK(cufftPlanMany(&dp,1,n,emb,1,kFFTLength,emb,1,kFFTLength,CUFFT_C2C,int(batch)*kNumDataSymbols));
    CUFFT_CHECK(cufftExecC2C(lp,li,lo,CUFFT_FORWARD));
    CUFFT_CHECK(cufftExecC2C(dp,di,doo,CUFFT_FORWARD));
    dim3 ag((kActiveToneCount+th-1)/th,(unsigned)batch);
    extractActiveBatchBackendKernel<<<ag,th>>>(lo,doo,sharedData.ltfFFTIndices,sharedData.dataFFTIndices,la,da,int(batch));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(evOFDMEnd));

    // Stage 6: Channel Estimation.
    estimateChannelBatchBackendKernel<<<ag,th>>>(la,sharedData.knownLTF,ch,int(batch));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(evChannelEstimationEnd));

    // Stage 7: Equalization = pilot phase tracking + equalization.
    pilotRotationBatchBackendKernel<<<int(batch)*kNumDataSymbols,256,256*sizeof(cufftComplex)>>>(da,ch,sharedData.pilotIndices,sharedData.pilotReference,sharedData.pilotCount,rot,int(batch));
    CUDA_CHECK(cudaGetLastError());
    dim3 eqg((kActiveToneCount*kNumDataSymbols+th-1)/th,(unsigned)batch);
    equalizeBatchBackendKernel<<<eqg,th>>>(da,ch,rot,nv,eq,int(batch));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(evEqualizationEnd));

    // Stage 8: 4096-QAM Demapping. The segment-quantization kernel is the
    // GPU-side preparation of equalized constellation samples consumed by
    // the soft 4096-QAM demapper, so it is included in this stage.
    dim3 sqg(kNumSegments,(unsigned)batch);
    segmentQuantizeBatchBackendKernel<<<sqg,256>>>(eq,ch,sharedData.dataIndices,nv,qi,qq,csi,invq,int(batch));
    CUDA_CHECK(cudaGetLastError());
    int bt=int(batch)*kTotalSymbols;
    qamBatchBackendKernel<<<(bt+th-1)/th,th>>>(qi,qq,ellr,invq,nv,float(llrScale),int(batch));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(evQAMDemappingEnd));

    // Stage 9: LLR Generation = LLR rescaling/CSI weighting and mapping to
    // the encoded LDPC input ordering.
    int beN=int(batch)*kEncodedLength;
    mapWeightBatchBackendKernel<<<(beN+th-1)/th,th>>>(ellr,csi,sharedData.encodedSourceIndex,float(1.0/llrScale),enc,int(batch));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(evLLRGenerationEnd));

    // Stage 10: LDPC Decoding = codeword reconstruction, layered normalized
    // min-sum iterations, and hard decision.
    dim3 rg(kNumCodewords,(N+th-1)/th,(unsigned)batch);
    reconstructBatchBackendKernel<<<rg,th>>>(enc,sharedData.payloadBits,sharedData.punctureBits,sharedData.repeatBits,bel,msg,int(batch));
    CUDA_CHECK(cudaGetLastError());
    dim3 ldg(kNumCodewords,(unsigned)batch);
    layeredNMSBatchBackendKernel<<<ldg,128>>>(bel,msg,int(batch));
    CUDA_CHECK(cudaGetLastError());
    dim3 hdg(kNumCodewords,(K+th-1)/th,(unsigned)batch);
    hardDecisionBatchBackendKernel<<<hdg,th>>>(bel,sharedData.payloadBits,dec,int(batch));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(evLDPCDecodingEnd));

    // Stage 11: Descrambling = derive the scrambler sequence and apply it to
    // the complete decoded bitstream.
    deriveScramblerBatchBackendKernel<<<int(batch),1>>>(dec,seq,den,int(batch));
    CUDA_CHECK(cudaGetLastError());
    int descrN=int(batch)*kTotalDecodedBits;
    descrambleDecodedBatchBackendKernel<<<(descrN+th-1)/th,th>>>(dec,seq,den,descr,int(batch));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(evDescramblingEnd));

    // Stage 12: Payload Recovery = extract the payload region and calculate
    // the existing scalar checksum used by the benchmark.
    dim3 vg((kPayloadBits+th-1)/th,(unsigned)batch);
    checksumRecoveredPayloadBatchBackendKernel<<<vg,th>>>(descr,cs,int(batch));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(evPayloadRecoveryEnd));
    CUDA_CHECK(cudaDeviceSynchronize());
    // STEP 7: read the four front-end stage timings after device completion.
    float packetDetectionMs=0.0f, coarseCFOMs=0.0f, timingSyncMs=0.0f, fineCFOMs=0.0f;
    float ofdmMs=0.0f, channelEstimationMs=0.0f, equalizationMs=0.0f;
    float qamDemappingMs=0.0f, llrGenerationMs=0.0f, ldpcDecodingMs=0.0f;
    float descramblingMs=0.0f, payloadRecoveryMs=0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&packetDetectionMs,evPacketDetectStart,evPacketDetectEnd));
    CUDA_CHECK(cudaEventElapsedTime(&coarseCFOMs,evPacketDetectEnd,evCoarseCFOEnd));
    CUDA_CHECK(cudaEventElapsedTime(&timingSyncMs,evCoarseCFOEnd,evTimingSyncEnd));
    CUDA_CHECK(cudaEventElapsedTime(&fineCFOMs,evTimingSyncEnd,evFineCFOEnd));
    CUDA_CHECK(cudaEventElapsedTime(&ofdmMs,evOFDMStart,evOFDMEnd));
    CUDA_CHECK(cudaEventElapsedTime(&channelEstimationMs,evOFDMEnd,evChannelEstimationEnd));
    CUDA_CHECK(cudaEventElapsedTime(&equalizationMs,evChannelEstimationEnd,evEqualizationEnd));
    CUDA_CHECK(cudaEventElapsedTime(&qamDemappingMs,evEqualizationEnd,evQAMDemappingEnd));
    CUDA_CHECK(cudaEventElapsedTime(&llrGenerationMs,evQAMDemappingEnd,evLLRGenerationEnd));
    CUDA_CHECK(cudaEventElapsedTime(&ldpcDecodingMs,evLLRGenerationEnd,evLDPCDecodingEnd));
    CUDA_CHECK(cudaEventElapsedTime(&descramblingMs,evLDPCDecodingEnd,evDescramblingEnd));
    CUDA_CHECK(cudaEventElapsedTime(&payloadRecoveryMs,evDescramblingEnd,evPayloadRecoveryEnd));

    std::vector<unsigned long long> hcs(batch);CUDA_CHECK(cudaMemcpy(hcs.data(),cs,batch*sizeof(unsigned long long),cudaMemcpyDeviceToHost));
    const char* fs[]={"BatchSize","PacketsProcessed","Checksums","ReturnedBulkOutputBytes","Pass","PacketDetectionTimeMs","CoarseCFOTimeMs","TimingSynchronizationTimeMs","FineCFOTimeMs","OFDMDemodulationTimeMs","ChannelEstimationTimeMs","EqualizationTimeMs","QAMDemappingTimeMs","LLRGenerationTimeMs","LDPCDecodingTimeMs","DescramblingTimeMs","PayloadRecoveryTimeMs"};
    mxArray*out=mxCreateStructMatrix(1,1,17,fs);mxArray*mc=mxCreateDoubleMatrix(batch,1,mxREAL);double*pc=mxGetPr(mc);for(mwSize b=0;b<batch;++b)pc[b]=double(hcs[b]);mxSetField(out,0,"BatchSize",mxCreateDoubleScalar(double(batch)));mxSetField(out,0,"PacketsProcessed",mxCreateDoubleScalar(double(batch)));mxSetField(out,0,"Checksums",mc);mxSetField(out,0,"ReturnedBulkOutputBytes",mxCreateDoubleScalar(0));mxSetField(out,0,"Pass",mxCreateLogicalScalar(true));
    mxSetField(out,0,"PacketDetectionTimeMs",mxCreateDoubleScalar(double(packetDetectionMs)));
    mxSetField(out,0,"CoarseCFOTimeMs",mxCreateDoubleScalar(double(coarseCFOMs)));
    mxSetField(out,0,"TimingSynchronizationTimeMs",mxCreateDoubleScalar(double(timingSyncMs)));
    mxSetField(out,0,"FineCFOTimeMs",mxCreateDoubleScalar(double(fineCFOMs)));
    mxSetField(out,0,"OFDMDemodulationTimeMs",mxCreateDoubleScalar(double(ofdmMs)));
    mxSetField(out,0,"ChannelEstimationTimeMs",mxCreateDoubleScalar(double(channelEstimationMs)));
    mxSetField(out,0,"EqualizationTimeMs",mxCreateDoubleScalar(double(equalizationMs)));
    mxSetField(out,0,"QAMDemappingTimeMs",mxCreateDoubleScalar(double(qamDemappingMs)));
    mxSetField(out,0,"LLRGenerationTimeMs",mxCreateDoubleScalar(double(llrGenerationMs)));
    mxSetField(out,0,"LDPCDecodingTimeMs",mxCreateDoubleScalar(double(ldpcDecodingMs)));
    mxSetField(out,0,"DescramblingTimeMs",mxCreateDoubleScalar(double(descramblingMs)));
    mxSetField(out,0,"PayloadRecoveryTimeMs",mxCreateDoubleScalar(double(payloadRecoveryMs)));

    CUDA_CHECK(cudaEventDestroy(evPayloadRecoveryEnd));
    CUDA_CHECK(cudaEventDestroy(evDescramblingEnd));
    CUDA_CHECK(cudaEventDestroy(evLDPCDecodingEnd));
    CUDA_CHECK(cudaEventDestroy(evLLRGenerationEnd));
    CUDA_CHECK(cudaEventDestroy(evQAMDemappingEnd));
    CUDA_CHECK(cudaEventDestroy(evEqualizationEnd));
    CUDA_CHECK(cudaEventDestroy(evChannelEstimationEnd));
    CUDA_CHECK(cudaEventDestroy(evOFDMEnd));
    CUDA_CHECK(cudaEventDestroy(evOFDMStart));
    CUDA_CHECK(cudaEventDestroy(evFineCFOEnd));CUDA_CHECK(cudaEventDestroy(evTimingSyncEnd));CUDA_CHECK(cudaEventDestroy(evCoarseCFOEnd));CUDA_CHECK(cudaEventDestroy(evPacketDetectEnd));CUDA_CHECK(cudaEventDestroy(evPacketDetectStart));
    if(lp)cufftDestroy(lp);if(dp)cufftDestroy(dp);cudaFree(cs);cudaFree(den);cudaFree(descr);cudaFree(seq);cudaFree(dec);cudaFree(msg);cudaFree(bel);cudaFree(enc);cudaFree(ellr);cudaFree(invq);cudaFree(csi);cudaFree(qq);cudaFree(qi);cudaFree(eq);cudaFree(rot);cudaFree(ch);cudaFree(da);cudaFree(la);cudaFree(doo);cudaFree(di);cudaFree(lo);cudaFree(li);cudaFree(df);cudaFree(lf);cudaFree(nv);cudaFree(fc);cudaFree(fin);cudaFree(corr);cudaFree(met);cudaFree(cc);cudaFree(off);cudaFree(is);cudaFree(tr);cudaFree(w);cudaFree(rq);cudaFree(ri);if(nlhs==1)plhs[0]=out;else mxDestroyArray(out);
}

void e2eBatchLegacyCommand(
    int nlhs,mxArray* plhs[],int nrhs,const mxArray* prhs[])
{
    // Legacy correctness implementation retained for reference.
    // command + 14 inputs, matching e2e, except RxI/RxQ are
    // [samplesPerPacket x batchSize] and RxScale may be scalar or batchSize.
    if (nrhs != 15)
        mexErrMsgIdAndTxt(
            "eht_receiver_cuda_e2e:InputCount",
            "e2eBatch expects command plus 14 inputs.");
    if (nlhs > 1)
        mexErrMsgIdAndTxt(
            "eht_receiver_cuda_e2e:OutputCount",
            "e2eBatch returns one batch validation structure.");

    if (!mxIsInt16(prhs[1]) || mxIsComplex(prhs[1]) ||
        !mxIsInt16(prhs[2]) || mxIsComplex(prhs[2]))
        mexErrMsgIdAndTxt(
            "eht_receiver_cuda_e2e:Input",
            "RxI and RxQ must be real int16 matrices.");

    const mwSize samplesPerPacket = mxGetM(prhs[1]);
    const mwSize batchSize = mxGetN(prhs[1]);
    if (samplesPerPacket < kPacketLengthSamples || batchSize < 1 ||
        mxGetM(prhs[2]) != samplesPerPacket ||
        mxGetN(prhs[2]) != batchSize)
        mexErrMsgIdAndTxt(
            "eht_receiver_cuda_e2e:Input",
            "RxI and RxQ must be [samplesPerPacket x batchSize] with at least %d samples per packet.", kPacketLengthSamples);

    const mwSize scaleCount = mxGetNumberOfElements(prhs[3]);
    if (scaleCount != 1 && scaleCount != batchSize)
        mexErrMsgIdAndTxt(
            "eht_receiver_cuda_e2e:Input",
            "RxScale must be scalar or contain one value per packet in the batch.");

    // Reference bits may be one common [kPayloadBits x 1] vector or
    // [kPayloadBits x batchSize] for packet-specific validation.
    if (mxIsComplex(prhs[14]) ||
        !(mxIsInt8(prhs[14]) || mxIsLogical(prhs[14])))
        mexErrMsgIdAndTxt(
            "eht_receiver_cuda_e2e:Input",
            "referencePayloadBits must be int8/logical.");
    const mwSize refRows = mxGetM(prhs[14]);
    const mwSize refCols = mxGetN(prhs[14]);
    if (refRows != kPayloadBits || (refCols != 1 && refCols != batchSize))
        mexErrMsgIdAndTxt(
            "eht_receiver_cuda_e2e:Input",
            "referencePayloadBits must be [%d x 1] or [%d x batchSize].",kPayloadBits,kPayloadBits);

    const char* fields[] = {
        "BatchSize","PacketsProcessed","TotalBitErrors","FailedPackets",
        "Checksums","BitErrorsPerPacket","PacketErrorsPerPacket",
        "ReturnedBulkOutputBytes","Pass"
    };
    mxArray* out = mxCreateStructMatrix(1,1,9,fields);
    mxArray* checksums = mxCreateDoubleMatrix(batchSize,1,mxREAL);
    mxArray* bitErrors = mxCreateDoubleMatrix(batchSize,1,mxREAL);
    mxArray* packetErrors = mxCreateDoubleMatrix(batchSize,1,mxREAL);
    double* checksumData = mxGetPr(checksums);
    double* bitErrorData = mxGetPr(bitErrors);
    double* packetErrorData = mxGetPr(packetErrors);

    const int16_t* allI = static_cast<const int16_t*>(mxGetData(prhs[1]));
    const int16_t* allQ = static_cast<const int16_t*>(mxGetData(prhs[2]));

    double totalBitErrors = 0.0;
    double failedPackets = 0.0;
    double returnedBytes = 0.0;

    for (mwSize b = 0; b < batchSize; ++b) {
        mxArray* rxI = mxCreateNumericMatrix(samplesPerPacket,1,mxINT16_CLASS,mxREAL);
        mxArray* rxQ = mxCreateNumericMatrix(samplesPerPacket,1,mxINT16_CLASS,mxREAL);
        std::memcpy(mxGetData(rxI),allI + b*samplesPerPacket,
                    size_t(samplesPerPacket)*sizeof(int16_t));
        std::memcpy(mxGetData(rxQ),allQ + b*samplesPerPacket,
                    size_t(samplesPerPacket)*sizeof(int16_t));

        double scaleValue = 0.0;
        if (mxIsDouble(prhs[3])) {
            const double* p = mxGetDoubles(prhs[3]);
            scaleValue = p[scaleCount == 1 ? 0 : b];
        } else if (mxIsSingle(prhs[3])) {
            const float* p = mxGetSingles(prhs[3]);
            scaleValue = double(p[scaleCount == 1 ? 0 : b]);
        } else {
            scaleValue = scalar(prhs[3],"RxScale");
        }
        mxArray* scale = mxCreateDoubleScalar(scaleValue);

        mxArray* refBits = nullptr;
        if (mxIsInt8(prhs[14])) {
            refBits = mxCreateNumericMatrix(kPayloadBits,1,mxINT8_CLASS,mxREAL);
            const int8_t* src = static_cast<const int8_t*>(mxGetData(prhs[14]));
            const mwSize col = refCols == 1 ? 0 : b;
            std::memcpy(mxGetData(refBits),src + col*kPayloadBits,
                        size_t(kPayloadBits)*sizeof(int8_t));
        } else {
            refBits = mxCreateLogicalMatrix(kPayloadBits,1);
            const mxLogical* src = mxGetLogicals(prhs[14]);
            const mwSize col = refCols == 1 ? 0 : b;
            std::memcpy(mxGetLogicals(refBits),src + col*kPayloadBits,
                        size_t(kPayloadBits)*sizeof(mxLogical));
        }

        const mxArray* onePrhs[15] = {
            prhs[0], rxI, rxQ, scale, prhs[4], prhs[5], prhs[6], prhs[7],
            prhs[8], prhs[9], prhs[10], prhs[11], prhs[12], prhs[13], refBits
        };
        mxArray* oneOut[1] = {nullptr};
        e2eCommand(1,oneOut,15,onePrhs);

        mxArray* f = mxGetField(oneOut[0],0,"BitErrors");
        bitErrorData[b] = mxGetScalar(f);
        f = mxGetField(oneOut[0],0,"PacketErrors");
        packetErrorData[b] = mxGetScalar(f);
        f = mxGetField(oneOut[0],0,"PayloadChecksum");
        checksumData[b] = mxGetScalar(f);
        f = mxGetField(oneOut[0],0,"ReturnedBulkOutputBytes");
        returnedBytes += mxGetScalar(f);

        totalBitErrors += bitErrorData[b];
        failedPackets += packetErrorData[b];

        mxDestroyArray(oneOut[0]);
        mxDestroyArray(refBits);
        mxDestroyArray(scale);
        mxDestroyArray(rxQ);
        mxDestroyArray(rxI);
    }

    mxSetField(out,0,"BatchSize",mxCreateDoubleScalar(double(batchSize)));
    mxSetField(out,0,"PacketsProcessed",mxCreateDoubleScalar(double(batchSize)));
    mxSetField(out,0,"TotalBitErrors",mxCreateDoubleScalar(totalBitErrors));
    mxSetField(out,0,"FailedPackets",mxCreateDoubleScalar(failedPackets));
    mxSetField(out,0,"Checksums",checksums);
    mxSetField(out,0,"BitErrorsPerPacket",bitErrors);
    mxSetField(out,0,"PacketErrorsPerPacket",packetErrors);
    mxSetField(out,0,"ReturnedBulkOutputBytes",mxCreateDoubleScalar(returnedBytes));
    mxSetField(out,0,"Pass",mxCreateLogicalScalar(
        totalBitErrors == 0.0 && failedPackets == 0.0 && returnedBytes == 0.0));

    if (nlhs == 1) plhs[0] = out;
    else mxDestroyArray(out);
}

void submitCommand(
    int nlhs,mxArray* plhs[],int nrhs,const mxArray* prhs[])
{
    // command + 13 required inputs + optional reference payload.
    if (nrhs != 14 && nrhs != 15)
        mexErrMsgIdAndTxt(
            "eht_receiver_cuda_async:InputCount",
            "submit expects command plus 13 required inputs and "
            "an optional reference payload.");

    if (nlhs > 1)
        mexErrMsgIdAndTxt(
            "eht_receiver_cuda_async:OutputCount",
            "submit returns one ticket.");

    ensureInitialized();

    Slot* slot = findFreeSlot();
    if (!slot)
        mexErrMsgIdAndTxt(
            "eht_receiver_cuda_async:NoFreeSlot",
            "Both asynchronous slots are occupied.");

    double noiseVariance = scalar(prhs[3],"noiseVariance");
    double llrScale = scalar(prhs[4],"llrScale");
    int validationMode =
        static_cast<int>(scalar(prhs[5],"validationMode"));
    int ltfFFTStart =
        static_cast<int>(scalar(prhs[8],"ltfFFTStart"));

    if (!(noiseVariance > 0.0) || !(llrScale > 0.0))
        mexErrMsgIdAndTxt(
            "eht_receiver_cuda_async:Input",
            "noiseVariance and llrScale must be positive.");

    if (validationMode != 0 && validationMode != 1)
        mexErrMsgIdAndTxt(
            "eht_receiver_cuda_async:Input",
            "validationMode must be 0 or 1.");

    if (!mxIsNumeric(prhs[9]) || mxIsComplex(prhs[9]) ||
        mxGetNumberOfElements(prhs[9]) != kNumDataSymbols)
        mexErrMsgIdAndTxt(
            "eht_receiver_cuda_async:Input",
            "dataFFTStarts must contain %d real values.",kNumDataSymbols);

    int dataStarts[kNumDataSymbols];
    for (int sym = 0; sym < kNumDataSymbols; ++sym) {
        const double raw = mxIsDouble(prhs[9])
            ? mxGetDoubles(prhs[9])[sym]
            : double(mxGetSingles(prhs[9])[sym]);
        dataStarts[sym] = int(raw);
        if (double(dataStarts[sym]) != raw || dataStarts[sym] < 1)
            mexErrMsgIdAndTxt(
                "eht_receiver_cuda_async:Input",
                "FFT starts must be positive integers.");
    }
    if (ltfFFTStart < 1)
        mexErrMsgIdAndTxt(
            "eht_receiver_cuda_async:Input",
            "FFT starts must be positive integers.");

    initializeFrontendMetadata(
        prhs[6],  // ltfFFTIndices
        prhs[7],  // dataFFTIndices
        prhs[10], // knownLTF
        prhs[11], // pilotIndices
        prhs[12], // pilotReference
        prhs[13]  // dataIndices
    );

    copyComplexSingleWindow(
        prhs[1],ltfFFTStart-1,kLTFInputCount,
        slot->hLTF,"rxEHTLTF");

    for (int sym = 0; sym < kNumDataSymbols; ++sym) {
        copyComplexSingleWindow(
            prhs[2],dataStarts[sym]-1,kFFTLength,
            slot->hData+size_t(sym)*kFFTLength,"rxEHTData");
    }

    const mxArray* reference = nrhs == 15 ? prhs[14] : nullptr;
    if (validationMode == 1 && !reference)
        mexErrMsgIdAndTxt(
            "eht_receiver_cuda_async:Input",
            "Validation mode requires referencePayloadBits.");

    if (validationMode == 1) {
        if (mxIsComplex(reference) ||
            mxGetNumberOfElements(reference) != kPayloadBits ||
            !(mxIsInt8(reference) || mxIsLogical(reference)))
            mexErrMsgIdAndTxt(
                "eht_receiver_cuda_async:Input",
                "referencePayloadBits must be int8/logical "
                "with %d elements.",kPayloadBits);

        if (mxIsInt8(reference)) {
            const int8_t* p =
                static_cast<const int8_t*>(mxGetData(reference));
            for (int i = 0; i < kPayloadBits; ++i) {
                if (p[i] != 0 && p[i] != 1)
                    mexErrMsgIdAndTxt(
                        "eht_receiver_cuda_async:Input",
                        "referencePayloadBits must contain 0 or 1.");
                slot->hReferenceBits[i] = p[i];
            }
        } else {
            const mxLogical* p = mxGetLogicals(reference);
            for (int i = 0; i < kPayloadBits; ++i)
                slot->hReferenceBits[i] =
                    p[i] ? int8_t(1) : int8_t(0);
        }
    }

    slot->ticket = nextTicket++;
    slot->validationMode = validationMode;
    slot->occupied = true;
    *slot->hChecksum = 0;
    *slot->hBitErrors = 0;

    if (validationMode == 1) {
        CUDA_CHECK(cudaMemcpyAsync(
            slot->dReferenceBits,slot->hReferenceBits,
            size_t(kPayloadBits)*sizeof(int8_t),
            cudaMemcpyHostToDevice,slot->stream));
    }

    CUDA_CHECK(cudaMemsetAsync(
        slot->dChecksum,0,sizeof(unsigned long long),slot->stream));
    CUDA_CHECK(cudaMemsetAsync(
        slot->dBitErrors,0,sizeof(unsigned int),slot->stream));

    CUDA_CHECK(cudaEventRecord(slot->start,slot->stream));

    CUDA_CHECK(cudaMemcpyAsync(
        slot->dLTFInput,slot->hLTF,
        size_t(kLTFInputCount)*sizeof(cufftComplex),
        cudaMemcpyHostToDevice,slot->stream));
    CUDA_CHECK(cudaMemcpyAsync(
        slot->dDataInput,slot->hData,
        size_t(kDataInputCount)*sizeof(cufftComplex),
        cudaMemcpyHostToDevice,slot->stream));

    CUDA_CHECK(cudaEventRecord(slot->afterH2D,slot->stream));

    CUFFT_CHECK(cufftExecC2C(
        slot->ltfPlan,slot->dLTFInput,
        slot->dLTFOutput,CUFFT_FORWARD));
    CUFFT_CHECK(cufftExecC2C(
        slot->dataPlan,slot->dDataInput,
        slot->dDataOutput,CUFFT_FORWARD));

    int threads = 256;
    int blocks = (kActiveToneCount+threads-1)/threads;

    extractActiveKernel<<<blocks,threads,0,slot->stream>>>(
        slot->dLTFOutput,slot->dDataOutput,
        sharedData.ltfFFTIndices,
        sharedData.dataFFTIndices,
        slot->dLTFActive,slot->dDataActive);
    CUDA_CHECK(cudaGetLastError());

    estimateChannelKernel<<<blocks,threads,0,slot->stream>>>(
        slot->dLTFActive,sharedData.knownLTF,
        slot->dChannelEstimate);
    CUDA_CHECK(cudaGetLastError());

    estimatePilotRotationKernel<<<
        kNumDataSymbols,256,256*sizeof(cufftComplex),
        slot->stream>>>(
            slot->dDataActive,slot->dChannelEstimate,
            sharedData.pilotIndices,
            sharedData.pilotReference,
            sharedData.pilotCount,
            slot->dPilotRotation);
    CUDA_CHECK(cudaGetLastError());

    blocks = (kActiveToneCount*kNumDataSymbols+
        threads-1)/threads;
    equalizeKernel<<<blocks,threads,0,slot->stream>>>(
        slot->dDataActive,slot->dChannelEstimate,
        slot->dPilotRotation,float(noiseVariance),
        slot->dEqualizedActive);
    CUDA_CHECK(cudaGetLastError());

    segmentQuantizeCSIKernel<<<4,256,0,slot->stream>>>(
        slot->dEqualizedActive,slot->dChannelEstimate,
        sharedData.dataIndices,float(noiseVariance),
        slot->dI,slot->dQ,slot->dCSI,
        slot->dInverseQuantScales);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(
        slot->afterFrontend,slot->stream));

    blocks = (kTotalSymbols+threads-1)/threads;
    qam4096DemapKernel<<<blocks,threads,0,slot->stream>>>(
        slot->dI,slot->dQ,slot->dExternalLLR,kTotalSymbols,
        slot->dInverseQuantScales,float(1.0/noiseVariance),
        float(llrScale));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(slot->afterDemap,slot->stream));

    blocks = (kEncodedLength+threads-1)/threads;
    mapWeightKernel<<<blocks,threads,0,slot->stream>>>(
        slot->dExternalLLR,slot->dCSI,
        sharedData.encodedSourceIndex,
        float(1.0/llrScale),slot->dEncoded);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(slot->afterMap,slot->stream));

    dim3 reconBlock(256);
    dim3 reconGrid(
        kNumCodewords,(N+reconBlock.x-1)/reconBlock.x);

    reconstructKernel<<<reconGrid,reconBlock,0,slot->stream>>>(
        slot->dEncoded,sharedData.payloadBits,
        sharedData.punctureBits,sharedData.repeatBits,
        slot->dBeliefs,slot->dMessages);
    CUDA_CHECK(cudaGetLastError());

    layeredNMSKernel<<<kNumCodewords,128,0,slot->stream>>>(
        slot->dBeliefs,slot->dMessages);
    CUDA_CHECK(cudaGetLastError());

    dim3 decisionBlock(256);
    dim3 decisionGrid(
        kNumCodewords,(K+decisionBlock.x-1)/decisionBlock.x);

    hardDecisionKernel<<<decisionGrid,decisionBlock,0,slot->stream>>>(
        slot->dBeliefs,sharedData.payloadBits,slot->dDecoded);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(slot->afterLDPC,slot->stream));

    deriveScramblerSequenceKernel<<<1,1,0,slot->stream>>>(
        slot->dDecoded,slot->dScrambleSequence,
        slot->dDescrambleEnabled);
    CUDA_CHECK(cudaGetLastError());

    blocks = (kPayloadBits+threads-1)/threads;
    validatePayloadKernel<<<blocks,threads,0,slot->stream>>>(
        slot->dDecoded,slot->dScrambleSequence,
        slot->dDescrambleEnabled,validationMode,
        slot->dReferenceBits,slot->dChecksum,slot->dBitErrors);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(slot->afterValidate,slot->stream));

    CUDA_CHECK(cudaMemcpyAsync(
        slot->hChecksum,slot->dChecksum,
        sizeof(unsigned long long),
        cudaMemcpyDeviceToHost,slot->stream));
    CUDA_CHECK(cudaMemcpyAsync(
        slot->hBitErrors,slot->dBitErrors,
        sizeof(unsigned int),
        cudaMemcpyDeviceToHost,slot->stream));
    CUDA_CHECK(cudaEventRecord(slot->done,slot->stream));

    if (nlhs == 1)
        plhs[0] = mxCreateDoubleScalar(double(slot->ticket));
}

void collectCommand(
    int nlhs,mxArray* plhs[],int nrhs,const mxArray* prhs[])
{
    if (nrhs != 2)
        mexErrMsgIdAndTxt(
            "eht_receiver_cuda_async:InputCount",
            "collect expects command and ticket.");
    if (nlhs > 1)
        mexErrMsgIdAndTxt(
            "eht_receiver_cuda_async:OutputCount",
            "collect returns one stats structure.");

    ensureInitialized();

    uint64_t ticket =
        static_cast<uint64_t>(scalar(prhs[1],"ticket"));
    Slot* slot = findTicket(ticket);

    if (!slot)
        mexErrMsgIdAndTxt(
            "eht_receiver_cuda_async:UnknownTicket",
            "Ticket is not active or has already been collected.");

    CUDA_CHECK(cudaEventSynchronize(slot->done));

    if (nlhs == 1)
        plhs[0] = makeStats(*slot);

    slot->occupied = false;
    slot->ticket = 0;
    slot->validationMode = 0;
}

void pollCommand(
    int nlhs,mxArray* plhs[],int nrhs,const mxArray* prhs[])
{
    if (nrhs != 2)
        mexErrMsgIdAndTxt(
            "eht_receiver_cuda_async:InputCount",
            "poll expects command and ticket.");
    if (nlhs > 1)
        mexErrMsgIdAndTxt(
            "eht_receiver_cuda_async:OutputCount",
            "poll returns one logical scalar.");

    ensureInitialized();

    uint64_t ticket =
        static_cast<uint64_t>(scalar(prhs[1],"ticket"));
    Slot* slot = findTicket(ticket);

    if (!slot)
        mexErrMsgIdAndTxt(
            "eht_receiver_cuda_async:UnknownTicket",
            "Ticket is not active or has already been collected.");

    cudaError_t status = cudaEventQuery(slot->done);
    if (status != cudaSuccess && status != cudaErrorNotReady)
        CUDA_CHECK(status);

    if (nlhs == 1)
        plhs[0] = mxCreateLogicalScalar(status == cudaSuccess);
}

void statusCommand(int nlhs,mxArray* plhs[]) {
    ensureInitialized();

    const char* fields[] = {
        "BufferCount","OccupiedSlots","ActiveTickets",
        "Asynchronous","UsesPinnedHostMemory","UsesTwoCUDAStreams",
        "IntegratedFrontend","BulkIntermediateD2H"
    };

    mxArray* info = mxCreateStructMatrix(1,1,8,fields);
    int occupied = 0;
    mxArray* tickets = mxCreateDoubleMatrix(1,kNumSlots,mxREAL);
    double* ticketValues = mxGetPr(tickets);

    for (int i = 0; i < kNumSlots; ++i) {
        if (slots[i].occupied) ++occupied;
        ticketValues[i] =
            slots[i].occupied ? double(slots[i].ticket) : 0.0;
    }

    mxSetField(info,0,"BufferCount",
        mxCreateDoubleScalar(kNumSlots));
    mxSetField(info,0,"OccupiedSlots",
        mxCreateDoubleScalar(occupied));
    mxSetField(info,0,"ActiveTickets",tickets);
    mxSetField(info,0,"Asynchronous",
        mxCreateLogicalScalar(true));
    mxSetField(info,0,"UsesPinnedHostMemory",
        mxCreateLogicalScalar(true));
    mxSetField(info,0,"UsesTwoCUDAStreams",
        mxCreateLogicalScalar(true));
    mxSetField(info,0,"IntegratedFrontend",
        mxCreateLogicalScalar(true));
    mxSetField(info,0,"BulkIntermediateD2H",
        mxCreateLogicalScalar(false));

    if (nlhs == 1) plhs[0] = info;
    else mxDestroyArray(info);
}

void resetCommand() {
    ensureInitialized();
    for (int i = 0; i < kNumSlots; ++i) {
        if (slots[i].occupied) {
            CUDA_CHECK(cudaEventSynchronize(slots[i].done));
            slots[i].occupied = false;
            slots[i].ticket = 0;
            slots[i].validationMode = 0;
        }
    }
}

} // namespace

void mexFunction(
    int nlhs,mxArray* plhs[],int nrhs,const mxArray* prhs[])
{
    if (nrhs < 1)
        mexErrMsgIdAndTxt(
            "eht_receiver_cuda_async:Command",
            "A command is required.");

    std::string command = commandString(prhs[0]);

    if (command == "reconstruct") {
        reconstructCommand(nlhs,plhs,nrhs,prhs);
    } else if (command == "packetDetect") {
        packetDetectCommand(nlhs,plhs,nrhs,prhs);
    } else if (command == "coarseCFO") {
        coarseCFOCommand(nlhs,plhs,nrhs,prhs);
    } else if (command == "timingSync") {
        timingSyncCommand(nlhs,plhs,nrhs,prhs);
    } else if (command == "fineCFO") {
        fineCFOCommand(nlhs,plhs,nrhs,prhs);
    } else if (command == "part7") {
        part7Command(nlhs,plhs,nrhs,prhs);
    } else if (command == "e2e") {
        e2eCommand(nlhs,plhs,nrhs,prhs);
    } else if (command == "batchFrontend") {
        batchFrontendCommand(nlhs,plhs,nrhs,prhs);
    } else if (command == "e2eBatch") {
        e2eBatchCommand(nlhs,plhs,nrhs,prhs);
    } else if (command == "e2eBatchPrepare") {
        e2eBatchPrepareCommand(nlhs,plhs,nrhs,prhs);
    } else if (command == "e2eBatchSubmit") {
        e2eBatchSubmitCommand(nlhs,plhs,nrhs,prhs);
    } else if (command == "e2eBatchCollect") {
        e2eBatchCollectCommand(nlhs,plhs,nrhs,prhs);
    } else if (command == "e2eBatchPoll") {
        e2eBatchPollCommand(nlhs,plhs,nrhs,prhs);
    } else if (command == "e2eBatchReset") {
        if (nrhs != 1)
            mexErrMsgIdAndTxt(
                "eht_receiver_cuda_e2e:InputCount",
                "e2eBatchReset expects only the command.");
        e2eBatchResetCommand();
    } else if (command == "e2eBatchRun") {
        e2eBatchRunCommand(nlhs,plhs,nrhs,prhs);
    } else if (command == "submit") {
        submitCommand(nlhs,plhs,nrhs,prhs);
    } else if (command == "collect") {
        collectCommand(nlhs,plhs,nrhs,prhs);
    } else if (command == "poll") {
        pollCommand(nlhs,plhs,nrhs,prhs);
    } else if (command == "status") {
        if (nrhs != 1)
            mexErrMsgIdAndTxt(
                "eht_receiver_cuda_async:InputCount",
                "status expects only the command.");
        statusCommand(nlhs,plhs);
    } else if (command == "reset") {
        if (nrhs != 1)
            mexErrMsgIdAndTxt(
                "eht_receiver_cuda_async:InputCount",
                "reset expects only the command.");
        resetCommand();
    } else {
        mexErrMsgIdAndTxt(
            "eht_receiver_cuda_async:Command",
            "Unknown command: %s",command.c_str());
    }
}
