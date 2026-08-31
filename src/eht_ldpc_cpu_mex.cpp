// CPU-only IEEE 802.11 rate-5/6, N=1944 layered normalized-min-sum decoder.
// MATLAB API:
//   decodedBits = eht_ldpc_cpu_mex(encodedData, ...
//       vecPayloadBits,vecShortenBits,vecPunctureBits,vecRepeatBits, ...
//       alpha,maxIterations)
//
// encodedData: real single column vector of post-FEC-padding LLRs.
// decodedBits: int8 column vector containing concatenated payload bits.

#include "mex.h"
#include <cstdint>
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>

namespace {
constexpr int Z = 81;
constexpr int BASE_ROWS = 4;
constexpr int BASE_COLS = 24;
constexpr int N = 1944;
constexpr int K = 1620;
constexpr int M = 324;
constexpr int BASE_EDGES = 79;
constexpr int EDGES = BASE_EDGES * Z;
constexpr float SHORTEN_LLR = 1.2676506002282294e30f; // 2^100

// Exact IEEE 802.11 N=1944, rate-5/6 QC shift matrix.
constexpr int8_t P[BASE_ROWS * BASE_COLS] = {
  13,48,80,66, 4,74, 7,30,76,52,37,60,-1,49,73,31,74,73,23,-1, 1, 0,-1,-1,
  69,63,74,56,64,77,57,65, 6,16,51,-1,64,-1,68, 9,48,62,54,27,-1, 0, 0,-1,
  51,15, 0,80,24,25,42,54,44,71,71, 9,67,35,-1,58,-1,29,-1,53, 0,-1, 0, 0,
  16,29,36,41,44,56,59,37,50,24,-1,65, 4,65,52,-1, 4,-1,73,52, 1,-1,-1, 0
};

template<typename T>
T scalarValue(const mxArray* a, const char* name) {
    if (!mxIsNumeric(a) || mxIsComplex(a) || mxGetNumberOfElements(a) != 1) {
        mexErrMsgIdAndTxt("eht_ldpc_cpu:Input", "%s must be a real scalar.", name);
    }
    return static_cast<T>(mxGetScalar(a));
}

std::vector<int> numericVectorToInt(const mxArray* a, const char* name) {
    if (!mxIsDouble(a) || mxIsComplex(a)) {
        mexErrMsgIdAndTxt("eht_ldpc_cpu:Input", "%s must be a real double vector.", name);
    }
    const size_t n = mxGetNumberOfElements(a);
    const double* p = static_cast<const double*>(mxGetData(a));
    std::vector<int> out(n);
    for (size_t i = 0; i < n; ++i) {
        const double x = p[i];
        if (!std::isfinite(x) || x < 0 || std::floor(x) != x) {
            mexErrMsgIdAndTxt("eht_ldpc_cpu:Input", "%s must contain nonnegative integers.", name);
        }
        out[i] = static_cast<int>(x);
    }
    return out;
}

void reconstructCodeword(const float* encoded, int in0, int payload,
                         int puncture, float* belief) {
    const int parityAvailable = M - puncture;
    for (int bit = 0; bit < N; ++bit) {
        if (bit < payload) {
            belief[bit] = encoded[in0 + bit];
        } else if (bit < K) {
            belief[bit] = SHORTEN_LLR;
        } else if (bit < K + parityAvailable) {
            belief[bit] = encoded[in0 + payload + (bit - K)];
        } else {
            belief[bit] = 0.0f; // punctured parity bit
        }
    }
}

void layeredNMS(float* L, float* R, float alpha, int maxIterations) {
    for (int iter = 0; iter < maxIterations; ++iter) {
        int layerEdgeBase = 0;
        for (int br = 0; br < BASE_ROWS; ++br) {
            // Each QC row r is independent within one base-matrix layer.
            for (int r = 0; r < Z; ++r) {
                float min1 = std::numeric_limits<float>::max();
                float min2 = std::numeric_limits<float>::max();
                int minCol = -1;
                int signProduct = 1;
                int edgeOrdinal = layerEdgeBase;

                // First pass: extrinsic values and two smallest magnitudes.
                for (int bc = 0; bc < BASE_COLS; ++bc) {
                    const int shift = static_cast<int>(P[br * BASE_COLS + bc]);
                    if (shift < 0) continue;
                    const int v = bc * Z + ((r + shift) % Z);
                    const int e = edgeOrdinal * Z + r;
                    const float q = L[v] - R[e];
                    const float a = std::fabs(q);
                    const int s = (q < 0.0f) ? -1 : 1;
                    signProduct *= s;
                    if (a < min1) {
                        min2 = min1;
                        min1 = a;
                        minCol = bc;
                    } else if (a < min2) {
                        min2 = a;
                    }
                    ++edgeOrdinal;
                }

                // Second pass: check-node result and layered belief update.
                edgeOrdinal = layerEdgeBase;
                for (int bc = 0; bc < BASE_COLS; ++bc) {
                    const int shift = static_cast<int>(P[br * BASE_COLS + bc]);
                    if (shift < 0) continue;
                    const int v = bc * Z + ((r + shift) % Z);
                    const int e = edgeOrdinal * Z + r;
                    const float oldR = R[e];
                    const float q = L[v] - oldR;
                    const int ownSign = (q < 0.0f) ? -1 : 1;
                    const float mag = (bc == minCol) ? min2 : min1;
                    const float newR = alpha * mag * static_cast<float>(signProduct * ownSign);
                    R[e] = newR;
                    L[v] = q + newR;
                    ++edgeOrdinal;
                }
            }

            // Advance to the first edge ordinal of the next base-matrix row.
            for (int bc = 0; bc < BASE_COLS; ++bc) {
                if (P[br * BASE_COLS + bc] >= 0) ++layerEdgeBase;
            }
        }
    }
}

} // namespace

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    if (nrhs != 7) {
        mexErrMsgIdAndTxt("eht_ldpc_cpu:Input",
            "Expected 7 inputs: encodedData, payload, shorten, puncture, repeat, alpha, maxIterations.");
    }
    if (nlhs > 1) {
        mexErrMsgIdAndTxt("eht_ldpc_cpu:Output", "One output is supported.");
    }
    if (!mxIsSingle(prhs[0]) || mxIsComplex(prhs[0])) {
        mexErrMsgIdAndTxt("eht_ldpc_cpu:Input", "encodedData must be real single.");
    }

    const float* encoded = static_cast<const float*>(mxGetData(prhs[0]));
    const int totalInput = static_cast<int>(mxGetNumberOfElements(prhs[0]));
    const std::vector<int> payload = numericVectorToInt(prhs[1], "vecPayloadBits");
    const std::vector<int> shorten = numericVectorToInt(prhs[2], "vecShortenBits");
    const std::vector<int> puncture = numericVectorToInt(prhs[3], "vecPunctureBits");
    const std::vector<int> repeat = numericVectorToInt(prhs[4], "vecRepeatBits");
    const int numCW = static_cast<int>(payload.size());

    if (numCW <= 0 || shorten.size() != payload.size() ||
        puncture.size() != payload.size() || repeat.size() != payload.size()) {
        mexErrMsgIdAndTxt("eht_ldpc_cpu:Input",
            "All codeword parameter vectors must have equal nonzero length.");
    }

    const float alpha = scalarValue<float>(prhs[5], "alpha");
    const int maxIterations = scalarValue<int>(prhs[6], "maxIterations");
    if (!(alpha > 0.0f && alpha <= 1.0f) || maxIterations <= 0) {
        mexErrMsgIdAndTxt("eht_ldpc_cpu:Input",
            "alpha must be in (0,1], and maxIterations must be positive.");
    }

    std::vector<int> inputOffset(numCW), outputOffset(numCW);
    int expectedInput = 0;
    int totalOutput = 0;
    for (int cw = 0; cw < numCW; ++cw) {
        if (payload[cw] + shorten[cw] != K || puncture[cw] > M || repeat[cw] < 0) {
            mexErrMsgIdAndTxt("eht_ldpc_cpu:Input",
                "Parameters are incompatible with WLAN N=1944 rate-5/6.");
        }
        inputOffset[cw] = expectedInput;
        outputOffset[cw] = totalOutput;
        expectedInput += payload[cw] + (M - puncture[cw]) + repeat[cw];
        totalOutput += payload[cw];
    }
    if (expectedInput != totalInput) {
        mexErrMsgIdAndTxt("eht_ldpc_cpu:Input",
            "encodedData length is %d; parameters require %d.", totalInput, expectedInput);
    }

    plhs[0] = mxCreateNumericMatrix(totalOutput, 1, mxINT8_CLASS, mxREAL);
    int8_t* out = static_cast<int8_t*>(mxGetData(plhs[0]));

    std::vector<float> belief(N);
    std::vector<float> messages(EDGES);

    for (int cw = 0; cw < numCW; ++cw) {
        std::fill(messages.begin(), messages.end(), 0.0f);
        reconstructCodeword(encoded, inputOffset[cw], payload[cw], puncture[cw], belief.data());
        layeredNMS(belief.data(), messages.data(), alpha, maxIterations);
        for (int i = 0; i < payload[cw]; ++i) {
            out[outputOffset[cw] + i] = (belief[i] < 0.0f) ? int8_t(1) : int8_t(0);
        }
    }
}
