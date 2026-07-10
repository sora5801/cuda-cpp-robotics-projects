// ===========================================================================
// tensorrt_path.cpp — OPTIONAL TensorRT engine path for project 12.01
//                     TensorRT deployment with custom CUDA pre/post kernels
//
// STATUS (read this first): this entire file compiles to a single stub
// function unless the build defines USE_TENSORRT (off by default — see
// README "Build"). The DEFAULT build of this project needs ZERO TensorRT
// headers or libraries; nothing else in src/ ever references anything in
// this file except the always-present tensorrt_path_available() query
// (CLAUDE.md §5 dependency policy, taken literally for this repo's first
// heavy-SDK project). The code inside #ifdef USE_TENSORRT below is written
// to the best of the author's documented knowledge of the TensorRT C++ API
// (the "modern", TensorRT 8.5+/10.x style: createNetworkV2, addConvolutionNd,
// buildSerializedNetwork, setTensorAddress + enqueueV3) — it has NOT been
// compiled or run in this session, because no TensorRT SDK is installed on
// the reference machine. Treat it as a carefully-reasoned reference, not a
// tested artifact: verify against your installed TensorRT version's headers
// (API details — especially around explicit-batch flags and the
// enqueueV2->enqueueV3 transition — have moved between major versions).
//
// THE KEY INSIGHT THIS FILE DEMONSTRATES
// ---------------------------------------
// Our fallback pipeline (main.cu) already treats "run the 2 conv layers
// and the detection head" as three device-pointer-in, device-pointer-out
// calls: launch_conv2d(d_net_chw, ..., d_conv1_out, ...), and so on. Every
// kernel AFTER the head — argmax_decode, threshold_box_decode, the IoU
// matrix, keypoint_extract — only ever touches device float* buffers by
// their documented shape (kernels.cuh SECTION 1). NONE of them know or
// care whether d_head_out was produced by three __global__ launches or by
// one TensorRT engine's context->enqueueV3(). That is deliberate: it is
// exactly how a real deployment is structured, and it is why swapping the
// inference core for TensorRT touches ONLY this file — the custom pre/post
// kernels this project is actually about are UNCHANGED, unlinked from
// TensorRT entirely, and still built, run, and verified by the default
// path. THEORY.md "What TensorRT actually does" expands on this.
//
// WHY NO CUSTOM TENSORRT PLUGIN? A production team gluing custom CUDA
// pre/post kernels to a TensorRT engine has two integration choices:
//   (a) keep pre/post as ORDINARY separate kernel launches that read/write
//       the engine's input/output DEVICE BUFFERS directly (what this file
//       does) — simplest, works with any TensorRT version, and is the
//       right choice whenever the custom op does not need to be part of
//       the engine's OWN captured graph;
//   (b) wrap the custom op as an IPluginV2DynamicExt PLUGIN so the builder
//       can place it INSIDE the engine (needed if you want the whole
//       pre->infer->post sequence captured as one CUDA graph, or if the
//       plugin needs to participate in the builder's layer-fusion/tactic
//       search). NMS-as-a-plugin is common in production detector
//       pipelines (e.g. NVIDIA's own EfficientNMS plugin, shipped with
//       TensorRT's OSS plugin library) for exactly that reason.
// This project takes path (a) — it is simpler, is the far more common
// integration for a first deployment, and keeps the custom kernels 100%
// TensorRT-independent (they build and run with zero TensorRT present,
// which is the whole point of the fallback-path design rule). The plugin
// alternative is real and named here so the reader knows it exists and
// when to reach for it (README "Prior art").
// ===========================================================================

#include "kernels.cuh"

#include <cstdio>

#ifdef USE_TENSORRT
// ===========================================================================
// Everything from here to the matching #else lives ONLY in a build that
// defines USE_TENSORRT and points its include/lib paths at an installed
// TensorRT SDK (README "Build" documents the exact flags). Not compiled,
// not linked, not exercised by the committed demo/expected_output.txt.
// ===========================================================================

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <cstring>
#include <fstream>
#include <memory>
#include <vector>

#include "util/cuda_check.cuh"

// ---------------------------------------------------------------------------
// Logger — every TensorRT object needs an nvinfer1::ILogger. TensorRT calls
// log() from its own threads during build and inference to report
// everything from fatal errors to per-tactic profiling chatter; a real
// deployment usually routes this into its own logging framework and
// filters by severity. We just forward warnings and errors to stderr,
// prefixed so they are visibly distinct from this project's own "[info]"
// lines (main.cu's output contract).
// ---------------------------------------------------------------------------
class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override
    {
        if (severity <= Severity::kWARNING)
            std::fprintf(stderr, "[trt] %s\n", msg);
    }
};

// nvinfer1::Weights wraps a host pointer TensorRT reads DURING network
// construction (addConvolutionNd) — the WeightBlob it points into must
// outlive the builder call, which it does here (build_engine holds it on
// its own stack for the whole function).
static nvinfer1::Weights wrap_weights(const float* data, int64_t count)
{
    nvinfer1::Weights w;
    w.type = nvinfer1::DataType::kFLOAT;
    w.values = data;
    w.count = count;
    return w;
}

// ---------------------------------------------------------------------------
// build_engine — construct an INetworkDefinition that is LAYER-FOR-LAYER
// equivalent to our hand-rolled conv1 -> conv2 -> head pipeline (same
// shapes, same weights, same ReLUs — kernels.cuh SECTION 1 is the single
// source of truth both paths read from), then let TensorRT's BUILDER
// compile it into an engine for the LOCAL GPU.
//
// WHAT THE BUILDER ACTUALLY DOES (conceptual — see THEORY.md for depth):
//   * LAYER FUSION — the builder recognizes patterns like "convolution
//     immediately followed by ReLU" and emits ONE fused kernel instead of
//     two, avoiding a round-trip to global memory for the intermediate
//     tensor. Our conv1+ReLU and conv2+ReLU are exactly this pattern.
//   * TACTIC SELECTION — for each layer, TensorRT has a LIBRARY of
//     candidate GPU kernels (different tiling strategies, different use of
//     Tensor Cores, different memory layouts) and BENCHMARKS several of
//     them on the actual target GPU during the build call, keeping the
//     fastest. This is why buildSerializedNetwork() can take seconds even
//     for a tiny network, and why a serialized engine is NOT portable
//     across GPU architectures — it encodes the winning tactics for the
//     specific GPU it was built on.
//   * PRECISION — setFlag(BuilderFlag::kFP16) below asks the builder to
//     additionally consider FP16 tactics wherever the layer supports them
//     (our weights and activations are small enough that FP16 does not
//     lose meaningful precision here). INT8 needs an extra CALIBRATION
//     step (an IInt8Calibrator fed a representative batch of inputs, used
//     to pick a per-tensor scale factor that maps the observed float range
//     into signed 8-bit) which this teaching integration does not attempt
//     — see THEORY.md "Numerical considerations" for why INT8 calibration
//     is a real, separate piece of engineering, not a flag flip.
// ---------------------------------------------------------------------------
static bool build_engine(Logger& logger, const WeightBlob& w,
                         std::vector<char>& out_serialized)
{
    std::unique_ptr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger));
    if (!builder) return false;

    // "Explicit batch" (batch size folded into the tensor's own leading
    // dimension rather than implied) has been the only supported mode
    // since TensorRT 7 — the flag is kept here for readability/portability
    // across SDK versions that still require it to be passed explicitly.
    const auto flags = 1U << static_cast<uint32_t>(
        nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    std::unique_ptr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(flags));
    if (!network) return false;

    // Input tensor: matches d_net_chw exactly — CHW, batch size 1 (our
    // preprocessing kernel already produced this; the engine consumes it
    // as-is, so NO TensorRT layer duplicates the resize/normalize work).
    nvinfer1::ITensor* input = network->addInput(
        "net_chw", nvinfer1::DataType::kFLOAT,
        nvinfer1::Dims4{1, kConv1In, kNetH, kNetW});
    if (!input) return false;

    // conv1: 3->2, 3x3, stride 2, pad 1, ReLU.
    nvinfer1::Weights c1w = wrap_weights(&w.conv1_w[0][0][0][0],
                                         static_cast<int64_t>(kConv1Out) * kConv1In * kConv1K * kConv1K);
    nvinfer1::Weights c1b = wrap_weights(&w.conv1_b[0], kConv1Out);
    nvinfer1::IConvolutionLayer* conv1 = network->addConvolutionNd(
        *input, kConv1Out, nvinfer1::DimsHW{kConv1K, kConv1K}, c1w, c1b);
    if (!conv1) return false;
    conv1->setStrideNd(nvinfer1::DimsHW{kConv1Stride, kConv1Stride});
    conv1->setPaddingNd(nvinfer1::DimsHW{kConv1Pad, kConv1Pad});
    nvinfer1::IActivationLayer* relu1 =
        network->addActivation(*conv1->getOutput(0), nvinfer1::ActivationType::kRELU);
    if (!relu1) return false;

    // conv2: 2->2, 3x3, stride 2, pad 1, ReLU.
    nvinfer1::Weights c2w = wrap_weights(&w.conv2_w[0][0][0][0],
                                         static_cast<int64_t>(kConv2Out) * kConv2In * kConv2K * kConv2K);
    nvinfer1::Weights c2b = wrap_weights(&w.conv2_b[0], kConv2Out);
    nvinfer1::IConvolutionLayer* conv2 = network->addConvolutionNd(
        *relu1->getOutput(0), kConv2Out, nvinfer1::DimsHW{kConv2K, kConv2K}, c2w, c2b);
    if (!conv2) return false;
    conv2->setStrideNd(nvinfer1::DimsHW{kConv2Stride, kConv2Stride});
    conv2->setPaddingNd(nvinfer1::DimsHW{kConv2Pad, kConv2Pad});
    nvinfer1::IActivationLayer* relu2 =
        network->addActivation(*conv2->getOutput(0), nvinfer1::ActivationType::kRELU);
    if (!relu2) return false;

    // head: 2->6, 1x1, LINEAR (no activation) — a 1x1 conv, same op as
    // above with K=1, pad=0, stride=1 (kernels.cuh's "1x1 conv is a
    // per-cell linear layer" point holds inside TensorRT too).
    nvinfer1::Weights hw = wrap_weights(&w.head_w[0][0][0][0],
                                        static_cast<int64_t>(kHeadOut) * kHeadIn * kHeadK * kHeadK);
    nvinfer1::Weights hb = wrap_weights(&w.head_b[0], kHeadOut);
    nvinfer1::IConvolutionLayer* head = network->addConvolutionNd(
        *relu2->getOutput(0), kHeadOut, nvinfer1::DimsHW{kHeadK, kHeadK}, hw, hb);
    if (!head) return false;
    head->setStrideNd(nvinfer1::DimsHW{kHeadStride, kHeadStride});
    head->setPaddingNd(nvinfer1::DimsHW{kHeadPad, kHeadPad});

    head->getOutput(0)->setName("head_out");
    network->markOutput(*head->getOutput(0));

    std::unique_ptr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
    if (!config) return false;
    // Workspace: scratch memory the builder may hand a tactic (im2col
    // buffers, etc.). This network is tiny; 64 MiB is generous headroom.
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE,
                               static_cast<size_t>(64) * 1024 * 1024);
    if (builder->platformHasFastFp16())
        config->setFlag(nvinfer1::BuilderFlag::kFP16);   // see the precision note above

    std::unique_ptr<nvinfer1::IHostMemory> serialized(
        builder->buildSerializedNetwork(*network, *config));
    if (!serialized) return false;

    out_serialized.assign(static_cast<const char*>(serialized->data()),
                          static_cast<const char*>(serialized->data()) + serialized->size());
    return true;
}

// ---------------------------------------------------------------------------
// run_optional_tensorrt_demo — build (or would-be-cached) engine, run ONE
// inference on the preprocessed tensor our OWN preprocess kernel already
// produced (d_net_chw, from main.cu — reused unmodified), and hand the
// engine's output buffer directly to our existing post-processing kernels.
// Every call below (argmax/threshold/iou/keypoint) is the EXACT SAME
// function main.cu's fallback path calls — this is the payoff of the
// device-buffer-shaped kernel contract in kernels.cuh.
// ---------------------------------------------------------------------------
void run_optional_tensorrt_demo(const char* weights_path, const float* d_net_chw)
{
    std::printf("[info] [trt] building TensorRT engine from %s ...\n", weights_path);

    // Re-load the weight blob independently (main.cu already validated it
    // for the fallback path; re-reading here keeps this function
    // self-contained and easy to lift into a standalone tool).
    std::ifstream in(weights_path, std::ios::binary);
    if (!in.is_open()) {
        std::fprintf(stderr, "[trt] could not reopen %s\n", weights_path);
        return;
    }
    WeightBlob w{};
    char magic[8]; uint32_t version = 0;
    in.read(magic, 8);
    in.read(reinterpret_cast<char*>(&version), sizeof(version));
    in.read(reinterpret_cast<char*>(&w.conv1_w[0][0][0][0]), sizeof(w.conv1_w));
    in.read(reinterpret_cast<char*>(&w.conv1_b[0]),          sizeof(w.conv1_b));
    in.read(reinterpret_cast<char*>(&w.conv2_w[0][0][0][0]), sizeof(w.conv2_w));
    in.read(reinterpret_cast<char*>(&w.conv2_b[0]),          sizeof(w.conv2_b));
    in.read(reinterpret_cast<char*>(&w.head_w[0][0][0][0]),  sizeof(w.head_w));
    in.read(reinterpret_cast<char*>(&w.head_b[0]),           sizeof(w.head_b));
    if (!in) { std::fprintf(stderr, "[trt] short read on %s\n", weights_path); return; }

    Logger logger;
    std::vector<char> engine_bytes;
    if (!build_engine(logger, w, engine_bytes)) {
        std::fprintf(stderr, "[trt] engine build FAILED\n");
        return;
    }
    std::printf("[info] [trt] engine built: %zu bytes serialized\n", engine_bytes.size());

    std::unique_ptr<nvinfer1::IRuntime> runtime(nvinfer1::createInferRuntime(logger));
    std::unique_ptr<nvinfer1::ICudaEngine> engine(
        runtime->deserializeCudaEngine(engine_bytes.data(), engine_bytes.size()));
    std::unique_ptr<nvinfer1::IExecutionContext> context(engine->createExecutionContext());
    if (!engine || !context) {
        std::fprintf(stderr, "[trt] deserialize/context creation FAILED\n");
        return;
    }

    // Output buffer: same shape/layout our own conv2d_kernel chain would
    // have produced (kHeadOut, kGridH, kGridW) — this is what makes the
    // downstream kernels reusable without modification.
    float* d_head_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_head_out, static_cast<size_t>(kHeadOut) * kGridH * kGridW * sizeof(float)));

    // TensorRT 8.5+/10.x "named tensor address" binding API, replacing the
    // older positional-binding-index style: each I/O tensor is set by the
    // NAME given via setName()/addInput() above.
    context->setTensorAddress("net_chw", const_cast<float*>(d_net_chw));
    context->setTensorAddress("head_out", d_head_out);

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));
    const bool ok = context->enqueueV3(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
    if (!ok) {
        std::fprintf(stderr, "[trt] enqueueV3 FAILED\n");
        CUDA_CHECK(cudaFree(d_head_out));
        return;
    }

    // From here down: IDENTICAL calls to main.cu's fallback path. This is
    // the point of the whole file — swap the inference core, keep the
    // custom pre/post kernels.
    const int ncells = kGridH * kGridW;
    int* d_best_class = nullptr; float* d_best_score = nullptr;
    Detection* d_candidates = nullptr; int* d_count = nullptr;
    CUDA_CHECK(cudaMalloc(&d_best_class, static_cast<size_t>(ncells) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_best_score, static_cast<size_t>(ncells) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_candidates, static_cast<size_t>(kMaxCandidates) * sizeof(Detection)));
    CUDA_CHECK(cudaMalloc(&d_count, sizeof(int)));

    launch_argmax_decode(d_head_out, d_best_class, d_best_score);
    launch_threshold_box_decode(d_best_class, d_best_score, d_head_out, d_candidates, d_count);
    int n = 0;
    CUDA_CHECK(cudaMemcpy(&n, d_count, sizeof(int), cudaMemcpyDeviceToHost));
    std::printf("[info] [trt] TensorRT-engine path: %d pre-NMS candidates (informational only — "
               "not part of the checked stable output; see README/THEORY for why)\n", n);

    CUDA_CHECK(cudaFree(d_head_out));
    CUDA_CHECK(cudaFree(d_best_class));
    CUDA_CHECK(cudaFree(d_best_score));
    CUDA_CHECK(cudaFree(d_candidates));
    CUDA_CHECK(cudaFree(d_count));
}

bool tensorrt_path_available() { return true; }

#else  // ---------------------------------------------------------------
// Default build: TensorRT SDK not required, not referenced, not linked.
// This translation unit still defines the always-present query function
// (declared unconditionally in kernels.cuh) so main.cu can report the
// build's TensorRT status uniformly regardless of which configuration
// produced this .exe — and so this object file is never empty (avoids the
// MSVC LNK4221 "no public symbols" warning, keeping the zero-new-warnings
// gate honest rather than suppressed).
bool tensorrt_path_available() { return false; }

#endif // USE_TENSORRT
