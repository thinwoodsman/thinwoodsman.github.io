---
layout: post
title: "Quickie: ROCM VLLM build on a Framework 16"
date: 2024-06-20 18:27:13 -0600
tags: ROCm PyTorch HIP LLM AI Quickies
category: bug
---
Quick background: The Framework 16 has a Radeon 7700S discrete GPU in addition to its Radeon 780M integrated GPU. HIP detects these cards as gfx1102 and gfx1103 (I haven't determined which is which).

Right off the bat, VLLM requires CUDA, which is a no-go if you haven't overpaid for an NVidia card in recent years. But there is a [https://github.com/EmbeddedLLM/vllm-rocm](VLLM ROCm fork) and it *seems* like it should work.

Problem 1:  Dependency on flash-attention
Flash-Attention *also* requires CUDA to work (see a pattern here? NVidia is the guy selling shovels during the AI gold rush), but there is a [https://github.com/ROCm/flash-attention](Flash-Attention ROCm fork), and that should do nicely.

According to [https://www.reddit.com/r/ROCm/comments/1aslqba/comment/kstal0h/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button](this comment), the `howiejay/navi_support` branch has to be built, in order to support AMD video cards with RDNA3 (Radeon 7600-7900, roughly):
```sh
git clone --recursive https://github.com/ROCm/flash-attention.git
cd flash-attention
git checkout howiejay/navi_support
pip install -e .
```
Problem 2: undeclared identifier `CK_BUFFER_RESOURCE_3RD_DWORD` 
The flash-attention build fails due to an undeclared identifier which, if you grep -r for it, is right there in ck.hpp. 

Solution:
Now, you can spelunk the CMake files to try to find which toggle to flip, but according to [https://github.com/ROCm/composable_kernel/issues/775](ROCm composable_kernel issue 775) the proper fix is to make the following modification to csrc/flash_attn_rocm/composable_kernel/include/ck/ck.hpp :
```c++
#elif defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) // for GPU code
#define CK_BUFFER_RESOURCE_3RD_DWORD 0x31004000
#endif
```
becomes
```c++
#elif defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) // for GPU code
#define CK_BUFFER_RESOURCE_3RD_DWORD 0x31004000
#endif
```
That fixes the flash-attention build for ROCm. Back to VLLM.

Problem 3: architectures "gfx1102;gfx1103" are not supported
VLLM has, you guessed it, hard-coded chip architectures that it supports. No wonder nobody can get this stuff to build! Have we returned to the 70s or something?
```
CMake Warning at CMakeLists.txt:124 (message):
  Pytorch version 2.1.1 expected for ROCMm 6.x build, saw 2.3.0 instead.


-- HIP supported arches: gfx906;gfx908;gfx90a;gfx940;gfx941;gfx942;gfx1030;gfx1100
CMake Error at cmake/utils.cmake:166 (message):
  None of the detected ROCm architectures: gfx1102;gfx1103 is supported.
  Supported ROCm architectures are:
  gfx906;gfx908;gfx90a;gfx940;gfx941;gfx942;gfx1030;gfx1100.
```
OK, this one *can* be solved by poking around in the CMake files. Specifically, in CMakeLists.txt change
```c++
# Supported AMD GPU architectures.
set(HIP_SUPPORTED_ARCHS "gfx906;gfx908;gfx90a;gfx940;gfx941;gfx942;gfx1030;gfx11
00;")
```
to
```c++
# Supported AMD GPU architectures.
set(HIP_SUPPORTED_ARCHS "gfx906;gfx908;gfx90a;gfx940;gfx941;gfx942;gfx1030;gfx11
00;gfx1102;gfx1103;") 
```
OK, the build chugs along nicely for a bit, until...

Problem 4: duplicate symbol in ROCm objects
```
[12/12] Linking HIP shared module /hom...llm/_C.cpython-311-x86_64-linux-gnu.so
FAILED: /home/build/ai/vllm-rocm/build/lib.linux-x86_64-cpython-311/vllm/_C.cpython-311-x86_64-linux-gnu.so

ld.lld: error: duplicate symbol: __float2bfloat16(float)
>>> defined at amd_hip_bf16.h:146 (/opt/rocm-6.0.2/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:146)
...
clang: error: linker command failed with exit code 1 (use -v to see invocation)
ninja: build stopped: subcommand failed.
```
OK, I attempted to build ROCm from scratch to address this, and encountered two immediate problems: 1) the current /opt/rocm is 22GB (how is this OK?!?!) and i might use up my bandwidth and/or available drive space building a new one, but more importantly 2) the ROCm build scripts are incredibly broken. Seriously, they make a lot of assumptions that, it turns out, don't hold on my system. "Be liberal with what you accept, but conservative in what you send" ring a bell, anyone?

Solution:
This one took awhile to track down, as it's an error in ROCm and not in one of these half-baked AI projects (meaning, presumably, there are *real* software engineers behind this, not purported "researchers"). But it is reported in [https://github.com/vllm-project/vllm/issues/2725](VLLM issue 2725) and a [https://github.com/vllm-project/vllm/pull/2790/files](patch) to ROCm is referenced. You can apply the patch or, if you want to save some time downloading it and checking it, you can
```sh
find /opt/rocm -name amd_hip_bf16.h
```
and change
```c++
#if defined(__HIPCC_RTC__)
#define __HOST_DEVICE__ __device__
#else
#include <climits>
#define __HOST_DEVICE__ __host__ __device__
#endif
```
to
```c++
#if defined(__HIPCC_RTC__)
#define __HOST_DEVICE__ __device__ static
#else
#include <climits>
#define __HOST_DEVICE__ __host__ __device__ static inline
#endif
```

And lo, it builds! And installs! But does it run?

```python
#!/usr/bin/env python
from os import putenv

putenv("DRI_PRIME", "1")
putenv("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
putenv("TRANSFORMERS_OFFLINE", "1")

from vllm import LLM, SamplingParams
prompts = [
        "How to explain Internet for a medieval knight?",
        "How to smoke pork ribs for barbeque",
        "What is the population, political system, GDP, and chief exports of Liberia?"

]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=500)
llm = LLM(model="microsoft/Phi-3-mini-128k-instruct", device="auto")
outputs = llm.generate(prompts, sampling_params)

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

```
Nope!
```
  File "/usr/local/lib/python3.11/dist-packages/vllm-0.4.0.post1+rocm603-py3.11-linux-x86_64.egg/vllm/config.py", line 1115, in _get_and_verify_max_len
    assert "factor" in rope_scaling
           ^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError
```
Seriously, to hell with this garbage. Like every other AI project, they probably
expect me to use a Python Venv or a Docker image so I can download a 5GB copy
of the developer's workstation to my location machine in order to make their
specific piece of software work. The industry has forgotten how to code.
