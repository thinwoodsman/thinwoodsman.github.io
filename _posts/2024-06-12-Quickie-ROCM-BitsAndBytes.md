---
layout: post
title: "Quickie: ROCm bitsandbytes 'NoneType' object has no attribute 'split'"
date: 2024-06-12 20:08:13 -0600
tags: ROCm Transformers BitsAndBytes LLM AI
category: bugs
---
So you're trying to serve a model on some half-baked LLM project like LMQL, and you've got pytorch-rocm installed because you're using an AMD video card, and you get stuck on the bitsandbytes CUDA check:

```
File "/usr/local/lib/python3.11/dist-packages/bitsandbytes/cuda_specs.py", line 24, in get_cuda_version_tuple
    major, minor = map(int, torch.version.cuda.split("."))
                            ^^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: 'NoneType' object has no attribute 'split'
```
Well, the root cause here is that pytorch-rocm doesn't return a CUDA version, because why would it, nor does it have sense enough to fake it. So if you've got an ROCm card, then torch.cuda.is_available() returns True, but then torch.version.cuda returns None, which is just a big drag for everybody.


And bitsandbytes, which like most of these half-baked AI projects doesn't understand that you need to be flexible about your requirements in order to avoid the dependency hell that makes people think shipping an entire copy of the developer's work environment (aka a docker container or a Python venv) is a good idea, checks the CUDA version assuming it has to be non-NULL because after all there's CUDA devices present, aren't there?


OK, so the easy fix, when writing your own code, is to simply do this:
```python
import torch
torch.version.cuda = "10.1"
```
ideally before doing anything else. You never know what dummy is going to hide that CUDA check in their init.py.


The harder fix, for when you're using someone else's code (like the aforementioned LMQL), is to patch the bitsandbytes code:


`vi /usr/local/lib/python3.11/dist-packages/bitsandbytes/cuda_specs.py`

```python
def get_cuda_version_tuple() -> Tuple[int, int]:
	if torch.version.cuda:
        major, minor = map(int, torch.version.cuda.split("."))
	else:
        major = 10
        minor = 1
    return major, minor
```

That's all there is to it. Now get back to work!

NOTE: There is a third, proper solution, which involves removing bitsandbytes and building+installing bitsandbytes-rocm, but while that is the proper solution, it requires build tools from Nvidia (according to [https://github.com/agrocylo/bitsandbytes-rocm/blob/main/compile_from_source.md](Compile BitsAndBytes-ROCm from source) ), and who got time for that?
