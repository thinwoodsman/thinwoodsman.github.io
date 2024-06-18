---
layout: post
title: "Quickie: ROCM and PyTorch \"RuntimeError: HIP error: invalid device function\""
date: 2024-06-13 20:08:13 -0600
tags: ROCm PyTorch HIP LLM AI
category: bug
---
[https://github.com/ROCm/ROCm/issues/2536](ROCm Issue 2536)


Solution: define ENV variable HSA_OVERRIDE_GFX_VERSION before importing torch:
```python
from os import putenv
putenv("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
import torch
```

Might be 10.3 if your card or libraries are older. Note that this can be prepended to a command line like any other ENV variable if you are getting the error in Someone Else's Code.
