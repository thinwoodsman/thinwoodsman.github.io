---
layout: post
title: "Working with HuggingFace models locally"
date: 2024-06-17 17:08:13 -0600
tags: HuggingFace LLamaCPP Transformers LLM AI
category: ai-toolkits
---
#Working offline

In addition to the 

`os.putenv("TRANSFORMERS_OFFLINE", "1")

Here's a bonus one for you, to shut TF up a bit:

`os.putenv('TF_CPP_MIN_LOG_LEVEL', '3') # filter out all messages

#Checkpoint Sharts

OK, fine, checkpoint shards. Point is, nobody likes seeing these things download at the start of their script.

#Dealing with missing files

The way these HF models tend to work is that there's a bare git repo, with all of the actual code/data in the blobs directory, and snapshots that contain symlinks which provide filenames for those BLOBs.. Quite often, a missing file just means the symlinks weren't created correctly - could be a network timeout, could be the user interrupted by 

#Generating ONNX
ONNX is a handy format, and allows you to use tools like [https://github.com/zetane/viewer](Zetane) and [https://github.com/lutzroeder/Netron](Netron) to examine a model. A generic ONNX runtime is provided by [https://onnxruntime.ai/](Optimum), and they also provide a utility to convert Pytorch and Tensorflow models to ONNX:
```
pip install optimum[exporters]
```

This will probably downgrade your transformers package, because AI developers are idiots, but just make a note of it and do an upgrade afterwards:
```sh
bash# pip install optimum[exporters]
  ...
      Successfully uninstalled transformers-4.41.2
Successfully installed transformers-4.40.2
bash# pip install --upgrade transformers
Collecting transformers
  Using cached transformers-4.41.2-py3-none-any.whl.metadata (43 kB)
  ...
      Successfully uninstalled transformers-4.40.2
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
optimum 1.19.2 requires transformers[sentencepiece]<4.41.0,>=4.26.0, but you have transformers 4.41.2 which is incompatible.
Successfully installed transformers-4.41.2
```
The optimum-cli still works, because all well-behaved software is backwards-compatible across minor version numbers, which is why you *never specify a maximum version number for a dependency unless there is absolutely no alternative*:
```sh
bash# optimum-cli export onnx --model ~/.cache/huggingface/hub/models--t5-base --task text2text-generation  /opt/ai-models/t5-base-text2text.onnx
...
The ONNX export succeeded and the exported model was saved at: /opt/ai-models/t5-base-text2text.onnx
```
If you don't specify a task, you'll probably get an error message listing the
available tasks to choose from. These currently are:

    * object-detection
    * feature-extraction
    * question-answering
    * automatic-speech-recognition
    * audio-xvector
    * sentence-similarity
    * image-segmentation
    * semantic-segmentation
    * image-classification
    * zero-shot-image-classification
    * zero-shot-object-detection
    * token-classification
    * text-to-audio
    * stable-diffusion-xl
    * text2text-generation
    * stable-diffusion
    * text-classification
    * conversational
    * image-to-image
    * mask-generation
    * text-generation
    * audio-frame-classification
    * fill-mask
    * depth-estimation
    * audio-classification
    * multiple-choice
    * image-to-text
    * masked-im

Look your model up on HuggingFace if you aren't sure what tasks it supports.


#Generating GGUF
Many AI front-ends have started using the GGUF format for loading models, 
which is the format supported by [https://github.com/ggerganov/llama.cpp](llama.cpp). You can end up pretty far in the weeds trying to find out how to get
llama.cpp to convert your favorite non-Llama model. Here's a pretty basic
shell script which will do the job, provided you set DEFAULT_OUTPUT_DIR and 
LLAMA_CPP_DIR to the correct locations for your local machine:
```sh
#!/usr/bin/env bash
DEFAULT_OUTPUT_DIR="/opt/ai-models"
LLAMA_CPP_DIR="/opt/llama.cpp"

MODEL_PATH="$1"
if [[ "$MODEL_PATH" != /* ]]
then
	MODEL_PATH="$PWD/$1"
fi
MODEL_NAME=`basename $MODEL_PATH | sed 's/^models--//'`

DTYPE="$2" # e.g.  'f32', 'f16', 'bf16', 'q8_0', 'auto'
if [ "$DTYPE" = "" ]
then
	DTYPE="auto"
fi

if [ "$GGUF_DIR" = "" ]
then
	GGUF_DIR="$DEFAULT_OUTPUT_DIR"
fi
GGUF_NAME="$GGUF_DIR/$MODEL_NAME.gguf"


echo $MODEL_PATH " @ " $DTYPE
# ----------------------------------------------------------------------
cd $LLAMA_CPP_DIR

MODEL_TYPE_FLAG=""
if [ "$GGUF_MODEL_NAME" != "" ]
then
	MODEL_TYPE_FLAG="--model-name $GGUF_MODEL_NAME"
fi

# make GGUF
echo ./convert-hf-to-gguf.py --outtype=$DTYPE $MODEL_TYPE_FLAG --outfile=$GGUF_NAME $MODEL_PATH
./convert-hf-to-gguf.py --outtype=$DTYPE $MODEL_TYPE_FLAG --outfile=$GGUF_NAME $MODEL_PATH
```
It's a bit of a toss-up, whether this script will *just work* or not. Here are
some problems you are likely to encounter, along with possible solutions:

#Quantizing GGUF
A lot of AI models are going to be too large to load on something like a 
laptop, even one with a discrete video card. Once you have the GGUF, though, 
you can quantize the model, which basically means using a less precise datatype
(for example, 1 byte or 4 bits per weight, instead of 4 bytes) in order to 
save a bit of memory. Keep in mind that the number of parameters in a model,
for example 7B(illion), is the number of weights, so multiply that number by
2 for a float16 model (14B) or 4 for a float32 model (28B) to get the memory
required (in gigabytes) to run that model. Quantizing a 7B float16 model (14Gb) to 4 bits (3.5Gb) makes running the model on commodity hardware much more
feasible. 

Of course, you are reducing the precision of the weights, so each 
weight will be less distinct from its fellows, and this will gradually narrow 
the options that the model has for generating output the as you reduce the 
precision. Keep this in when determining whether to quantize a particular
model: for an assistant, or a grammar checker, or a tool-using agent, the
reduction in output possibilities might not matter, but for something like 
code generation or medical diagnosis, the extra precision will capture edge
cases that you do not want to discard.
```sh
#!/usr/bin/env bash
DEFAULT_OUTPUT_DIR="/opt/ai-models"
LLAMA_CPP_DIR="/opt/llama.cpp"

GGUF_PATH="$1"
if [[ "$GGUF_PATH" != /* ]]
then
	GGUF_PATH="$PWD/$1"
fi

QUANT="$2" # e.g.  "q4_k_m" "q3_k_m" "q2_k"
if [ "$QUANT" = "" ]
then
	QUANT="q4_k_m"
fi

MODEL_NAME=`basename $GGUF_PATH | sed 's/\.gguf$//'`
if [ "$GGUF_DIR" = "" ]
then
	GGUF_DIR="$DEFAULT_OUTPUT_DIR"
fi

QUANT_NAME="$GGUF_DIR/$MODEL_NAME-$QUANT.gguf"

# ----------------------------------------------------------------------
cd $LLAMA_CPP_DIR

echo ./llama-quantize $GGUF_PATH $QUANT_NAME $QUANT
./llama-quantize $GGUF_PATH $QUANT_NAME $QUANT
```
to get a list of support quantization levels:
```sh
bash# llama.cpp/llama-quantize --help
Allowed quantization types:
   2  or  Q4_0    :  3.56G, +0.2166 ppl @ LLaMA-v1-7B
   3  or  Q4_1    :  3.90G, +0.1585 ppl @ LLaMA-v1-7B
   8  or  Q5_0    :  4.33G, +0.0683 ppl @ LLaMA-v1-7B
   9  or  Q5_1    :  4.70G, +0.0349 ppl @ LLaMA-v1-7B
  19  or  IQ2_XXS :  2.06 bpw quantization
  20  or  IQ2_XS  :  2.31 bpw quantization
  28  or  IQ2_S   :  2.5  bpw quantization
  29  or  IQ2_M   :  2.7  bpw quantization
  24  or  IQ1_S   :  1.56 bpw quantization
  31  or  IQ1_M   :  1.75 bpw quantization
  10  or  Q2_K    :  2.63G, +0.6717 ppl @ LLaMA-v1-7B
  21  or  Q2_K_S  :  2.16G, +9.0634 ppl @ LLaMA-v1-7B
  23  or  IQ3_XXS :  3.06 bpw quantization
  26  or  IQ3_S   :  3.44 bpw quantization
  27  or  IQ3_M   :  3.66 bpw quantization mix
  12  or  Q3_K    : alias for Q3_K_M
  22  or  IQ3_XS  :  3.3 bpw quantization
  11  or  Q3_K_S  :  2.75G, +0.5551 ppl @ LLaMA-v1-7B
  12  or  Q3_K_M  :  3.07G, +0.2496 ppl @ LLaMA-v1-7B
  13  or  Q3_K_L  :  3.35G, +0.1764 ppl @ LLaMA-v1-7B
  25  or  IQ4_NL  :  4.50 bpw non-linear quantization
  30  or  IQ4_XS  :  4.25 bpw non-linear quantization
  15  or  Q4_K    : alias for Q4_K_M
  14  or  Q4_K_S  :  3.59G, +0.0992 ppl @ LLaMA-v1-7B
  15  or  Q4_K_M  :  3.80G, +0.0532 ppl @ LLaMA-v1-7B
  17  or  Q5_K    : alias for Q5_K_M
  16  or  Q5_K_S  :  4.33G, +0.0400 ppl @ LLaMA-v1-7B
  17  or  Q5_K_M  :  4.45G, +0.0122 ppl @ LLaMA-v1-7B
  18  or  Q6_K    :  5.15G, +0.0008 ppl @ LLaMA-v1-7B
   7  or  Q8_0    :  6.70G, +0.0004 ppl @ LLaMA-v1-7B
   1  or  F16     : 14.00G, -0.0020 ppl @ Mistral-7B
  32  or  BF16    : 14.00G, -0.0050 ppl @ Mistral-7B
   0  or  F32     : 26.00G              @ 7B
          COPY    : only copy tensors, no quantizing
```

#Quantizing ONNX
Can you quantize an ONNX file?
Sure! The quickest way to do this is to perform *dynamic* quantization. The
first step is to pre-process the model:
```sh
bash# python -m onnxruntime.quantization.preprocess --input /opt/ai-models/t5-base-text2text.onnx/encoder_model.onnx --output /opt/ai-models/t5-base-text2text.onnx/encoder.pp.onnx
```
This will create a model about the same size as the original:
```sh
bash# du -sh /opt/ai-models/t5-base-text2text.onnx/encoder_model.onnx /opt/ai-models/t5-base-text2text.onnx/encoder.pp.onnx
419M /opt/ai-models/t5-base-text2text.onnx/encoder_model.onnx
419M /opt/ai-models/t5-base-text2text.onnx/encoder.pp.onnx
```
That is important to keep in mind, as you will be doubling the storage size of
the model before you even being to qunatize.

Next, some actual python code is needed. This can be run from a script, or in
ipython:
```python
from onnxruntime.quantization import QuantFormat, QuantType, quantize_dynamic
input_path = "/opt/ai-models/t5-base-text2text.onnx/encoder.pp.onnx"
output_path = "/opt/ai-models/t5-base-text2text.onnx/encoder.q8.onnx"
quantize_dynamic(input_path, output_path, per_channel=False, weight_type=QuantType.QInt8)
```
This will reduce the size of the model by roughly 75%:
```sh
bash du -sh /opt/ai-models/t5-base-text2text.onnx/encoder.q8.onnx "
105M /opt/ai-models/t5-base-text2text.onnx/encoder.q8.onnx "
```
Needless to say, all of this can be tied together in a script which deletes
the pre-processed file on exit. 

The quantized ONNX file can the be loaded per the [https://onnxruntime.ai/docs/api/python/api_summary.html](API specs):
```python
import onnxruntime

model_path = "/opt/ai-models/t5-base-text2text.onnx/encoder.q8.onnx"
session = onnxruntime.InferenceSession(model_path)
outputs = session.run([output names], inputs)
```
