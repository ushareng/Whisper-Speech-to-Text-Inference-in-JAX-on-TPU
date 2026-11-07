# Whisper Speech-to-Text Inference in JAX on TPU

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ushareng/Whisper-Speech-to-Text-Inference-in-JAX-on-TPU/blob/main/whisper_jax_tpu_inference.ipynb)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

A from-scratch **OpenAI Whisper** inference pipeline in **pure JAX**, built for Cloud TPU. It loads the official Whisper checkpoints from the Hugging Face Hub and runs the whole pipeline on the TPU:

- log-mel spectrogram
- encoder
- KV-cached greedy decoder

On a single **TPU v5e** chip, `whisper-large-v3-turbo` transcribes **8 minutes of audio in 1.8 seconds** with **3.2% WER**, and reaches up to **655× real time** at batch size 64.

No Flax, no PyTorch: `jax`, `numpy`, `safetensors`, and `transformers` (for the tokenizer and reference feature extractor only).

---

## Why not `whisper-jax`?

The popular [`whisper-jax`](https://github.com/sanchit-gandhi/whisper-jax) package is built on the Flax Whisper model in `transformers`, which `transformers` v5 removed. This notebook implements the model directly in JAX, so it works with current JAX and `transformers` releases.

## Highlights

- **Any Whisper checkpoint:** `tiny` → `large-v3`, `large-v3-turbo`, and English-only `*.en` models. Mel bins, vocab size, special tokens and layer counts are all read from the checkpoint config.
- **Log-mel spectrogram on the TPU:** STFT + mel filterbank in `jnp` under `jit`. It matches the Hugging Face feature extractor to within 2e-5.
- **Fast compilation:** per-layer weights are stacked and the encoder and decoder run as a single `lax.scan`, so compile time doesn't grow with depth.
- **KV-cached decoding in one compiled loop:**
  - the whole greedy generation runs inside one jitted `lax.while_loop`
  - cross-attention K/V are computed once per utterance
  - generation stops early when every sequence in the batch has finished
- **Language detection**, **transcribe / translate** tasks, and token suppression matching OpenAI's generation config.
- **Batch split across all TPU cores** with `jax.sharding` (`NamedSharding(mesh, P("data"))`).
- **bfloat16** weights and matmuls. LayerNorm, softmax and logits stay in float32.
- **Long-form audio:** split at quiet points into ≤30 s chunks and transcribe them in one batch.
- **Built-in evaluation and benchmark:** WER on LibriSpeech, and encoder/decoder timing and real-time factor vs batch size.

## Pipeline

```
audio (16 kHz)
  │  pad / trim to 30 s
  ▼
log-mel spectrogram  [B, 3000, n_mels]       ← jnp STFT + mel filterbank (jit)
  ▼
encoder: conv → conv(stride 2) → N × transformer → LayerNorm   [B, 1500, D]   (lax.scan)
  ▼
cross-attention K/V for all decoder layers, computed once        [L, B, 1500, H, Dh]
  ▼
greedy decode in lax.while_loop  (self-attn KV cache [L, B, T, H, Dh])
  ▼
token ids → tokenizer.decode → text
```

## Quick start

1. Click **Open in Colab** above.
2. **Runtime → Change runtime type → TPU** (v5e-1, v6e-1 or v2-8).
3. **Runtime → Run all**.

To transcribe your own file, set `TRY_UPLOAD = True` in the upload cell, or call the functions directly:

```python
transcribe([audio_16khz])                      # ≤30 s clips, auto language detection
transcribe([audio], language="en")             # force a language
transcribe([audio], task="translate")          # speech → English text
text, n_chunks = transcribe_long(long_audio)   # any length
```

Pick the checkpoint with `MODEL_ID`. The default is `openai/whisper-large-v3-turbo` on TPU and `openai/whisper-tiny` on CPU.

| model | params | notes |
|---|---|---|
| `openai/whisper-tiny` / `base` / `small` | 39M / 74M / 244M | fast, good for experimenting |
| `openai/whisper-medium` | 769M | |
| `openai/whisper-large-v3` | 1.55B | best accuracy, 128 mel bins |
| `openai/whisper-large-v3-turbo` | 809M | large-v3 encoder + 4-layer decoder (not trained for translation) |
| `*.en` | | English-only |

## Results (TPU v5e-1, `whisper-large-v3-turbo`, bfloat16, JAX 0.11.2)

### Accuracy

| check | result |
|---|---|
| log-mel vs Hugging Face feature extractor | max abs diff **1.9e-5** |
| WER, 73 LibriSpeech utterances (8.0 min), Whisper English normalizer | **3.17%** |

```
reference : MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL
whisper   : Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel.
```

### Long-form

8.0 min of audio → 18 chunks → transcribed in **1.8 s** (**271× real time**, including any compilation).

### Throughput vs batch size

Each item is one ≤30 s chunk of real speech. The first call at each batch size compiles and is excluded from the timings.

| batch | encode (ms) | decode (ms) | decode steps | total (ms) | × real time |
|---:|---:|---:|---:|---:|---:|
| 1 | 35.2 | 52.9 | 90 | 88.1 | 311× |
| 8 | 285.3 | 138.8 | 117 | 424.1 | 522× |
| 16 | 542.8 | 251.8 | 117 | 794.6 | 558× |
| 32 | 1047.0 | 453.1 | 117 | 1500.1 | 579× |
| 64 | 1640.2 | 983.0 | 117 | 2623.2 | **655×** |

The encoder is a single large, MXU-friendly pass. Decoding is memory-bound and runs one step per token, so batching is the main lever for throughput.

## Notebook outline

1. **Choose a model**: read config and generation config.
2. **Load weights**: convert PyTorch layouts, stack layers, cast to bf16, replicate across devices.
3. **Log-mel spectrogram on TPU**: checked against the Hugging Face extractor.
4. **Model**: encoder, cross-attention K/V projection, KV-cached decoder step.
5. **Prompts, token suppression and greedy decoding**: language detection included.
6. **`transcribe()`**: batched and sharded end-to-end inference.
7. **WER on LibriSpeech**.
8. **Long-form audio**: silence-aware chunking.
9. **Language detection, translation, and your own audio**.
10. **Benchmark**: real-time factor vs batch size.

## Requirements

- Colab TPU runtime (JAX preinstalled), or any machine with JAX ≥ 0.4.30
- `transformers`, `safetensors`, `huggingface_hub`, `soundfile`, `pyarrow` (installed by the first cell)
- `librosa`, only for the optional upload cell (preinstalled on Colab)

It also runs on CPU (slowly) for testing. Set `JAX_PLATFORMS=cpu` and `XLA_FLAGS=--xla_force_host_platform_device_count=8` to try the multi-device sharding.

## Limitations and next steps

- **Greedy decoding only.** Beam search and temperature fallback can replace the `argmax` step without changing the KV-cache layout.
- **No timestamps.** OpenAI-style sequential long-form needs the timestamp tokens and their pairing rules.
- **Data parallel only.** For very large checkpoints on multi-chip slices, add a `model` mesh axis and shard the attention/MLP weights.
- **Speculative decoding** with `whisper-tiny` as a draft model could speed up decoding further.

## References

- Radford et al., [*Robust Speech Recognition via Large-Scale Weak Supervision*](https://arxiv.org/abs/2212.04356), 2022.
- [openai/whisper](https://github.com/openai/whisper) and [Whisper checkpoints on Hugging Face](https://huggingface.co/openai).
- JAX docs: [Distributed arrays and automatic parallelization](https://docs.jax.dev/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html).

## License

[Apache 2.0](LICENSE). Whisper model weights are released by OpenAI under the MIT License.
