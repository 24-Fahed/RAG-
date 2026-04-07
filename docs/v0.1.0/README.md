# v0.1.0 Notes

## Known Issues

### 1. Model compatibility

`llama3-8b-ragga` is an instruction-tuned chat model. After downgrading to `transformers==4.48.2`, the `apply_chat_template(..., return_tensors="pt")` path may return a tensor instead of a `BatchEncoding` object. This can cause compatibility issues if the code assumes `inputs["input_ids"]` is always available.

Reason:
- Different `transformers` versions return different input container types for chat-template generation.
- This release is focused on restoring the main retrieval and inference chain first.

Decision for this version:
- Record the compatibility risk.
- Leave broader compatibility hardening to the next version.

### 2. Empty answer output

`llama3-8b-ragga` may return an empty answer when it is prompted as a plain completion model.

Reason:
- This model is a fine-tuned instruct/chat model, not a plain causal completion model.
- A prompt like `Context ... Question ... Answer:` can lead to `generated_tokens=0`, even when retrieval, reranking, and compression all succeed.

Decision for this version:
- Record the issue and keep diagnostic logs.
- Move full prompt-format refinement and output stabilization to the next version.

## Why keep these items for the next version

This version is intended to stabilize the deployment chain first:
- server to inference connectivity
- model preload readiness
- indexing
- retrieval
- reranking
- compression

Generator prompt refinement and broader cross-version compatibility cleanup will be handled in the next release after the current chain is stable and reproducible.
