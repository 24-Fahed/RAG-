"""
LLM 生成器模型。

使用语言模型从上下文 + 查询生成最终答案。
默认：LLaMA-3-8B-Instruct（可配置）。
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

_generator_instance: Optional["Generator"] = None


class Generator:
    """基于 LLM 的答案生成器。"""

    def __init__(self, model_path: str = "", max_out_len: int = 50):
        self.model_path = model_path
        self.max_out_len = max_out_len
        self.model = None
        self.tokenizer = None

        if model_path:
            self._load_model(model_path)

    def _load_model(self, model_path: str):
        """加载 LLM 模型。"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
            )
            logger.info(f"Loaded generator model: {model_path}")
            logger.info(
                "Generator model meta model=%s tokenizer_class=%s model_class=%s chat_template=%s bos_token_id=%s eos_token_id=%s pad_token_id=%s generation_config=%s",
                model_path,
                type(self.tokenizer).__name__ if self.tokenizer is not None else None,
                type(self.model).__name__ if self.model is not None else None,
                bool(getattr(self.tokenizer, "chat_template", None)),
                getattr(self.tokenizer, "bos_token_id", None),
                getattr(self.tokenizer, "eos_token_id", None),
                getattr(self.tokenizer, "pad_token_id", None),
                {
                    "do_sample": getattr(getattr(self.model, "generation_config", None), "do_sample", None),
                    "temperature": getattr(getattr(self.model, "generation_config", None), "temperature", None),
                    "top_p": getattr(getattr(self.model, "generation_config", None), "top_p", None),
                    "repetition_penalty": getattr(getattr(self.model, "generation_config", None), "repetition_penalty", None),
                    "eos_token_id": getattr(getattr(self.model, "generation_config", None), "eos_token_id", None),
                    "pad_token_id": getattr(getattr(self.model, "generation_config", None), "pad_token_id", None),
                    "bos_token_id": getattr(getattr(self.model, "generation_config", None), "bos_token_id", None),
                },
            )
        except Exception as e:
            logger.warning(f"Could not load model from {model_path}: {e}")

    def _preview(self, text: str, limit: int = 200) -> str:
        return text.replace("\n", "\\n")[:limit]

    def _preview_token_ids(self, token_ids, limit: int = 16) -> str:
        if token_ids is None:
            return ""
        ids = [int(token_id) for token_id in token_ids[-limit:]]
        return ",".join(str(token_id) for token_id in ids)

    def _preview_topk_tokens(self, token_ids, scores, limit: int = 10) -> str:
        pairs = []
        for token_id, score in zip(token_ids[:limit], scores[:limit]):
            token_text = self.tokenizer.decode([int(token_id)], skip_special_tokens=False)
            pairs.append(f"{int(token_id)}:{float(score):.4f}:{self._preview(token_text, 40)}")
        return " | ".join(pairs)

    def _normalize_generate_output(self, outputs):
        if hasattr(outputs, "sequences") and outputs.sequences is not None:
            return outputs.sequences, {
                "output_type": type(outputs).__name__,
                "has_sequences": True,
                "sequences_shape": tuple(outputs.sequences.shape),
            }
        if hasattr(outputs, "shape"):
            return outputs, {
                "output_type": type(outputs).__name__,
                "has_sequences": False,
                "sequences_shape": tuple(outputs.shape),
            }
        raise TypeError(f"Unsupported generate output type: {type(outputs)!r}")

    def _build_messages(self, query: str, context: str) -> list[dict[str, str]]:
        if context:
            user_content = (
                "Use the retrieved context to answer the question concisely.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {query}"
            )
        else:
            user_content = f"Answer the question concisely.\n\nQuestion: {query}"

        return [
            {
                "role": "system",
                "content": "You are a helpful question answering assistant.",
            },
            {
                "role": "user",
                "content": user_content,
            },
        ]

    def _build_generation_inputs(self, query: str, context: str):
        use_chat_template = bool(
            self.tokenizer is not None
            and hasattr(self.tokenizer, "apply_chat_template")
            and getattr(self.tokenizer, "chat_template", None)
        )

        if use_chat_template:
            import torch

            messages = self._build_messages(query, context)
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            prompt_token_count = len(
                self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                )
            )
            chat_inputs = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                truncation=True,
                max_length=2048,
                return_tensors="pt",
            )
            if isinstance(chat_inputs, torch.Tensor):
                inputs = {
                    "input_ids": chat_inputs,
                    "attention_mask": torch.ones_like(chat_inputs),
                }
            else:
                inputs = chat_inputs
        else:
            if context:
                prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
            else:
                prompt = f"Question: {query}\n\nAnswer:"

            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            prompt_token_count = len(self.tokenizer(prompt, add_special_tokens=True)["input_ids"])

        input_token_count = int(inputs["input_ids"].shape[1])
        truncated = prompt_token_count > input_token_count
        return inputs, prompt, input_token_count, truncated, use_chat_template

    def generate(self, query: str, context: str = "") -> str:
        """
        根据查询和可选上下文生成答案。

        Args:
            query: 用户的问题。
            context: 检索并压缩后的上下文。

        Returns:
            生成的答案字符串。
        """
        if self.model is None:
            # 回退方案：使用简单模板
            if context:
                return f"[Based on retrieved context]: {context[:200]}..."
            return "[No model loaded]"

        inputs, prompt, input_token_count, truncated, used_chat_template = self._build_generation_inputs(
            query,
            context,
        )

        logger.info(
            "Generator input model=%s context_chars=%s query_chars=%s prompt_tokens=%s truncated=%s used_chat_template=%s prompt_preview=%s",
            self.model_path,
            len(context),
            len(query),
            input_token_count,
            truncated,
            used_chat_template,
            self._preview(prompt),
        )

        input_ids = inputs["input_ids"][0]
        logger.info(
            "Generator tokens model=%s bos_token_id=%s eos_token_id=%s pad_token_id=%s generation_eos_token_id=%s input_tail_ids=%s input_tail_preview=%s",
            self.model_path,
            getattr(self.tokenizer, "bos_token_id", None),
            getattr(self.tokenizer, "eos_token_id", None),
            getattr(self.tokenizer, "pad_token_id", None),
            getattr(getattr(self.model, "generation_config", None), "eos_token_id", None),
            self._preview_token_ids(input_ids),
            self._preview(self.tokenizer.decode(input_ids[-32:], skip_special_tokens=False), 240),
        )

        if hasattr(self.model, "device"):
            if hasattr(inputs, "to"):
                inputs = inputs.to(self.model.device)
            else:
                inputs = {key: value.to(self.model.device) for key, value in inputs.items()}

        import torch
        with torch.no_grad():
            forward_outputs = self.model(**inputs)
            next_token_logits = forward_outputs.logits[0, -1]
            next_token_probs = torch.softmax(next_token_logits, dim=-1)
            topk = torch.topk(next_token_probs, k=10)
            logger.info(
                "Generator first-step model=%s topk=%s",
                self.model_path,
                self._preview_topk_tokens(topk.indices.tolist(), topk.values.tolist()),
            )

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_out_len,
                return_dict_in_generate=False,
                use_cache=False,
            )

        sequences, output_meta = self._normalize_generate_output(outputs)
        logger.info(
            "Generator generate-call model=%s meta=%s",
            self.model_path,
            output_meta,
        )

        generated_ids = sequences[0][inputs["input_ids"].shape[1]:]
        generated_token_count = int(generated_ids.shape[0])
        first_token_id = int(generated_ids[0].item()) if generated_token_count > 0 else None
        raw_response = self.tokenizer.decode(generated_ids, skip_special_tokens=False)
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        stripped_response = response.strip()

        logger.info(
            "Generator output model=%s generated_tokens=%s first_token_id=%s raw_preview=%s decoded_preview=%s stripped_len=%s",
            self.model_path,
            generated_token_count,
            first_token_id,
            self._preview(raw_response),
            self._preview(response),
            len(stripped_response),
        )

        if not stripped_response:
            logger.warning(
                "Generator returned empty text model=%s query_preview=%s context_preview=%s",
                self.model_path,
                self._preview(query, 120),
                self._preview(context, 120),
            )

        return stripped_response


def get_generator(model_path: str = "", max_out_len: int = 50) -> Generator:
    """获取或创建单例生成器实例。"""
    global _generator_instance
    if _generator_instance is None:
        _generator_instance = Generator(model_path=model_path, max_out_len=max_out_len)
    return _generator_instance
