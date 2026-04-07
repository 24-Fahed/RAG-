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
        except Exception as e:
            logger.warning(f"Could not load model from {model_path}: {e}")

    def _preview(self, text: str, limit: int = 200) -> str:
        return text.replace("\n", "\\n")[:limit]

    def _preview_token_ids(self, token_ids, limit: int = 16) -> str:
        if token_ids is None:
            return ""
        ids = [int(token_id) for token_id in token_ids[-limit:]]
        return ",".join(str(token_id) for token_id in ids)

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
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_out_len,
                do_sample=False,
                temperature=1.0,
            )

        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
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
