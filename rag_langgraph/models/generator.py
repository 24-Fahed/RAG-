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

        if context:
            prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        else:
            prompt = f"Question: {query}\n\nAnswer:"

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        if hasattr(self.model, "device"):
            inputs = inputs.to(self.model.device)

        import torch
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_out_len,
                do_sample=False,
                temperature=1.0,
            )

        response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return response.strip()


def get_generator(model_path: str = "", max_out_len: int = 50) -> Generator:
    """获取或创建单例生成器实例。"""
    global _generator_instance
    if _generator_instance is None:
        _generator_instance = Generator(model_path=model_path, max_out_len=max_out_len)
    return _generator_instance
