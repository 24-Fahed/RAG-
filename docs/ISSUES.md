# 已知问题

## 1. transformers torch.load 安全检查阻止模型加载

**状态**：未解决

**现象**：Inference Worker 加载 rerank 模型（`castorini/monot5-base-msmarco-10k`）时报错：

```
ValueError: Due to a serious vulnerability issue in `torch.load`, even with `weights_only=True`,
we now require users to upgrade torch to at least v2.6
```

**原因**：`transformers >= 4.51` 在 `modeling_utils.load_state_dict()` 中调用 `check_torch_load_is_safe()`，检查 torch 版本。AutoDL 的 torch 版本低于 2.6，触发该检查。

**调用链**：
```
rerank.py -> get_reranker() -> MonoT5Reranker.__init__()
  -> T5ForConditionalGeneration.from_pretrained()
    -> modeling_utils.load_state_dict()
      -> check_torch_load_is_safe()  ← 此处抛出异常
```

**已尝试的方案**（均失败）：
- 在 `inference/main.py` 中 monkey-patch `transformers.utils.import_utils.check_torch_load_is_safe`
  - 失败原因：`import transformers.utils.import_utils` 触发 transformers 全量初始化，但此时 `modeling_utils` 尚未被导入；待 rerank 路由首次被调用时，`modeling_utils` 才被导入并绑定原始函数引用
- 遍历 `sys.modules` 修补所有已加载模块
  - 失败原因：同上，`modeling_utils` 在修补时还未出现在 `sys.modules` 中

**可行的解决方案**（任选其一）：
1. **降级 transformers**：`pip install transformers==4.48.0`（不含此安全检查）
2. **升级 torch**：`pip install torch>=2.6`（满足安全检查要求）
3. **正确的 monkey-patch**：在 `rag_langgraph/models/rerankers.py` 中，在 `from transformers import ...` 之前插入 patch，或在 `transformers/modeling_utils.py` 源码中直接注释掉 `check_torch_load_is_safe()` 调用
4. **使用 safetensors 格式的模型**：错误信息提示 safetensors 格式不受此限制

**影响范围**：所有通过 `torch.load` 加载的 `.bin` 格式模型（rerank、compress 等），不影响 safetensors 格式的模型（embed、classify 已正常工作）

---

## 2. 分类器使用未微调的 BERT 基础模型

**状态**：已临时绕过

**现象**：分类器对所有查询返回 label 0（不需要检索），导致检索结果为空。

**原因**：`rag_langgraph/models/classifier.py` 加载 `google-bert/bert-base-multilingual-cased`，该模型没有经过检索必要性判断的微调。分类头权重为随机初始化（日志中 `classifier.weight | MISSING`）。

**当前处理**：在 `server/services/pipeline.py` 中注释掉分类步骤，强制 `label = 1`（所有查询走检索）。

**正确修复**：获取微调后的分类器权重文件，配置 `weights_path` 参数。

---

## 3. monkey-patch 时序问题

**状态**：未解决

**现象**：在 `inference/main.py` 中对 transformers 的 monkey-patch 对部分模块无效。

**原因**：Python 模块导入机制。`from transformers.utils.import_utils import check_torch_load_is_safe` 会在导入模块时绑定函数的本地引用。后续修改源模块的属性不会影响已绑定的引用。

正确的修补必须在目标模块（如 `modeling_utils`）导入之前完成，但 `import transformers.utils.import_utils` 会触发 `transformers.__init__` 的执行，导致其他子模块在 patch 之前就已经完成了导入和绑定。

---

## 4. Shell 脚本中文输出在 Linux 终端乱码

**状态**：已解决

**原因**：Windows 环境下编辑的 `.sh` 文件包含中文 echo 输出，通过 scp 传到 Linux 后编码不一致导致乱码。

**修复**：所有 `.sh` 脚本的 echo 输出改为英文，注释保留中文。
