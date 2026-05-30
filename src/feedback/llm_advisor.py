"""LLM 健身教练建议生成 — 使用 Qwen2-7B 生成中文矫正建议.

支持两种后端:
  - transformers: 本地加载 HuggingFace 模型（需要 GPU 显存）
  - ollama: 通过 Ollama HTTP API 调用（需先启动 ollama serve）
"""

import json
from dataclasses import dataclass, field
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class AdviceContext:
    """LLM 输入上下文."""
    movement: str = ""
    movement_name: str = ""
    phase: str = ""
    score: float = 0.0
    errors: list = field(default_factory=list)
    rep_count: int = 0
    duration: float = 0.0


class LLMAdvisor:
    """基于 Qwen2-7B 的健身建议生成器.

    输入: 错误列表、关节角度偏差、当前阶段
    输出: 简短中文纠正建议

    使用方式:
        advisor = LLMAdvisor(model_path="Qwen/Qwen2-7B-Instruct")
        advisor.load()
        advice = advisor.generate(context)
    """

    SYSTEM_PROMPT = """你是一名拥有10年经验的专业健身教练。你的任务是根据用户的动作数据给出简短、精准的纠正建议。

规则：
1. 每次只给出1-2条建议，每条不超过20字
2. 使用简洁直接的指令式语言，如"膝盖向外打开"
3. 优先指出最严重的问题
4. 使用中文
5. 不要输出问候语、解释或其他无关内容"""

    def __init__(
        self,
        model_path: str = "Qwen/Qwen2-7B-Instruct",
        device: str = "cuda",
        use_4bit: bool = True,
        use_ollama: bool = False,
        ollama_host: str = "http://localhost:11434",
    ):
        self.model_path = model_path
        self.device = device
        self.use_4bit = use_4bit
        self.use_ollama = use_ollama
        self.ollama_host = ollama_host
        self._model = None
        self._tokenizer = None
        self._loaded = False

    def load(self) -> bool:
        """加载模型."""
        if self.use_ollama:
            return self._load_ollama()
        return self._load_transformers()

    def _load_ollama(self) -> bool:
        """检查 Ollama 服务是否可用."""
        try:
            import urllib.request
            req = urllib.request.Request(f"{self.ollama_host}/api/tags")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
            models = [m["name"] for m in data.get("models", [])]
            logger.info(f"[LLMAdvisor] Ollama 可用，已有模型: {models}")
            self._loaded = True
            return True
        except Exception as e:
            logger.error(f"[LLMAdvisor] Ollama 连接失败 ({self.ollama_host}): {e}")
            return False

    def _load_transformers(self) -> bool:
        """加载 HuggingFace transformers 模型."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

            if self.use_4bit:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype="float16",
                    bnb_4bit_use_double_quant=True,
                )
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True,
                )
            else:
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype="auto",
                    device_map="auto",
                    trust_remote_code=True,
                )

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=True
            )
            self._loaded = True
            logger.info(f"[LLMAdvisor] 模型已加载: {self.model_path}")
            return True

        except ImportError:
            logger.warning("[LLMAdvisor] transformers 未安装，使用规则模式")
            return False
        except Exception as e:
            logger.error(f"[LLMAdvisor] 模型加载失败: {e}")
            return False

    def generate(self, context: AdviceContext) -> str:
        """根据运动上下文生成矫正建议.

        Args:
            context: 动作上下文（动作名、阶段、评分、错误列表）

        Returns:
            中文矫正建议字符串
        """
        if not self._loaded:
            return self._generate_rule_based(context)
        if self.use_ollama:
            return self._generate_ollama(context)
        elif self._model is not None:
            return self._generate_llm(context)
        else:
            return self._generate_rule_based(context)

    def _generate_ollama(self, context: AdviceContext) -> str:
        """通过 Ollama API 生成建议."""
        error_text = ", ".join(
            f"{e.name}({e.severity})" for e in context.errors[:3]
        ) if context.errors else "无明显错误"

        prompt = f"""{self.SYSTEM_PROMPT}

当前动作: {context.movement_name}
动作阶段: {context.phase}
综合评分: {context.score:.0f}/100
检测到的问题: {error_text}

请给出纠正建议："""

        try:
            import urllib.request
            body = json.dumps({
                "model": self.model_path,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "num_predict": 80,
                },
            }).encode("utf-8")
            req = urllib.request.Request(
                f"{self.ollama_host}/api/generate",
                data=body,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
            return data.get("response", "").strip()
        except Exception as e:
            logger.error(f"[LLMAdvisor] Ollama 生成失败: {e}")
            return self._generate_rule_based(context)

    def _generate_llm(self, context: AdviceContext) -> str:
        """使用 LLM 生成建议."""
        error_text = ", ".join(
            f"{e.name}({e.severity})" for e in context.errors[:3]
        ) if context.errors else "无明显错误"

        prompt = f"""{self.SYSTEM_PROMPT}

当前动作: {context.movement_name}
动作阶段: {context.phase}
综合评分: {context.score:.0f}/100
检测到的问题: {error_text}

请给出纠正建议："""

        try:
            messages = [
                {"role": "user", "content": prompt},
            ]
            text = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self._tokenizer([text], return_tensors="pt").to(self.device)
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=80,
                temperature=0.3,
                do_sample=True,
                pad_token_id=self._tokenizer.eos_token_id,
            )
            response = self._tokenizer.decode(
                outputs[0][len(inputs.input_ids[0]):],
                skip_special_tokens=True,
            )
            return response.strip()

        except Exception as e:
            logger.error(f"[LLMAdvisor] 生成失败: {e}")
            return self._generate_rule_based(context)

    def _generate_rule_based(self, context: AdviceContext) -> str:
        """基于规则的建议生成（LLM不可用时的回退方案）."""
        if not context.errors:
            if context.score >= 90:
                return "动作标准，继续保持！"
            elif context.score >= 75:
                return "动作基本标准，注意细节控制。"
            else:
                return "注意动作规范，放慢速度感受发力。"

        # 取最严重的错误
        severity_order = {"high": 0, "medium": 1, "low": 2}
        sorted_errors = sorted(
            context.errors,
            key=lambda e: severity_order.get(e.severity, 3)
        )

        # 最多返回2条
        parts = []
        for e in sorted_errors[:2]:
            if e.advice:
                parts.append(e.advice)

        if not parts:
            return "请注意动作姿势，保持核心收紧。"

        return "；".join(parts)

    def generate_rep_feedback(self, context: AdviceContext) -> dict:
        """一次完整动作的反馈.

        Returns:
            {summary, key_improvement, correction, score_breakdown}
        """
        advice = self.generate(context)

        # 评分等级
        if context.score >= 90:
            level = "优秀"
        elif context.score >= 75:
            level = "良好"
        elif context.score >= 60:
            level = "需要注意"
        else:
            level = "需要改进"

        return {
            "level": level,
            "score": f"{context.score:.0f}/100",
            "advice": advice,
            "errors": [
                {"name": e.name, "severity": e.severity, "advice": e.advice}
                for e in context.errors[:3]
            ],
            "summary": f"{context.movement_name} · {level} · {context.score:.0f}分",
        }
