"""TTS 语音播报 — 使用 edge-tts 将文字转为语音."""

import asyncio
import threading
import tempfile
import os
import platform
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class TTSEngine:
    """文字转语音引擎.

    支持多种后端:
    - edge-tts: Microsoft Edge TTS (免费, 需要网络)
    - pyttsx3: 离线TTS (Windows SAPI5 / Linux espeak)
    """

    def __init__(
        self,
        engine: str = "edge-tts",
        voice: str = "zh-CN-XiaoxiaoNeural",
        rate: str = "+10%",
    ):
        self.engine_type = engine
        self.voice = voice
        self.rate = rate
        self._enabled = True
        self._speaking = False
        self._lock = threading.Lock()

        self._init_engine()

    def _init_engine(self):
        """初始化TTS引擎."""
        if self.engine_type == "pyttsx3":
            try:
                import pyttsx3
                self._pyttsx = pyttsx3.init()
                voices = self._pyttsx.getProperty("voices")
                # 尝试选中文语音
                for v in voices:
                    if "zh" in v.id.lower() or "chinese" in v.name.lower():
                        self._pyttsx.setProperty("voice", v.id)
                        break
                self._pyttsx.setProperty("rate", 180)
                logger.info("[TTS] pyttsx3 已初始化")
            except ImportError:
                logger.warning("[TTS] pyttsx3 未安装，回退到 edge-tts")
                self.engine_type = "edge-tts"
            except Exception as e:
                logger.warning(f"[TTS] pyttsx3 初始化失败: {e}")
                self.engine_type = "edge-tts"

    def speak(self, text: str, block: bool = False):
        """播报文本.

        Args:
            text: 要朗读的文本
            block: True=阻塞等待播完, False=异步播报
        """
        if not self._enabled or not text.strip():
            return

        if block:
            self._speak_sync(text)
        else:
            threading.Thread(target=self._speak_sync, args=(text,), daemon=True).start()

    def _speak_sync(self, text: str):
        """同步播报."""
        with self._lock:
            self._speaking = True
            try:
                if self.engine_type == "edge-tts":
                    asyncio.run(self._edge_tts_speak(text))
                elif self.engine_type == "pyttsx3":
                    self._pyttsx.say(text)
                    self._pyttsx.runAndWait()
            except Exception as e:
                logger.error(f"[TTS] 播报失败: {e}")
            finally:
                self._speaking = False

    async def _edge_tts_speak(self, text: str):
        """使用 edge-tts 播报."""
        try:
            import edge_tts

            communicate = edge_tts.Communicate(
                text,
                self.voice,
                rate=self.rate,
            )

            # 写入临时文件然后播放
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                tmp_path = f.name

            await communicate.save(tmp_path)
            self._play_audio(tmp_path)

            # 清理临时文件
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

        except ImportError:
            logger.warning("[TTS] edge-tts 未安装")
        except Exception as e:
            logger.error(f"[TTS] edge-tts 播报失败: {e}")

    def _play_audio(self, filepath: str):
        """使用系统默认播放器播放音频."""
        system = platform.system()
        try:
            if system == "Windows":
                import winsound
                winsound.PlaySound(filepath, winsound.SND_FILENAME)
            elif system == "Linux":
                # 尝试多种播放器
                for player in ["ffplay", "aplay", "mpg123", "paplay"]:
                    if os.system(f"which {player} > /dev/null 2>&1") == 0:
                        os.system(f'{player} "{filepath}" > /dev/null 2>&1')
                        break
            else:
                logger.warning(f"[TTS] 不支持的平台: {system}")
        except Exception as e:
            logger.error(f"[TTS] 音频播放失败: {e}")

    def stop(self):
        """停止当前播报."""
        if self.engine_type == "pyttsx3":
            try:
                self._pyttsx.stop()
            except Exception:
                pass
        self._speaking = False

    def set_enabled(self, enabled: bool):
        self._enabled = enabled

    @property
    def is_speaking(self) -> bool:
        return self._speaking


# ─── 便捷函数 ─────────────────────────────────────────────────

_global_tts: Optional[TTSEngine] = None


def speak(text: str):
    """全局TTS播报快捷函数."""
    global _global_tts
    if _global_tts is None:
        _global_tts = TTSEngine()
    _global_tts.speak(text)
