"""LLM narrative interpretation for regression diagnostics."""

from __future__ import annotations

import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-haiku-4-5-20251001",
    "gemini": "gemini-3.1-pro-preview",
}

SYSTEM_PROMPT = """You are a senior quantitative researcher who specialises in regression analysis for behavioural and product data.

Given a JSON summary of OLS regression diagnostics, write a clear 3-4 sentence narrative for a researcher or analyst who is not a statistician. Cover:
1. The overall model fit (R²)
2. Which assumptions were violated and what that means practically
3. Whether multicollinearity is a concern
4. One concrete recommendation (e.g. transform a variable, collect more data, check influential points)

Use plain language. Do not use the phrase "it is important to note"."""


class RegressionNarrator:
    def __init__(self, backend: str, model: Optional[str] = None, api_key: Optional[str] = None) -> None:
        if backend not in DEFAULT_MODELS:
            raise ValueError(f"Unknown backend '{backend}'. Choose from: {list(DEFAULT_MODELS)}")
        self.backend = backend
        self.model = model or DEFAULT_MODELS[backend]
        self.api_key = api_key

    def narrate(self, result_dict: dict) -> str:
        prompt = f"Interpret these OLS regression diagnostics:\n\n{json.dumps(result_dict, indent=2)}"
        try:
            if self.backend == "openai":
                return self._call_openai(prompt)
            elif self.backend == "anthropic":
                return self._call_anthropic(prompt)
            else:
                return self._call_gemini(prompt)
        except Exception as exc:
            logger.warning("LLM narration failed: %s", exc)
            return ""

    def _call_openai(self, prompt: str) -> str:
        try:
            import openai
        except ImportError as exc:
            raise ImportError("pip install 'regression-diagnostics[openai]'") from exc
        kwargs = {"api_key": self.api_key} if self.api_key else {}
        client = openai.OpenAI(**kwargs)
        resp = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            temperature=0.3, max_tokens=300,
        )
        return resp.choices[0].message.content or ""

    def _call_anthropic(self, prompt: str) -> str:
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError("pip install 'regression-diagnostics[anthropic]'") from exc
        kwargs = {"api_key": self.api_key} if self.api_key else {}
        client = anthropic.Anthropic(**kwargs)
        resp = client.messages.create(
            model=self.model, max_tokens=300, system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}], temperature=0.3,
        )
        return resp.content[0].text if resp.content else ""

    def _call_gemini(self, prompt: str) -> str:
        try:
            import google.generativeai as genai
        except ImportError as exc:
            raise ImportError("pip install 'regression-diagnostics[gemini]'") from exc
        if self.api_key:
            genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel(model_name=self.model, system_instruction=SYSTEM_PROMPT)
        resp = model.generate_content(prompt, generation_config={"temperature": 0.3, "max_output_tokens": 300})
        return resp.text or ""
