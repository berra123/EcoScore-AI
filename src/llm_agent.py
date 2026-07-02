"""
LLM Green Recommendation Agent
==============================

This module provides AI-powered sustainability recommendations using
Google Gemini API.

Responsibilities
----------------
- Read API key from .env or Streamlit Secrets
- Send production parameters to Gemini
- Generate practical Green Manufacturing recommendations
- Return Markdown formatted output
- Handle API failures gracefully

Author:
EcoScore AI
"""

from __future__ import annotations

import os
from typing import Dict, Any

from dotenv import load_dotenv

# Load .env if available
load_dotenv()


# =============================================================================
# API KEY
# =============================================================================
def _get_api_key() -> str:
    """
    Reads Gemini API Key.

    Priority:
    1. Streamlit Secrets
    2. Environment Variables (.env)

    Returns
    -------
    str
        Gemini API Key

    Raises
    ------
    RuntimeError
        If API key cannot be found.
    """

    # ---------------------------------------------------------
    # Streamlit Secrets
    # ---------------------------------------------------------
    try:
        import streamlit as st

        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass

    # ---------------------------------------------------------
    # Environment Variable
    # ---------------------------------------------------------
    api_key = os.getenv("GEMINI_API_KEY")

    if api_key:
        return api_key

    raise RuntimeError(
        "GEMINI_API_KEY bulunamadı. "
        ".env dosyasına veya Streamlit Secrets'a ekleyiniz."
    )


# =============================================================================
# Prompt Builders
# =============================================================================
SYSTEM_PROMPT = """
Sen Dentaş Kağıt Sanayi'nde çalışan uzman bir
Sürdürülebilir Üretim ve Enerji Optimizasyonu Yapay Zekâ Asistanısın.

Görevin;

• Verilen üretim parametrelerini incelemek
• Karbon ayak izini azaltmak
• Enerji tüketimini düşürmek
• Üretim kalitesini korumak

ve mühendislere Türkçe dilinde uygulanabilir,
gerçekçi ve pratik yeşil üretim önerileri sunmaktır.

Kurallar:

- Kısa yaz.
- Gereksiz açıklama yapma.
- Markdown kullan.
- En fazla 6 madde yaz.
- Her öneri aksiyona dönüştürülebilir olsun.
- Sayısal parametreleri dikkate al.
- EcoScore'u yorumla.
- Karbon salımını azaltmaya odaklan.
- Enerji verimliliğini artırmaya odaklan.

Öneriler aşağıdaki başlıklardan uygun olanları içerebilir:

- Nem optimizasyonu
- Fırın sıcaklığı optimizasyonu
- Hat hızı optimizasyonu
- Enerji tasarrufu
- Doğalgaz tüketimi
- Elektrik tüketimi
- Atık azaltımı
- Operasyonel verimlilik

Yanıt tamamen Türkçe olmalıdır.
"""


def _build_user_prompt(
    siparis_detaylari: Dict[str, Any],
    mevcut_co2_kg: float,
    ecoscore_harfi: str,
) -> str:
    """
    Builds user prompt.
    """

    return f"""
Aşağıdaki üretim siparişini analiz et.

## Sipariş Bilgileri

{siparis_detaylari}

## Hesaplanan CO2

{mevcut_co2_kg:.2f} kg CO₂e

## EcoScore

{ecoscore_harfi}

Lütfen;

- üretim parametrelerini analiz et,
- karbon emisyonunu azalt,
- enerji tüketimini azalt,
- uygulanabilir yeşil reçete önerileri oluştur.

Yanıt formatı:

# 🌱 Yeşil Üretim Önerileri

- ...
- ...
- ...
"""


# =============================================================================
# Public API
# =============================================================================
def generate_green_recommendation(
    siparis_detaylari: Dict[str, Any],
    mevcut_co2_kg: float,
    ecoscore_harfi: str,
) -> str:
    """
    Generates AI-powered green manufacturing recommendations.

    Parameters
    ----------
    siparis_detaylari : dict
        Production parameters.

    mevcut_co2_kg : float
        Calculated carbon footprint.

    ecoscore_harfi : str
        EcoScore grade.

    Returns
    -------
    str
        Markdown formatted recommendation.
    """

    try:
        import google.generativeai as genai

    except ImportError:
        return (
            "❌ Google Generative AI kütüphanesi bulunamadı.\n\n"
            "Kurulum:\n"
            "```bash\n"
            "pip install google-generativeai\n"
            "```"
        )

    try:
        api_key = _get_api_key()

        genai.configure(api_key=api_key)

        model = genai.GenerativeModel(
    model_name="gemini-2.5-flash",
    system_instruction=SYSTEM_PROMPT,
)

        response = model.generate_content(
            _build_user_prompt(
                siparis_detaylari,
                mevcut_co2_kg,
                ecoscore_harfi,
            )
        )

        if (
            response is not None
            and hasattr(response, "text")
            and response.text
        ):
            return response.text.strip()

        return (
            "⚠️ Yapay zekâ anlamlı bir öneri üretemedi. "
            "Lütfen üretim parametrelerini kontrol edip tekrar deneyiniz."
        )

    except RuntimeError as exc:
        return f"❌ Yapılandırma Hatası\n\n{exc}"

    except Exception as exc:
        return (
            "❌ Yeşil Karar Destek Ajanı çalıştırılırken beklenmeyen "
            "bir hata oluştu.\n\n"
            f"**Hata Detayı:** {str(exc)}"
        )