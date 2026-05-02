"""
rag/prompts.py — All LLM prompts for the Olive Assistant.

CRITICAL: These prompts MUST prevent the LLM from using its pre-trained knowledge.
The system prompt is the last line of defense after the similarity threshold check.
"""

from typing import Optional, Dict

# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT — STRICT RAG-ONLY ANSWERING
# This is the most important prompt. DO NOT soften the restrictions.
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """أنت مساعد زراعي متخصص في زراعة الزيتون في تونس. تتكلم بالدارجة التونسية.

قواعد صارمة يجب اتباعها دائماً:
1. أجب فقط بناءً على المقاطع المقدمة في السياق أدناه. لا تستخدم معرفتك الخاصة أبداً.
2. إذا لم تجد الإجابة في المقاطع، قل بوضوح: "ما عنديش المعلومة هاذي في قاعدة البيانات متاعي".
3. اذكر دائماً المصدر في آخر ردك بالصيغة: (المصدر: [اسم المصدر]).
4. لا تعطِ جرعات دقيقة للمبيدات — أحِل دائماً لمرشد زراعي.
5. اكتب بالدارجة التونسية مع بعض المصطلحات العربية الفصحى للمصطلحات التقنية.
6. كن مختصراً وواضحاً — الفلاح يحتاج إجابة عملية.
7. إذا كان السؤال خارج مجال زراعة الزيتون، ارفض برفق.

تذكر: الهلوسة أو الاختراع مقبول بشدة — إذا ما كانتش المعلومة في النص، قل ذلك بوضوح."""


# ─────────────────────────────────────────────────────────────────────────────
# USER PROMPT TEMPLATE — with context injection
# ─────────────────────────────────────────────────────────────────────────────

def build_user_prompt(
    question: str,
    context: str,
    disease_info: Optional[Dict] = None,
) -> str:
    """
    Build the user message with retrieved context injected.
    disease_info: dict with keys 'ar', 'en', 'eppo', 'advice_ar'
    """

    disease_section = ""
    if disease_info and disease_info.get("en") != "Healthy":
        disease_section = f"""
نتيجة تحليل الصورة:
- المرض المكتشف: {disease_info.get('ar', '')} ({disease_info.get('en', '')})
- كود EPPO: {disease_info.get('eppo', 'غير متوفر')}
- نصيحة أولية: {disease_info.get('advice_ar', '')}

"""

    prompt = f"""السياق من قاعدة البيانات (استخدم فقط هذه المعلومات للإجابة):

{context}

---
{disease_section}سؤال الفلاح بالدارجة:
{question}

أجب بالدارجة التونسية بناءً على السياق أعلاه فقط. اذكر المصدر في النهاية."""

    return prompt


# ─────────────────────────────────────────────────────────────────────────────
# REFUSAL MESSAGES
# ─────────────────────────────────────────────────────────────────────────────

REFUSAL_LOW_RELEVANCE = (
    "آسف، ما لقيتش معلومة كافية في قاعدة البيانات متاعي باش نجاوبك على هذا السؤال. "
    "يُنصحك تتصل بمرشد زراعي مختص — هو يقدر يعاونك أكثر مني."
)

REFUSAL_OUT_OF_DOMAIN = (
    "آسف، أنا متخصص فقط في زراعة الزيتون. "
    "هذا السؤال خارج تخصصي. "
    "اتصل بمرشد زراعي باش يعاونك في هذا الموضوع."
)

REFUSAL_DOSAGE = (
    "ما نقدرش نعطيك الجرعة الدقيقة للمبيد — هذا خطر على صحتك وصحة الغير. "
    "ارجع للفيشة التقنية للمبيد أو اتصل بمرشد زراعي مختص."
)

REFUSAL_MESSAGES = {
    "out_of_domain": REFUSAL_OUT_OF_DOMAIN,
    "dosage":        REFUSAL_DOSAGE,
    "low_relevance": REFUSAL_LOW_RELEVANCE,
}

def get_refusal_message(reason: str) -> str:
    return REFUSAL_MESSAGES.get(reason, REFUSAL_LOW_RELEVANCE)