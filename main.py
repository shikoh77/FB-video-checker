import os
import json
import re

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import anthropic
import yt_dlp

app = FastAPI(title="FB Video Credibility Checker")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

ANALYSIS_MODEL = "claude-sonnet-4-6"


class AnalyzeRequest(BaseModel):
    url: str


def extract_video_metadata(url: str) -> dict:
    """Pull public metadata about the video without downloading the full file."""
    ydl_opts = {
        "quiet": True,
        "skip_download": True,
        "no_warnings": True,
        "extract_flat": False,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
        return {
            "title": info.get("title"),
            "description": info.get("description"),
            "uploader": info.get("uploader"),
            "uploader_url": info.get("uploader_url"),
            "upload_date": info.get("upload_date"),
            "duration": info.get("duration"),
            "view_count": info.get("view_count"),
            "webpage_url": info.get("webpage_url") or url,
        }
    except Exception as e:
        return {"error": str(e)}


ANALYSIS_PROMPT = """אתה כלי מקצועי לבדיקת אמינות סרטונים ברשתות חברתיות.
קיבלת את הנתונים הבאים על סרטון מפייסבוק:

כותרת: {title}
תיאור/טקסט מצורף: {description}
מעלה הסרטון: {uploader}
תאריך העלאה (אם ידוע): {upload_date}
קישור מקור: {webpage_url}

המשימה שלך:
1. חפש באינטרנט אם הסרטון, הכותרת, או הטענות שבו הופיעו באתרי בדיקת עובדות (כגון AFP Fact Check, Reuters Fact Check, "העובדה", FactCheck.org וכו').
2. חפש אם הסרטון או תמונות ממנו מוכרים כסרטון ישן/ממקור אחר שמשתמשים בו מחדש בהקשר שונה ("recycled video").
3. בדוק האם יש כתבות חדשות אמינות שמאששות או מכחישות את מה שמתואר בסרטון.
4. הערך את אמינות מקור ההעלאה אם יש מידע על כך.

החזר תשובה בפורמט JSON תקין בלבד, בלי טקסט נוסף לפני או אחרי, במבנה הבא:
{{
  "credibility_score": <מספר שלם 0 עד 100>,
  "summary": "<הסבר קצר וברור בעברית, 2-4 משפטים>",
  "red_flags": ["<דגל אדום אם נמצא>", "..."],
  "supporting_points": ["<נקודה שמחזקת אמינות, אם נמצאה>", "..."],
  "sources": ["<תיאור קצר של מקור + קישור אם נמצא>", "..."],
  "confidence": "<low|medium|high>"
}}

אם לא נמצא מידע מספק לקביעה ברורה, ציין זאת ב-summary ותן confidence נמוך, אל תמציא מידע.
"""


def build_prompt(metadata: dict) -> str:
    return ANALYSIS_PROMPT.format(
        title=metadata.get("title") or "לא זמין",
        description=(metadata.get("description") or "לא זמין")[:1200],
        uploader=metadata.get("uploader") or "לא זמין",
        upload_date=metadata.get("upload_date") or "לא זמין",
        webpage_url=metadata.get("webpage_url") or "לא זמין",
    )


def parse_analysis(raw_text: str) -> dict:
    cleaned = raw_text.strip()
    # Strip markdown code fences if the model added them despite instructions
    cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except Exception:
        return {
            "credibility_score": None,
            "summary": raw_text,
            "red_flags": [],
            "supporting_points": [],
            "sources": [],
            "confidence": "low",
        }


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    if not req.url or "http" not in req.url:
        raise HTTPException(status_code=400, detail="כתובת לא תקינה")

    metadata = extract_video_metadata(req.url)
    if "error" in metadata:
        raise HTTPException(
            status_code=400,
            detail=f"לא הצלחתי לחלץ מידע מהקישור (אולי הסרטון פרטי או דורש התחברות): {metadata['error']}",
        )

    prompt = build_prompt(metadata)

    message = client.messages.create(
        model=ANALYSIS_MODEL,
        max_tokens=1500,
        tools=[{"type": "web_search_20250305", "name": "web_search"}],
        messages=[{"role": "user", "content": prompt}],
    )

    text_parts = [block.text for block in message.content if block.type == "text"]
    raw_text = "\n".join(text_parts).strip()
    analysis = parse_analysis(raw_text)

    return {"metadata": metadata, "analysis": analysis}


@app.get("/")
def root():
    return {"status": "ok", "service": "fb-video-checker"}
