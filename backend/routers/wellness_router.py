"""
wellness_router.py — Wellness score calculation and history endpoints.

Calculates daily wellness from check-ins + AI session data.
Provides current score and historical trend for dashboard chart.
"""
from fastapi import APIRouter, Depends
from datetime import datetime, timezone, timedelta
from backend.auth import get_current_user
from backend.database import get_db

router = APIRouter(prefix="/api/v1/wellness", tags=["Wellness"])

# Mood → numerical score mapping
MOOD_SCORES = {
    "Calm": 100,
    "Neutral": 75,
    "Anxious": 40,
    "Stressed": 20,
}

# Stress category → score mapping
STRESS_SCORES = {
    "No Stress": 100,
    "Low Stress": 60,
    "High Stress": 15,
    "STRESS DETECTED": 25,
}


async def _calculate_daily_score(db, user_id: str, date: datetime = None) -> dict:
    """
    Calculate wellness score for a specific day.

    Formula:
      Daily Score = (mood_avg × 0.4) + (text_stress_avg × 0.4) + (engagement × 0.2)

    Returns: {"score": int|None, "mood_avg": float, "stress_avg": float, "checkins": int, "sessions": int}
    """
    if date is None:
        date = datetime.now(timezone.utc)

    # Day boundaries (UTC)
    day_start = date.replace(hour=0, minute=0, second=0, microsecond=0)
    day_end = day_start + timedelta(days=1)

    # 1. Check-in moods for this day
    checkins = await db.checkins.find({
        "user_id": user_id,
        "timestamp": {"$gte": day_start, "$lt": day_end},
    }).to_list(length=100)

    # 2. AI sessions/conversations for this day
    conversations = await db.conversations.find({
        "user_id": user_id,
        "timestamp": {"$gte": day_start, "$lt": day_end},
    }).to_list(length=100)

    if not checkins and not conversations:
        return {"score": None, "mood_avg": 0, "stress_avg": 0, "checkins": 0, "sessions": 0}

    # Mood average from check-ins
    mood_avg = 0
    if checkins:
        mood_values = [MOOD_SCORES.get(c.get("mood_state", "Neutral"), 75) for c in checkins]
        # Also factor in stress_score if available
        for c in checkins:
            if c.get("stress_score") is not None:
                # Convert stress score (0-100 where 100=high stress) to wellness (0-100 where 100=no stress)
                wellness_from_stress = max(0, 100 - c["stress_score"])
                mood_values.append(wellness_from_stress)
        mood_avg = sum(mood_values) / len(mood_values)

    # Stress average from AI sessions
    stress_avg = 100  # Default: no stress
    if conversations:
        stress_values = []
        for conv in conversations:
            emotions = conv.get("detected_emotions", {})
            text_result = emotions.get("text", {})
            if isinstance(text_result, dict) and text_result.get("label"):
                label = text_result["label"]
                score = STRESS_SCORES.get(label, 50)
                stress_values.append(score)
        if stress_values:
            stress_avg = sum(stress_values) / len(stress_values)

    # Engagement score
    engagement = 0
    if checkins:
        engagement += 50
    if conversations:
        engagement += 50

    # Final weighted score
    if checkins and conversations:
        score = (mood_avg * 0.4) + (stress_avg * 0.4) + (engagement * 0.2)
    elif checkins:
        score = (mood_avg * 0.6) + (engagement * 0.4)
    else:
        score = (stress_avg * 0.6) + (engagement * 0.4)

    return {
        "score": round(min(100, max(0, score))),
        "mood_avg": round(mood_avg, 1),
        "stress_avg": round(stress_avg, 1),
        "checkins": len(checkins),
        "sessions": len(conversations),
    }


@router.get("/score")
async def get_wellness_score(user: dict = Depends(get_current_user)):
    """Get the current wellness score (today's + 7-day rolling average)."""
    db = get_db()
    user_id = str(user["_id"])

    today = await _calculate_daily_score(db, user_id)

    # 7-day rolling average
    now = datetime.now(timezone.utc)
    week_scores = []
    for i in range(7):
        day = now - timedelta(days=i)
        day_data = await _calculate_daily_score(db, user_id, day)
        if day_data["score"] is not None:
            week_scores.append(day_data["score"])

    weekly_avg = round(sum(week_scores) / len(week_scores)) if week_scores else None

    return {
        "today": today["score"],
        "weekly_average": weekly_avg,
        "checkins_today": today["checkins"],
        "sessions_today": today["sessions"],
    }


@router.get("/history")
async def get_wellness_history(days: int = 30, user: dict = Depends(get_current_user)):
    """Get daily wellness scores for the past N days (for chart)."""
    db = get_db()
    user_id = str(user["_id"])

    now = datetime.now(timezone.utc)
    history = []
    for i in range(days - 1, -1, -1):  # oldest first
        day = now - timedelta(days=i)
        day_data = await _calculate_daily_score(db, user_id, day)
        history.append({
            "date": day.strftime("%Y-%m-%d"),
            "score": day_data["score"],
            "checkins": day_data["checkins"],
            "sessions": day_data["sessions"],
        })

    return {"history": history}


@router.get("/report")
async def get_weekly_report(user: dict = Depends(get_current_user)):
    """
    Generate a plain-text weekly wellness report.
    Summarizes check-ins, AI sessions, mood distribution, and wellness trend.
    """
    db = get_db()
    user_id = str(user["_id"])
    now = datetime.now(timezone.utc)
    week_ago = now - timedelta(days=7)
    display_name = user.get("display_name", "User")

    # Gather week data
    checkins = await db.checkins.find({
        "user_id": user_id,
        "timestamp": {"$gte": week_ago},
    }).to_list(length=500)

    conversations = await db.conversations.find({
        "user_id": user_id,
        "timestamp": {"$gte": week_ago},
    }).to_list(length=500)

    # Mood distribution
    mood_counts = {}
    for c in checkins:
        mood = c.get("mood_state", "Unknown")
        mood_counts[mood] = mood_counts.get(mood, 0) + 1

    # Stress distribution from AI sessions
    stress_counts = {}
    for conv in conversations:
        emotions = conv.get("detected_emotions", {})
        text_result = emotions.get("text", {})
        if isinstance(text_result, dict) and text_result.get("label"):
            label = text_result["label"]
            stress_counts[label] = stress_counts.get(label, 0) + 1

    # Daily scores
    daily_scores = []
    for i in range(6, -1, -1):
        day = now - timedelta(days=i)
        day_data = await _calculate_daily_score(db, user_id, day)
        daily_scores.append({
            "date": day.strftime("%a %d %b"),
            "score": day_data["score"] if day_data["score"] is not None else "--",
        })

    avg_score = None
    valid_scores = [d["score"] for d in daily_scores if isinstance(d["score"], (int, float))]
    if valid_scores:
        avg_score = round(sum(valid_scores) / len(valid_scores))

    # Build report
    report_lines = [
        f"═══════════════════════════════════════════════════",
        f"  StressMitigate — Weekly Wellness Report",
        f"  Generated for: {display_name}",
        f"  Period: {week_ago.strftime('%b %d')} – {now.strftime('%b %d, %Y')}",
        f"═══════════════════════════════════════════════════",
        "",
        f"📊 OVERVIEW",
        f"   Check-ins completed:  {len(checkins)}",
        f"   AI sessions:         {len(conversations)}",
        f"   Average wellness:    {avg_score if avg_score else '--'}/100",
        "",
        f"📅 DAILY SCORES",
    ]
    for d in daily_scores:
        bar = ""
        if isinstance(d["score"], int):
            bar = "█" * (d["score"] // 5) + "░" * (20 - d["score"] // 5)
        report_lines.append(f"   {d['date']:>10}  {str(d['score']):>3}  {bar}")

    report_lines.append("")
    report_lines.append(f"😊 MOOD DISTRIBUTION")
    if mood_counts:
        for mood, count in sorted(mood_counts.items(), key=lambda x: -x[1]):
            report_lines.append(f"   {mood:<12} {'●' * count} ({count})")
    else:
        report_lines.append("   No check-ins recorded this week.")

    report_lines.append("")
    report_lines.append(f"🧠 AI STRESS ANALYSIS")
    if stress_counts:
        for label, count in sorted(stress_counts.items(), key=lambda x: -x[1]):
            report_lines.append(f"   {label:<18} {'●' * count} ({count})")
    else:
        report_lines.append("   No AI sessions recorded this week.")

    # Recommendation in report
    report_lines.extend([
        "",
        f"💡 RECOMMENDATION",
    ])
    if avg_score is not None:
        if avg_score >= 75:
            report_lines.append("   You're doing great! Keep up your daily check-ins and mindfulness practice.")
        elif avg_score >= 50:
            report_lines.append("   Moderate stress detected. Try the breathing exercises and maintain regular check-ins.")
        else:
            report_lines.append("   Elevated stress levels. Consider speaking with a mental health professional.")
            report_lines.append("   Crisis helplines: AASRA (9820466726) | iCall (9152987821) | 988 (US)")
    else:
        report_lines.append("   Check in more regularly to receive personalized recommendations.")

    report_lines.extend([
        "",
        f"═══════════════════════════════════════════════════",
        f"  Stay mindful. You're not alone. 💚",
        f"═══════════════════════════════════════════════════",
    ])

    return {"report": "\n".join(report_lines)}


@router.get("/recommendations")
async def get_recommendations(user: dict = Depends(get_current_user)):
    """
    Personalized recommendations based on recent wellness data.
    Includes psychiatrist recommendation for persistent high stress.
    """
    db = get_db()
    user_id = str(user["_id"])
    now = datetime.now(timezone.utc)
    week_ago = now - timedelta(days=7)

    checkins = await db.checkins.find({
        "user_id": user_id,
        "timestamp": {"$gte": week_ago},
    }).to_list(length=500)

    conversations = await db.conversations.find({
        "user_id": user_id,
        "timestamp": {"$gte": week_ago},
    }).to_list(length=500)

    tips = []
    urgency = "normal"  # normal, moderate, high

    # Count stressed check-ins
    stressed_count = sum(1 for c in checkins if c.get("mood_state") in ("Stressed", "Anxious"))
    total_checkins = len(checkins)

    # Count high-stress AI sessions
    high_stress_sessions = 0
    for conv in conversations:
        emotions = conv.get("detected_emotions", {})
        text_result = emotions.get("text", {})
        if isinstance(text_result, dict) and text_result.get("label") in ("High Stress", "STRESS DETECTED"):
            high_stress_sessions += 1

    # Generate tips based on data
    if total_checkins == 0 and len(conversations) == 0:
        tips.append("Start by doing a daily check-in to track your mood patterns.")
        tips.append("Try a 5-minute breathing exercise from the Exercises tab.")
    elif total_checkins < 3:
        tips.append("Try checking in at least once daily for better tracking.")

    if stressed_count > 3:
        tips.append("You've been feeling stressed often. The 5-4-3-2-1 grounding exercise may help.")
        urgency = "moderate"

    if high_stress_sessions > 2:
        tips.append("Multiple sessions show elevated stress. Consider progressive muscle relaxation.")
        urgency = "moderate"

    # ── PSYCHIATRIST RECOMMENDATION ──
    if (stressed_count >= 5 or high_stress_sessions >= 4 or
            (total_checkins > 0 and stressed_count / max(total_checkins, 1) > 0.7)):
        urgency = "high"
        tips.insert(0, "⚠️ Persistent high stress detected. We strongly recommend speaking with a mental health professional.")
        tips.append("📞 AASRA: 9820466726 | iCall: 9152987821 | Vandrevala Foundation: 1860-2662-345")

    if not tips:
        tips.append("You're doing well! Keep maintaining your daily check-in habit.")
        tips.append("Try journaling your thoughts using the Mindful Journaling exercise.")

    return {
        "tips": tips,
        "urgency": urgency,
        "stats": {
            "total_checkins": total_checkins,
            "stressed_checkins": stressed_count,
            "ai_sessions": len(conversations),
            "high_stress_sessions": high_stress_sessions,
        },
    }


@router.get("/triggers")
async def get_stress_triggers(user: dict = Depends(get_current_user)):
    """
    Analyze time-of-day patterns to identify when user is most stressed.
    Groups check-ins into time buckets: Morning, Afternoon, Evening, Night.
    """
    db = get_db()
    user_id = str(user["_id"])
    now = datetime.now(timezone.utc)
    month_ago = now - timedelta(days=30)

    checkins = await db.checkins.find({
        "user_id": user_id,
        "timestamp": {"$gte": month_ago},
    }).to_list(length=500)

    if len(checkins) < 3:
        return {
            "has_data": False,
            "message": "Need at least 3 check-ins to identify patterns.",
            "buckets": [],
        }

    # Time buckets (using IST offset +5:30 from UTC)
    buckets = {
        "Morning (6AM–12PM)": {"total": 0, "stressed": 0, "icon": "☀️"},
        "Afternoon (12PM–5PM)": {"total": 0, "stressed": 0, "icon": "🌤️"},
        "Evening (5PM–9PM)": {"total": 0, "stressed": 0, "icon": "🌅"},
        "Night (9PM–6AM)": {"total": 0, "stressed": 0, "icon": "🌙"},
    }

    for c in checkins:
        ts = c.get("timestamp")
        if not ts:
            continue
        # Approximate IST
        hour = (ts.hour + 5) % 24  # +5 for IST offset (approx)
        mood = c.get("mood_state", "Neutral")
        is_stressed = mood in ("Stressed", "Anxious")

        if 6 <= hour < 12:
            bucket = "Morning (6AM–12PM)"
        elif 12 <= hour < 17:
            bucket = "Afternoon (12PM–5PM)"
        elif 17 <= hour < 21:
            bucket = "Evening (5PM–9PM)"
        else:
            bucket = "Night (9PM–6AM)"

        buckets[bucket]["total"] += 1
        if is_stressed:
            buckets[bucket]["stressed"] += 1

    # Find peak stress time
    result_buckets = []
    peak_bucket = None
    peak_ratio = 0

    for name, data in buckets.items():
        if data["total"] > 0:
            ratio = data["stressed"] / data["total"]
            result_buckets.append({
                "name": name,
                "icon": data["icon"],
                "total": data["total"],
                "stressed": data["stressed"],
                "ratio": round(ratio * 100),
            })
            if ratio > peak_ratio:
                peak_ratio = ratio
                peak_bucket = name

    return {
        "has_data": True,
        "peak_stress_time": peak_bucket,
        "peak_stress_ratio": round(peak_ratio * 100),
        "buckets": result_buckets,
    }

