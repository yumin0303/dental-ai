def get_verdict(crowding_score: float, threshold: float = 0.5) -> dict:
    """Convert a crowding probability (0-1) into a patient-facing verdict, based on the threshold."""
    high = threshold + 0.2
    if crowding_score >= high:
        return {
            "emoji": "🔴",
            "label": "We recommend consulting an orthodontist",
            "description": "The AI analysis suggests a high likelihood that your teeth alignment needs orthodontic treatment. Please get a thorough evaluation at a nearby orthodontic clinic.",
            "cta": "Consult with an orthodontist",
            "level": "high",
        }
    elif crowding_score >= threshold:
        return {
            "emoji": "🟡",
            "label": "You may want to consider an orthodontic consultation",
            "description": "The AI analysis detected mild alignment issues. It's not urgent, but getting a professional opinion could be helpful.",
            "cta": "Ask about orthodontics at your next regular checkup",
            "level": "medium",
        }
    else:
        return {
            "emoji": "🟢",
            "label": "You likely don't need orthodontic treatment right now",
            "description": "The AI analysis shows your teeth alignment looks relatively good. Keep up with regular dental checkups.",
            "cta": "Keep up regular checkups every 6 months",
            "level": "low",
        }


def get_score_bar(score: float) -> str:
    filled = int(score * 10)
    return "█" * filled + "░" * (10 - filled)
