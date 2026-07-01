def get_verdict(crowding_score: float, threshold: float = 0.5) -> dict:
    """crowding 확률(0~1)을 환자용 판정으로 변환. threshold 기준으로 판정."""
    high = threshold + 0.2
    if crowding_score >= high:
        return {
            "emoji": "🔴",
            "label": "교정 전문의 상담을 권장드려요",
            "description": "AI 분석 결과, 치아 배열에 교정이 필요할 가능성이 높습니다. 가까운 교정과에서 정밀 진단을 받아보세요.",
            "cta": "교정과 전문의와 상담하세요",
            "level": "high",
        }
    elif crowding_score >= threshold:
        return {
            "emoji": "🟡",
            "label": "교정 상담을 고려해볼 수 있어요",
            "description": "AI 분석 결과, 경미한 치아 배열 문제가 감지되었습니다. 지금 당장 급하지 않지만 전문의 의견을 들어보면 도움이 될 수 있어요.",
            "cta": "정기 검진 시 교정 상담도 함께 받아보세요",
            "level": "medium",
        }
    else:
        return {
            "emoji": "🟢",
            "label": "지금은 교정이 필요하지 않을 수 있어요",
            "description": "AI 분석 결과, 치아 배열이 비교적 양호합니다. 정기적인 치과 검진을 유지하세요.",
            "cta": "6개월마다 정기 검진을 유지하세요",
            "level": "low",
        }


def get_score_bar(score: float) -> str:
    filled = int(score * 10)
    return "█" * filled + "░" * (10 - filled)
