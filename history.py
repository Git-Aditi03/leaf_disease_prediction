# history.py
# Manages the last 5 prediction history shown in the UI

MAX_HISTORY = 5
history_store = []


def add_to_history(disease_name: str, confidence: float, is_healthy: bool, emoji: str):
    """Add a new prediction to history. Keeps only last MAX_HISTORY entries."""
    entry = {
        "name":    disease_name.replace("___", " — ").replace("_", " "),
        "conf":    confidence,
        "healthy": is_healthy,
        "emoji":   emoji,
    }
    history_store.insert(0, entry)
    if len(history_store) > MAX_HISTORY:
        history_store.pop()


def get_history_html() -> str:
    """Returns prediction history as styled HTML."""
    if not history_store:
        return "<div class='no-history'>No predictions yet. Upload a leaf to get started! 🌿</div>"

    items = ""
    for h in history_store:
        items += f"""
        <div class='history-item'>
            <span class='history-emoji'>{h['emoji']}</span>
            <span class='history-name'>{h['name']}</span>
            <span class='history-conf'>{h['conf']:.0f}%</span>
        </div>"""
    return f"<div class='history-list'>{items}</div>"


def clear_history():
    """Clears all prediction history."""
    history_store.clear()