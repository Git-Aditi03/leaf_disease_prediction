# weather.py
# Handles all weather API calls and treatment timing logic

import requests


def get_weather(city: str, api_key: str):
    """Fetch today's weather from Visual Crossing. Returns dict or None."""
    try:
        url = (
            f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/"
            f"timeline/{city}?unitGroup=metric&key={api_key}&contentType=json"
        )
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return None
        today = resp.json()["days"][0]
        return {
            "temp": today["temp"],
            "humidity": today["humidity"],
            "description": today["conditions"],
            "rain": today["precip"] > 0,
        }
    except Exception:
        return None


def get_treatment_timing(disease_name: str, weather, base_days: int) -> str:
    """Returns simple spray timing advice based on weather conditions."""
    if weather is None:
        return f"Spray every {base_days} days"

    h    = weather["humidity"]
    rain = weather["rain"]

    if "Late_blight" in disease_name and (rain or h > 85):
        return "🚨 Spray TODAY — rain/humidity makes this very urgent!"
    if rain and h > 80:
        return f"Spray every {max(base_days - 4, 1)} days (wet weather = more frequent)"
    if h > 85:
        return f"Spray every {max(base_days - 3, 1)} days (high humidity detected)"
    return f"Spray every {base_days} days (normal weather conditions)"


def build_weather_html(city: str, weather: dict) -> str:
    """Returns weather info as a styled HTML card."""
    if not weather:
        return ""
    rain_icon = "🌧️" if weather["rain"] else "☀️"
    return f"""
    <div class='weather-card'>
        {rain_icon} <b>{city}</b> — {weather['temp']}°C,
        Humidity {weather['humidity']}%, {weather['description']}
    </div>"""