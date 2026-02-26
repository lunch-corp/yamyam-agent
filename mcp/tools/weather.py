"""실제 날씨 API 연동 도구 (OpenWeatherMap)"""

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime


def get_weather(city: str = "서울") -> str:
    """
    OpenWeatherMap API를 사용하여 지정한 도시의 현재 날씨 정보를 반환합니다.
    
    OPENWEATHER_API_KEY 환경 변수가 설정되어 있어야 합니다.
    무료 API 키는 https://openweathermap.org/api 에서 발급받을 수 있습니다.
    
    Args:
        city: 도시 이름 (한글/영문 모두 가능, 기본값: "서울")
    
    Returns:
        날씨 정보 문자열 (API 오류 시 오류 메시지)
    """
    api_key = os.environ.get("OPENWEATHER_API_KEY", "").strip()
    
    if not api_key:
        return """❌ 오류: OPENWEATHER_API_KEY 환경 변수가 설정되지 않았습니다.

설정 방법:
1. https://openweathermap.org/api 에서 무료 API 키 발급
2. .env 파일에 다음 추가:
   OPENWEATHER_API_KEY=your_api_key_here
3. MCP 서버 재시작"""
    
    # 한글 도시명을 영문으로 매핑 (주요 도시)
    city_map = {
        "서울": "Seoul",
        "부산": "Busan",
        "인천": "Incheon",
        "대구": "Daegu",
        "대전": "Daejeon",
        "광주": "Gwangju",
        "울산": "Ulsan",
        "수원": "Suwon",
        "제주": "Jeju",
    }
    
    city_en = city_map.get(city, city)
    
    # OpenWeatherMap API 호출
    base_url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": city_en,
        "appid": api_key,
        "units": "metric",  # 섭씨 온도
        "lang": "kr",  # 한국어 설명
    }
    
    url = f"{base_url}?{urllib.parse.urlencode(params)}"
    
    try:
        req = urllib.request.Request(url, method="GET")
        req.add_header("Accept", "application/json")
        
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
        
        # 날씨 정보 파싱
        weather_desc = data["weather"][0]["description"]
        temp = data["main"]["temp"]
        feels_like = data["main"]["feels_like"]
        humidity = data["main"]["humidity"]
        pressure = data["main"]["pressure"]
        wind_speed = data["wind"]["speed"]
        
        # 날씨 아이콘 선택
        weather_id = data["weather"][0]["id"]
        if weather_id < 300:
            icon = "⛈️"
        elif weather_id < 600:
            icon = "🌧️"
        elif weather_id < 700:
            icon = "❄️"
        elif weather_id < 800:
            icon = "🌫️"
        elif weather_id == 800:
            icon = "☀️"
        else:
            icon = "☁️"
        
        return f"""{icon} {city} 날씨 정보
━━━━━━━━━━━━━━━━━━
날씨: {weather_desc}
기온: {temp:.1f}°C (체감: {feels_like:.1f}°C)
습도: {humidity}%
기압: {pressure} hPa
풍속: {wind_speed} m/s
조회 시간: {datetime.now().strftime('%Y-%m-%d %H:%M')}

※ OpenWeatherMap API 실시간 데이터"""
        
    except urllib.error.HTTPError as e:
        if e.code == 401:
            return "❌ API 키가 유효하지 않습니다. OPENWEATHER_API_KEY를 확인하세요."
        elif e.code == 404:
            return f"❌ 도시를 찾을 수 없습니다: {city}\n영문 도시명을 사용하거나 다른 도시명을 시도해보세요."
        else:
            body = e.read().decode() if e.fp else ""
            return f"❌ API 오류 (HTTP {e.code}): {body or e.reason}"
    
    except urllib.error.URLError as e:
        return f"❌ 연결 오류: {e.reason}\n인터넷 연결을 확인하세요."
    
    except (json.JSONDecodeError, KeyError) as e:
        return f"❌ 응답 파싱 오류: {e}"
    
    except Exception as e:
        return f"❌ 오류: {type(e).__name__}: {e}"
