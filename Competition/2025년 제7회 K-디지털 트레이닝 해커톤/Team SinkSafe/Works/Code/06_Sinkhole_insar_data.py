import requests
import os
from requests.auth import HTTPBasicAuth
import time

# ✅ 사용자 ASF 계정 정보
ASF_USERNAME = "glity"
ASF_PASSWORD = "Bceokps315!@"

# ✅ 저장 경로
DOWNLOAD_DIR = "./sentinel1_slc_download"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# ✅ 검색 범위 (전체 데이터를 위한 큰 범위)
WKT_POLYGON = "POLYGON((123.613 35.5489,129.7284 35.5489,129.7284 39.2903,123.613 39.2903,123.613 35.5489))"

# ✅ 날짜 범위
START_DATE = "2020-10-01T00:00:00Z"
END_DATE = "2025-07-26T23:59:59Z"

# ✅ 검색 URL
base_url = "https://api.daac.asf.alaska.edu/services/search/param"

# ✅ 초기 파라미터
params = {
    "platform": "Sentinel-1",
    "processingLevel": "SLC",
    "beamMode": "IW",
    "intersectsWith": WKT_POLYGON,
    "start": START_DATE,
    "end": END_DATE,
    "output": "json",
    "maxResults": 100
}

headers = {
    "Accept": "application/json"
}

# ✅ 전체 granule 저장 리스트
granules_all = []

print("🔍 Sentinel-1 SLC 전체 granule 검색 시작 (pagination 기반)...")

while True:
    response = requests.get(base_url, params=params, headers=headers, auth=HTTPBasicAuth(ASF_USERNAME, ASF_PASSWORD))

    if response.status_code != 200:
        print(f"❌ 요청 실패: HTTP {response.status_code}")
        print(response.text)
        break

    data = response.json()

    # ⭐ 이전 수정 부분: 응답 데이터가 리스트인지 딕셔너리인지 확인
    if isinstance(data, list):
        current_features = data
        next_url = None # 리스트 형태일 경우 더 이상 다음 페이지는 없음
    elif isinstance(data, dict):
        current_features = data.get("features", [])
        next_url = data.get("next")
    else:
        print(f"⚠️ 예상치 못한 응답 형식: {type(data)} → 다음 페이지로 넘어갈 수 없습니다.")
        break

    granules_all.extend(current_features)

    print(f"📦 현재까지 수집: {len(granules_all)}개")

    # 다음 페이지 링크가 있는 경우 계속
    if not next_url:
        print("✅ 모든 페이지 수집 완료")
        break

    # 다음 요청을 위해 URL만 변경
    base_url = next_url
    params = {}  # 다음 페이지에는 추가 파라미터 필요 없음
    time.sleep(1)

print(f"🎯 총 granule 수집 완료: {len(granules_all)}개")

# ---
## 파일 다운로드

# ✅ 다운로드 수행
for i, granule_item in enumerate(granules_all): # 변수명을 granule_item으로 변경하여 혼동 방지
    # ⭐ 수정된 부분: granule_item이 리스트인지 딕셔너리인지 확인
    if isinstance(granule_item, list) and len(granule_item) > 0:
        actual_granule = granule_item[0] # 리스트라면 첫 번째 요소를 가져옴
    elif isinstance(granule_item, dict):
        actual_granule = granule_item
    else:
        print(f"⚠️ {i+1}번 granule_item의 형식 오류 ({type(granule_item)}) → 건너김")
        continue

    props = actual_granule.get("properties", {})
    file_url = props.get("download_url")
    scene_id = props.get("sceneName")

    if not file_url or not scene_id:
        print(f"⚠️ {i+1}번 granule 정보 부족 → 건너김")
        continue

    save_path = os.path.join(DOWNLOAD_DIR, f"{scene_id}.zip")
    if os.path.exists(save_path):
        print(f"✅ {scene_id} 이미 다운로드됨 → 건너김")
        continue

    print(f"⬇️ [{i+1}/{len(granules_all)}] 다운로드 중: {scene_id}")
    try:
        with requests.get(file_url, auth=HTTPBasicAuth(ASF_USERNAME, ASF_PASSWORD), stream=True) as r:
            r.raise_for_status()
            with open(save_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"✅ 다운로드 완료: {scene_id}")
    except Exception as e:
        print(f"❌ 다운로드 실패: {scene_id} → 오류: {e}")