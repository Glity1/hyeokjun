import os
import requests
import zipfile
import subprocess
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from shapely.geometry import Point
from geopy.distance import geodesic

# ✅ 1. 기본 설정
CENTER_LAT = 37.4831   # 소사역 위도
CENTER_LON = 126.7951  # 소사역 경도
RADIUS_KM = 5
START_DATE = "2014-10-01"
END_DATE = "2025-07-13"

SAVE_DIR = "./insar_auto"
SAFE_DIR = os.path.join(SAVE_DIR, "SAFE")
os.makedirs(SAFE_DIR, exist_ok=True)

# ✅ 2. 반경 내 포함 확인
def is_within_radius(lat, lon, center_lat, center_lon, radius_km):
    return geodesic((lat, lon), (center_lat, center_lon)).km <= radius_km

# ✅ 3. ASF API 검색
def search_asf():
    print(f"\n🔍 검색 중: {START_DATE} ~ {END_DATE}")
    url = "https://api.daac.asf.alaska.edu/services/search/param"
    params = {
        "platform": "Sentinel-1",
        "processingLevel": "SLC",
        "beamMode": "IW",
        "polarization": "VV",
        "intersectsWith": f"POINT({CENTER_LON} {CENTER_LAT})",
        "start": START_DATE,
        "end": END_DATE,
        "output": "json",
        "maxResults": 2000
    }

    r = requests.get(url, params=params)
    try:
        data = r.json()
    except ValueError:
        print("❌ JSON 디코딩 실패")
        print("📎 응답 내용:", r.text)
        return []

    if not isinstance(data, dict) or "features" not in data:
        print("❌ 'features'가 응답에 없음 (데이터 없음 또는 검색 조건 오류)")
        print("📎 응답 내용:", data)
        return []

    results = data["features"]
    print(f"📦 총 {len(results)}개 장면 검색됨")

    filtered = []
    for r in results:
        coords = r.get("geometry", {}).get("coordinates", [[]])[0]
        for lon, lat in coords:
            if is_within_radius(lat, lon, CENTER_LAT, CENTER_LON, RADIUS_KM):
                filtered.append(r)
                break

    print(f"✅ 반경 {RADIUS_KM}km 이내 장면 수: {len(filtered)}")
    return filtered

# ✅ 4. 다운로드 및 압축 해제
def download_safe_files(results, limit=2):
    for i, item in enumerate(tqdm(results[:limit])):
        props = item["properties"]
        file_id = props["fileID"]
        url = props["url"]
        zip_path = os.path.join(SAFE_DIR, file_id + ".zip")
        extract_path = os.path.join(SAFE_DIR, file_id + ".SAFE")

        if not os.path.exists(zip_path):
            r = requests.get(url, stream=True)
            with open(zip_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        if not os.path.exists(extract_path):
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(SAFE_DIR)
    print("✅ 다운로드 및 압축 해제 완료")

# ✅ 5. GPT XML 생성
def create_gpt_xml(master_safe, slave_safe, output_folder):
    master_manifest = os.path.join(master_safe, "manifest.safe")
    slave_manifest = os.path.join(slave_safe, "manifest.safe")
    xml = f"""<graph id=\"InSAR Workflow\">
  <version>1.0</version>
  <node id=\"Read1\">
    <operator>Read</operator>
    <parameters>
      <file>{master_manifest}</file>
    </parameters>
  </node>
  <node id=\"Read2\">
    <operator>Read</operator>
    <parameters>
      <file>{slave_manifest}</file>
    </parameters>
  </node>
  <node id=\"BackGeocoding\">
    <operator>Back-Geocoding</operator>
    <sources>
      <sourceProduct>${{Read1}}</sourceProduct>
      <sourceProduct>${{Read2}}</sourceProduct>
    </sources>
  </node>
  <node id=\"Interferogram\">
    <operator>Interferogram</operator>
    <sources>
      <sourceProduct>${{BackGeocoding}}</sourceProduct>
    </sources>
  </node>
  <node id=\"Write\">
    <operator>Write</operator>
    <sources>
      <sourceProduct>${{Interferogram}}</sourceProduct>
    </sources>
    <parameters>
      <file>{output_folder}/insar_result.dim</file>
      <formatName>BEAM-DIMAP</formatName>
    </parameters>
  </node>
</graph>"""
    with open("insar_graph.xml", "w") as f:
        f.write(xml)

# ✅ 6. SNAP GPT 실행
def run_snap_gpt(master_path, slave_path):
    out_dir = os.path.join(SAVE_DIR, "insar_output")
    os.makedirs(out_dir, exist_ok=True)
    create_gpt_xml(master_path, slave_path, out_dir)
    cmd = ["gpt", "insar_graph.xml"]
    print("⚙️ SNAP GPT 실행 중...")
    subprocess.run(cmd)
    print(f"✅ InSAR 분석 완료: {out_dir}")
    return os.path.join(out_dir, "insar_result_phase.img")

# ✅ 7. 시각화
def visualize_insar(phase_img_path):
    with rasterio.open(phase_img_path) as src:
        data = src.read(1).astype(float)
        data[data == src.nodata] = np.nan
    plt.figure(figsize=(10, 8))
    plt.imshow(data, cmap="jet")
    plt.colorbar(label="위상 변화 (Phase)")
    plt.title("InSAR 위상 변화 결과")
    plt.xlabel("X (pixel)")
    plt.ylabel("Y (pixel)")
    plt.tight_layout()
    plt.show()

# ✅ 8. 전체 실행
if __name__ == "__main__":
    results = search_asf()
    if len(results) < 2:
        print("❗ 2개 이상의 장면이 있어야 InSAR 분석 가능. 검색 범위 또는 날짜를 조정해보세요.")
    else:
        download_safe_files(results, limit=2)
        safe_folders = sorted([f for f in os.listdir(SAFE_DIR) if f.endswith(".SAFE")])
        if len(safe_folders) < 2:
            print("❗ .SAFE 폴더가 부족하여 분석 불가.")
        else:
            master_safe = os.path.join(SAFE_DIR, safe_folders[0])
            slave_safe  = os.path.join(SAFE_DIR, safe_folders[1])
            phase_img_path = run_snap_gpt(master_safe, slave_safe)
            visualize_insar(phase_img_path)


#https://search.asf.alaska.edu/#/?zoom=14.944&center=126.792,37.485&polygon=POLYGON((-233.2205%2037.4817,-233.2026%2037.4817,-233.2026%2037.494,-233.2205%2037.494,-233.2205%2037.4817))&resultsLoaded=true&granule=S1A_IW_SLC__1SDV_20250630T093218_20250630T093244_059874_076FDD_AFC5-SLC&isDlOpen=true