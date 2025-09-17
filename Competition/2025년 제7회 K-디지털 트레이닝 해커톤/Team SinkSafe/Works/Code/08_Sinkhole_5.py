import requests
import os
from tqdm import tqdm

# ✅ 너의 API 키 입력
API_KEY = " a18e39f7516f0e61387a868df9b2638c"

# ✅ 사용할 DEM 타입
DEM_TYPES = ["SRTMGL1", "AW3D30"]

# ✅ 새롭게 지정한 영역 (네가 보낸 링크 기준)
south = 37.47139681053474
north = 37.53281591664346
west  = 126.72652244567874
east  = 126.80977821350098

# ✅ 저장 폴더
SAVE_DIR = "./dem_downloads"
os.makedirs(SAVE_DIR, exist_ok=True)

# ✅ 다운로드 함수
def download_dem(dem_type):
    url = "https://portal.opentopography.org/API/globaldem"
    params = {
        "demtype": dem_type,
        "south": south,
        "north": north,
        "west": west,
        "east": east,
        "outputFormat": "GTiff",
        "API_Key": API_KEY,
    }

    print(f"\n📥 {dem_type} 다운로드 중...")
    response = requests.get(url, params=params, stream=True)

    if response.status_code == 200:
        file_path = os.path.join(SAVE_DIR, f"{dem_type}.tif")
        with open(file_path, "wb") as f:
            for chunk in tqdm(response.iter_content(chunk_size=8192), desc=dem_type):
                f.write(chunk)
        print(f"✅ 저장 완료: {file_path}")
    else:
        print(f"❌ 오류 발생: {response.status_code} - {response.text}")

# ✅ 실행
if __name__ == "__main__":
    for dem in DEM_TYPES:
        download_dem(dem)


import rasterio
import numpy as np
import matplotlib.pyplot as plt
import os
import platform

# ✅ 폰트 설정 (한글 깨짐 방지)
if platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')
elif platform.system() == 'Darwin':
    plt.rc('font', family='AppleGothic')
else:
    plt.rc('font', family='NanumGothic')
plt.rcParams['axes.unicode_minus'] = False

# ✅ DEM 파일 경로
srtm_path = './dem_downloads/SRTMGL1.tif'
aw3d_path = './dem_downloads/AW3D30.tif'

# ✅ 파일 열기 및 데이터 읽기
with rasterio.open(srtm_path) as src1, rasterio.open(aw3d_path) as src2:
    srtm = src1.read(1).astype(float)
    aw3d = src2.read(1).astype(float)
    profile = src1.profile

# ✅ 결측값 처리
srtm[srtm == src1.nodata] = np.nan
aw3d[aw3d == src2.nodata] = np.nan

# ✅ 고도 차이 계산 (음수: 침하, 양수: 융기)
diff = srtm - aw3d

# ✅ 1. 고도차 시각화
plt.figure(figsize=(10, 8))
plt.imshow(diff, cmap='seismic', vmin=-10, vmax=10)
plt.colorbar(label='고도 차이 (m)')
plt.title('SRTMGL1 - AW3D30 고도 차이 (침하/융기)')
plt.xlabel('X (Pixel)')
plt.ylabel('Y (Pixel)')
plt.tight_layout()
plt.show()

# ✅ 2. 히스토그램: 고도차 분포
plt.figure(figsize=(8, 5))
plt.hist(diff[~np.isnan(diff)].flatten(), bins=100, color='gray', edgecolor='black')
plt.title('고도 차이 분포 (SRTMGL1 - AW3D30)')
plt.xlabel('고도 차이 (m)')
plt.ylabel('픽셀 수')
plt.grid(True)
plt.tight_layout()
plt.show()

# ✅ 3. 통계 요약 출력
print("📊 고도차 통계 요약:")
print(f" - 평균: {np.nanmean(diff):.3f} m")
print(f" - 표준편차: {np.nanstd(diff):.3f} m")
print(f" - 최소값: {np.nanmin(diff):.3f} m")
print(f" - 최대값: {np.nanmax(diff):.3f} m")

# ✅ 4. ±3m 이상 고도 변화 지역 마스킹
mask = np.abs(diff) >= 3

plt.figure(figsize=(10, 8))
plt.imshow(mask, cmap='gray')
plt.title('±3m 이상 고도차 발생 지역 (침하/융기)')
plt.xlabel('X (Pixel)')
plt.ylabel('Y (Pixel)')
plt.tight_layout()
plt.show()

# ✅ 고도차 계산
diff = srtm - aw3d  # 음수: 침하, 양수: 융기

# ✅ 시각화
plt.figure(figsize=(10, 8))
plt.imshow(diff, cmap='seismic', vmin=-10, vmax=10)
plt.colorbar(label='고도 차이 (m)')
plt.title('SRTMGL1 - AW3D30 고도 차이 (침하/융기)')
plt.xlabel('X (Pixel)')
plt.ylabel('Y (Pixel)')
plt.tight_layout()
plt.show()

# ✅ GeoTIFF 저장 설정
diff_output_path = './dem_downloads/diff_map.tif'
profile.update(
    dtype='float32',
    nodata=np.nan
)

# ✅ 고도차 GeoTIFF 저장
with rasterio.open(diff_output_path, 'w', **profile) as dst:
    dst.write(diff.astype(np.float32), 1)

print(f"✅ 고도차 GeoTIFF 저장 완료: {diff_output_path}")

