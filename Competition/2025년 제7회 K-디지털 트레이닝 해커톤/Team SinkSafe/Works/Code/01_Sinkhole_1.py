import rasterio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

# ✅ 한글 폰트 설정 (윈도우 기본: 'Malgun Gothic')
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 깨짐 방지

# ✅ 파일 경로
folder_path = r'C:\python\sinkhole'
before_path = os.path.join(folder_path, 'dem_before.tif')
after_path  = os.path.join(folder_path, 'dem_after.tif')

# ✅ DEM 읽기
with rasterio.open(before_path) as src_before:
    before = src_before.read(1)
    profile = src_before.profile

with rasterio.open(after_path) as src_after:
    after = src_after.read(1)

# ✅ 침하량 계산
diff = after - before  # 음수 = 침하

# ✅ 통계 출력
print("고도 변화량 통계:")
print(f"최대 상승: {np.nanmax(diff):.2f} m")
print(f"최대 침하: {np.nanmin(diff):.2f} m")
print(f"평균 변화: {np.nanmean(diff):.2f} m")

# ✅ 전체 고도 변화 시각화
plt.figure(figsize=(10, 8))
plt.title('지반 침하/융기 지도 (단위: m)')
img = plt.imshow(diff, cmap='bwr', vmin=-5, vmax=5)
plt.colorbar(img, label='고도 변화')
plt.tight_layout()
plt.show()

# ✅ 침하 마스킹 (1m 이상 침하 지역만)
sink_mask = diff < -1.0

plt.figure(figsize=(10, 8))
plt.title('침하 위험 지역 (1m 이상)')
plt.imshow(sink_mask, cmap='gray')
plt.tight_layout()
plt.show()
