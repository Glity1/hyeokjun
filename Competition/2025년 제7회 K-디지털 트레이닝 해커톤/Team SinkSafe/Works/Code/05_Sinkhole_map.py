import geopandas as gpd
import folium
import os

# ✅ 공통 설정
INPUT_DIR = './output'
OUTPUT_DIR = './output'
CENTER_LAT = 37.4826  # 소사역 위도
CENTER_LON = 126.7958  # 소사역 경도

# ✅ 지도 저장 함수
def save_folium_map(geojson_path, label, center_lat, center_lon):
    # GeoJSON 읽기
    gdf = gpd.read_file(geojson_path)

    # 지도 생성
    m = folium.Map(location=[center_lat, center_lon], zoom_start=15, tiles='cartodb positron')

    # Tooltip 필드 자동 지정 (좌표형 제외)
    tooltip_fields = [col for col in gdf.columns if col != 'geometry']

    # GeoJson 레이어 추가
    folium.GeoJson(
        gdf,
        name=f'{label} Grid',
        style_function=lambda x: {
            'fillColor': '#3186cc',
            'color': 'black',
            'weight': 1,
            'fillOpacity': 0.4
        },
        tooltip=folium.GeoJsonTooltip(fields=tooltip_fields)
    ).add_to(m)

    # 저장
    html_path = os.path.join(OUTPUT_DIR, f'map_{label}.html')
    m.save(html_path)
    print(f"✅ 저장 완료: {html_path}")

# ✅ 파일 리스트
files = {
    'sosa_50m': 'sosa_50m.geojson',
    'sosa_100m': 'sosa_100m.geojson',
    'sosa_200m': 'sosa_200m.geojson',
    'sosa_300m': 'sosa_300m.geojson',
    'bucheon_all': 'bucheon_all.geojson',
}

# ✅ 반복 저장
for label, filename in files.items():
    geojson_path = os.path.join(INPUT_DIR, filename)
    if os.path.exists(geojson_path):
        save_folium_map(geojson_path, label=label, center_lat=CENTER_LAT, center_lon=CENTER_LON)
    else:
        print(f"⚠️ 파일 없음: {geojson_path}")
