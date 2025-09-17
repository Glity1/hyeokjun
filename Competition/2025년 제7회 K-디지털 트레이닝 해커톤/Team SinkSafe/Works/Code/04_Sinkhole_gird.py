# import geopandas as gpd
# import pandas as pd
# import numpy as np
# from shapely.geometry import Point
# from geopy.distance import geodesic
# import os

# # ✅ 경고 제거용 설정 (Centroid 경고 방지)
# import warnings
# warnings.filterwarnings("ignore", category=UserWarning)

# # ✅ 설정
# CENTER_LAT = 37.4826
# CENTER_LON = 126.7958
# CENTER_POINT = (CENTER_LAT, CENTER_LON)
# RADIUS_LIST = [50, 100, 200, 300]  # 단위: meter
# INPUT_SHP_PATH = "./sinkhole/경기도 부천시_국가지점번호 기반_임의 격자 데이터_20221220/국가지점번호_기반_격자.shp"
# OUTPUT_DIR = "./output"
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # ✅ 격자 데이터 불러오기 및 좌표계 변환
# gdf = gpd.read_file(INPUT_SHP_PATH)
# gdf = gdf.to_crs(epsg=4326)  # WGS84 좌표계 (GPS용)

# # ✅ 중심점과 각 격자의 중심점 계산
# gdf["centroid"] = gdf.geometry.centroid
# gdf["x"] = gdf["centroid"].x  # 경도
# gdf["y"] = gdf["centroid"].y  # 위도

# # ✅ 중심 반경 필터 함수 (단위: meter)
# def is_within_radius(row, radius_m):
#     point = (row["y"], row["x"])  # 위도, 경도
#     return geodesic(CENTER_POINT, point).m <= radius_m

# # ✅ 반경별 필터링 및 저장
# for radius in RADIUS_LIST:
#     gdf_radius = gdf[gdf.apply(lambda row: is_within_radius(row, radius), axis=1)].copy()
    
#     # centroid 제거 (geometry 하나만 있어야 to_file 가능)
#     if "centroid" in gdf_radius.columns:
#         gdf_radius = gdf_radius.drop(columns=["centroid"])
    
#     # 저장
#     output_path = f"{OUTPUT_DIR}/sosa_{radius}m.geojson"
#     gdf_radius.to_file(output_path, driver="GeoJSON", index=False)
#     print(f"✅ 저장 완료: {output_path} (격자 수: {len(gdf_radius)})")

# # ✅ 부천시 전체 저장 (centroid 제거)
# gdf_total = gdf.drop(columns=["centroid"])
# gdf_total.to_file(f"{OUTPUT_DIR}/bucheon_all.geojson", driver="GeoJSON", index=False)
# print(f"✅ 부천시 전체 저장 완료: bucheon_all.geojson")


import geopandas as gpd
import folium

# ✅ GeoJSON 파일 불러오기 (파일 경로 확인 필요)
gdf = gpd.read_file('./output/sosa_200m.geojson')

# ✅ 중심 좌표 (소사역)
center_lat = 37.4826
center_lon = 126.7958

# ✅ Folium 지도 객체 생성
m = folium.Map(location=[center_lat, center_lon], zoom_start=15, tiles='cartodb positron')

# ✅ GeoJSON Layer 추가
folium.GeoJson(
    gdf,
    name='200m Grid',
    style_function=lambda x: {
        'fillColor': '#3186cc',
        'color': 'black',
        'weight': 1,
        'fillOpacity': 0.4
    },
    tooltip=folium.GeoJsonTooltip(fields=[], aliases=[], labels=False)
).add_to(m)

# ✅ HTML 파일로 저장
m.save('./output/map_sosa_200m.html')

print("✅ 지도 저장 완료: './output/map_sosa_200m.html' 파일을 브라우저에서 열어보세요!")

