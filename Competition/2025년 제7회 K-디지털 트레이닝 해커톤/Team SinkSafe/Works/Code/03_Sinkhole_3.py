import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import folium
from folium import Choropleth, LayerControl, Marker, Popup
from streamlit_folium import st_folium
from shapely.geometry import Point
from geopy.distance import geodesic

st.set_page_config(layout="wide")
st.title("🛰 SinkSafe - 소사역 중심 반경 1.5km 위험 예측 지도")

# ✅ 격자 데이터 불러오기
shp_path = "./sinkhole/경기도 부천시_국가지점번호 기반_임의 격자 데이터_20221220/국가지점번호_기반_격자.shp"
gdf = gpd.read_file(shp_path)

# ✅ 좌표계 WGS84로 변환 (Folium용)
gdf = gdf.to_crs(epsg=4326)

# ✅ 중심 좌표 기준: 소사역
center_lat = 37.4826
center_lon = 126.7958
center_point = (center_lat, center_lon)

# ✅ 중심 좌표 및 위경도 추가
gdf["centroid"] = gdf.geometry.centroid
gdf["lat"] = gdf.centroid.y
gdf["lon"] = gdf.centroid.x

# ✅ 반경 1.5km 이내 격자 필터링
def is_within_radius(row, center, radius_km=1.5):
    point = (row['lat'], row['lon'])
    return geodesic(center, point).km <= radius_km

gdf = gdf[gdf.apply(lambda row: is_within_radius(row, center_point), axis=1)]

# ✅ 임의 위험 점수 생성 (실제론 모델 예측 결과 사용)
np.random.seed(42)
gdf["risk_score"] = np.random.rand(len(gdf))

# ✅ 지도 초기 설정
m = folium.Map(location=[center_lat, center_lon], zoom_start=14, tiles="cartodb positron")

# ✅ 위험도 색상 설정 함수
def get_color(score):
    if score > 0.75:
        return "red"
    elif score > 0.5:
        return "orange"
    elif score > 0.25:
        return "yellow"
    else:
        return "green"

# ✅ 격자 표시
for _, row in gdf.iterrows():
    sim_geo = gpd.GeoSeries(row["geometry"]).simplify(0.001)
    geo_json = sim_geo.to_json()
    folium.GeoJson(
        data=geo_json,
        style_function=lambda x, color=get_color(row["risk_score"]): {
            "fillColor": color,
            "color": "black",
            "weight": 0.5,
            "fillOpacity": 0.6,
        },
        tooltip=folium.Tooltip(
            f"<b>격자ID:</b> {row['id'] if 'id' in row else 'N/A'}<br>"
            f"<b>위험도:</b> {row['risk_score']:.2f}"
        ),
    ).add_to(m)

# ✅ Streamlit 지도 출력
st_data = st_folium(m, width=1000, height=700)

