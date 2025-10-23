import streamlit as st
import pandas as pd
import numpy as np
import datetime
import io

# Mapping / spatial libs
import geopandas as gpd
import json
import requests
from pathlib import Path

# Viz
import plotly.express as px
import folium
from streamlit_folium import st_folium

st.set_page_config(layout="wide", page_title="Злочинність у регіонах")

# ---- Helper functions ----
@st.cache_data
def load_csv_from_url(url: str) -> pd.DataFrame:
    return pd.read_csv(url)

@st.cache_data
def load_csv(file) -> pd.DataFrame:
    return pd.read_csv(file)

def preprocess(df: pd.DataFrame, date_col='date', category_col='category', region_col='region'):
    # привідкнути назви колонок - можна налаштувати у UI
    df = df.copy()
    # приведення до нижнього регістру назв стовпців
    df.columns = [c.strip() for c in df.columns]
    # parse date
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col])
    # extract year/month
    df['year'] = df[date_col].dt.year
    df['month'] = df[date_col].dt.to_period('M').dt.to_timestamp()
    # ensure category and region present
    df[category_col] = df[category_col].astype(str)
    df[region_col] = df[region_col].astype(str)
    return df

def aggregate_by_region(df: pd.DataFrame, region_col='region', date_col='month'):
    agg = (df
           .groupby([region_col, date_col])
           .size()
           .reset_index(name='count'))
    return agg

def compute_crime_index(agg_df: pd.DataFrame, population_df: pd.DataFrame=None,
                        region_col='region', count_col='count', scale=100000):
    """
    Якщо population_df наданий і має стовпці region & population,
    повертаємо rate per scale (на scale людей, наприклад 100000).
    Інакше повертаємо нормалізований індекс 0-100 за минулий період.
    """
    df = agg_df.copy()
    # агрегація за регіоном (усього за весь період)
    total = df.groupby(region_col)[count_col].sum().reset_index()
    total = total.rename(columns={count_col: 'total_crimes'})
    if population_df is not None:
        pop = population_df.copy()
        pop.columns = [c.strip() for c in pop.columns]
        # припускаємо колонки 'region' і 'population'
        pop = pop.rename(columns={pop.columns[0]:'region', pop.columns[1]:'population'})
        merged = total.merge(pop, on='region', how='left')
        # якщо population пропущений -> залишаємо NaN
        merged['crime_rate_per_100k'] = merged['total_crimes'] / merged['population'] * scale
        # для зручності нормалізуємо в 0-100
        minv = merged['crime_rate_per_100k'].min()
        maxv = merged['crime_rate_per_100k'].max()
        if pd.isna(minv) or pd.isna(maxv) or minv==maxv:
            merged['crime_index'] = np.nan
        else:
            merged['crime_index'] = (merged['crime_rate_per_100k'] - minv) / (maxv - minv) * 100
        return merged
    else:
        # без population: нормалізація total_crimes в 0-100
        minv = total['total_crimes'].min()
        maxv = total['total_crimes'].max()
        if minv==maxv:
            total['crime_index'] = 50.0
        else:
            total['crime_index'] = (total['total_crimes'] - minv) / (maxv - minv) * 100
        return total

@st.cache_data
def load_geojson(url_or_path: str):
    try:
        if str(url_or_path).startswith("http"):
            gdf = gpd.read_file(url_or_path)
        else:
            gdf = gpd.read_file(str(url_or_path))
        return gdf
    except Exception as e:
        st.error(f"Не вдалося завантажити GeoJSON: {e}")
        return None

# ---- UI ----
st.sidebar.header("Джерело даних")
data_source = st.sidebar.radio("Виберіть джерело даних", ["Завантажити CSV", "Завантажити з URL", "Завантажити приклад"])

if data_source == "Завантажити CSV":
    uploaded_file = st.sidebar.file_uploader("CSV файл (потрібні стовпці: category, region, date)", type=['csv'])
    if uploaded_file is None:
        st.sidebar.info("Завантажте CSV або оберіть інший варіант.")
        st.stop()
    df = load_csv(uploaded_file)
elif data_source == "Завантажити з URL":
    url = st.sidebar.text_input("URL до CSV (пряме посилання)", "")
    if not url:
        st.sidebar.info("Вставте URL до CSV")
        st.stop()
    try:
        df = load_csv_from_url(url)
    except Exception as e:
        st.sidebar.error(f"Не вдалося завантажити CSV: {e}")
        st.stop()
else:
    # генеруємо прикладні дані
    st.sidebar.write("Генерую прикладні дані (синтетика).")
    rng = pd.date_range(end=pd.Timestamp.today(), periods=365)
    cats = ['Крадіжка', 'Шахрайство', 'Насильство', 'Побутове', 'Наркотики']
    regions = ['РегІон A','РегІон B','РегІон C','РегІон D']
    rows = []
    np.random.seed(0)
    for d in rng:
        for r in regions:
            # кількість за день
            n = np.random.poisson(lam=1.5)
            for i in range(n):
                rows.append({
                    'category': np.random.choice(cats, p=[0.4,0.2,0.15,0.15,0.1]),
                    'region': r,
                    'date': d + pd.to_timedelta(np.random.randint(0,24), unit='h')
                })
    df = pd.DataFrame(rows)

# колонка назви (дозволяємо користувачу вказати)
st.sidebar.markdown("---")
st.sidebar.header("Стовпці у файлі")
date_col = st.sidebar.text_input("Назва колонки з датою", value='date')
category_col = st.sidebar.text_input("Назва колонки з категорією", value='category')
region_col = st.sidebar.text_input("Назва колонки з регіоном", value='region')

# опційно population file
st.sidebar.markdown("---")
st.sidebar.header("Опційно: населення / GeoJSON")
population_file = st.sidebar.file_uploader("CSV з двома стовпцями: region, population (опційно)", type=['csv'])
geojson_source = st.sidebar.text_input("URL або шлях до GeoJSON для регіонів (опційно)", "")

# preprocess
df = preprocess(df, date_col=date_col, category_col=category_col, region_col=region_col)

# фільтри
st.header("Злочинність у регіонах — інтерактивний дашборд")
st.write("Фільтруйте дані, дивіться карту та динаміку. Можна завантажити результат.")

col1, col2 = st.columns([1,3])

with col1:
    st.subheader("Фільтри")
    min_date = df[date_col].min().date()
    max_date = df[date_col].max().date()
    date_range = st.date_input("Діапазон дат", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    selected_categories = st.multiselect("Категорії", options=sorted(df[category_col].unique()), default=sorted(df[category_col].unique()))
    selected_regions = st.multiselect("Регіони", options=sorted(df[region_col].unique()), default=sorted(df[region_col].unique()))
    btn_apply = st.button("Застосувати фільтри")

if not btn_apply:
    st.info("Натисніть 'Застосувати фільтри' щоб побачити оновлення.")
    st.stop()

# застосувати фільтри
start_dt = pd.to_datetime(date_range[0])
end_dt = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
mask = (df[date_col] >= start_dt) & (df[date_col] <= end_dt) & (df[category_col].isin(selected_categories)) & (df[region_col].isin(selected_regions))
df_f = df[mask].copy()

st.write(f"Показано записів: **{len(df_f)}** (з {len(df)} у вихідному наборі)")

# агрегація за місяцем
agg = aggregate_by_region(df_f, region_col=region_col, date_col='month')

# time series загалом та по категоріях
st.subheader("Динаміка злочинів")
ts = df_f.groupby(pd.Grouper(key=date_col, freq='W'))[category_col].count().reset_index(name='count')
fig_ts = px.line(ts, x=date_col, y='count', title='Загальна динаміка злочинності (по тижнях)')
st.plotly_chart(fig_ts, use_container_width=True)

# time series по категоріям (stacked)
ts_cat = (df_f
          .groupby([pd.Grouper(key=date_col, freq='M'), category_col])
          .size().reset_index(name='count'))
fig_stack = px.area(ts_cat, x=date_col, y='count', color=category_col, title='Динаміка по категоріям (місяці)')
st.plotly_chart(fig_stack, use_container_width=True)

# обчислюємо індекс криміногенності
pop_df = None
if population_file is not None:
    try:
        pop_df = pd.read_csv(population_file)
    except Exception as e:
        st.warning(f"Не вдалося прочитати population CSV: {e}")

index_df = compute_crime_index(agg, population_df=pop_df, region_col=region_col, count_col='count')

st.subheader("Індекс криміногенності (регіони)")
st.dataframe(index_df.sort_values('crime_index', ascending=False).reset_index(drop=True))

# карта
st.subheader("Карта: індекс / щільність злочинів")
map_col1, map_col2 = st.columns([1,1])
with map_col1:
    if geojson_source:
        gdf = load_geojson(geojson_source)
        if gdf is not None:
            # Спробуємо знайти колонку з назвою регіону у geojson: common names
            possible_names = ['name','NAME','region','region_name','adm1_name']
            gdf_cols = [c for c in gdf.columns if c.lower() in [p.lower() for p in possible_names]]
            if len(gdf_cols)==0:
                # залишимо першу не-geometry колонку як назву
                name_col = [c for c in gdf.columns if c!='geometry'][0]
            else:
                name_col = gdf_cols[0]
            # merge with index_df
            merged = gdf.merge(index_df, left_on=name_col, right_on=region_col, how='left')
            # create folium map
            m = folium.Map(location=[48.3794, 31.1656], zoom_start=6)  # координата України як приклад
            folium.Choropleth(
                geo_data=json.loads(merged.to_json()),
                name='choropleth',
                data=merged,
                columns=[name_col, 'crime_index'],
                key_on=f'feature.properties.{name_col}',
                fill_opacity=0.7,
                line_opacity=0.2,
                legend_name='Індекс криміногенності (0-100)'
            ).add_to(m)
            st_folium(m, width=700, height=450)
        else:
            st.write("Не вдалося показати карту з GeoJSON.")
    else:
        # якщо є координати у df_f — використаємо їх
        if {'latitude','longitude'}.issubset(df_f.columns) or {'lat','lon'}.issubset(df_f.columns):
            lat_col = 'latitude' if 'latitude' in df_f.columns else 'lat'
            lon_col = 'longitude' if 'longitude' in df_f.columns else 'lon'
            # агрегуємо по точці
            pts = df_f.groupby([lat_col, lon_col]).size().reset_index(name='count')
            m = folium.Map(location=[pts[lat_col].mean(), pts[lon_col].mean()], zoom_start=6)
            for _, row in pts.iterrows():
                folium.CircleMarker(location=[row[lat_col], row[lon_col]],
                                    radius=3 + np.log1p(row['count']),
                                    popup=f"count: {row['count']}").add_to(m)
            st_folium(m, width=700, height=450)
        else:
            st.info("Щоб побачити хлороплет/точкову карту, додайте GeoJSON у бічній панелі або стовпці latitude & longitude у CSV.")
            # Показати табличку агрегації замість карти
            region_total = agg.groupby(region_col)['count'].sum().reset_index().sort_values('count', ascending=False)
            fig_bar = px.bar(region_total, x=region_col, y='count', title='Загальна кількість злочинів по регіонах')
            st.plotly_chart(fig_bar, use_container_width=True)

with map_col2:
    st.subheader("Таблиця: деталі по регіонах")
    st.dataframe(index_df.sort_values('crime_index', ascending=False).reset_index(drop=True))

# Завантаження результату
st.subheader("Експорт")
to_download = st.button("Завантажити відфільтровані дані CSV")
if to_download:
    csv_bytes = df_f.to_csv(index=False).encode('utf-8')
    st.download_button("Клік для скачування", data=csv_bytes, file_name="filtered_crime_data.csv", mime="text/csv")

st.markdown("---")
st.write("Поради: щоб отримати більш інформативний індекс, додайте CSV з населенням по регіонах (region,population) або GeoJSON із контурами регіонів.")

