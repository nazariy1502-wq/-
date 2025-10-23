import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import folium
from streamlit_folium import st_folium
import requests
import json
import datetime

st.set_page_config(page_title="Злочинність у регіонах", layout="wide")

# ---- Кешування даних ----
@st.cache_data
def load_csv(file):
    return pd.read_csv(file)

@st.cache_data
def load_csv_from_url(url: str):
    return pd.read_csv(url)

@st.cache_data
def load_geojson(url_or_path: str):
    """Завантаження GeoJSON без geopandas."""
    try:
        if str(url_or_path).startswith("http"):
            response = requests.get(url_or_path)
            response.raise_for_status()
            return response.json()
        else:
            with open(url_or_path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        st.error(f"Не вдалося завантажити GeoJSON: {e}")
        return None

# ---- Функції ----
def preprocess(df, date_col='date', category_col='category', region_col='region'):
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col])
    df[category_col] = df[category_col].astype(str)
    df[region_col] = df[region_col].astype(str)
    df['month'] = df[date_col].dt.to_period('M').dt.to_timestamp()
    return df

def aggregate_by_region(df, region_col='region', date_col='month'):
    return (df.groupby([region_col, date_col])
              .size()
              .reset_index(name='count'))

def compute_crime_index(agg_df, population_df=None, region_col='region', count_col='count', scale=100000):
    df = agg_df.groupby(region_col)[count_col].sum().reset_index()
    df = df.rename(columns={count_col: 'total_crimes'})
    if population_df is not None:
        pop = population_df.rename(columns={pop.columns[0]:'region', pop.columns[1]:'population'})
        merged = df.merge(pop, on='region', how='left')
        merged['crime_rate'] = merged['total_crimes'] / merged['population'] * scale
        minv, maxv = merged['crime_rate'].min(), merged['crime_rate'].max()
        merged['crime_index'] = (merged['crime_rate'] - minv) / (maxv - minv) * 100
        return merged
    else:
        minv, maxv = df['total_crimes'].min(), df['total_crimes'].max()
        df['crime_index'] = (df['total_crimes'] - minv) / (maxv - minv) * 100
        return df

# ---- Інтерфейс ----
st.sidebar.header("Джерело даних")
option = st.sidebar.radio("Оберіть спосіб завантаження:", ["Прикладні дані", "CSV-файл", "URL"])

if option == "Прикладні дані":
    rng = pd.date_range(end=pd.Timestamp.today(), periods=365)
    categories = ['Крадіжка', 'Шахрайство', 'Насильство', 'Наркотики']
    regions = ['Київ', 'Львів', 'Одеса', 'Харків', 'Дніпро']
    np.random.seed(42)
    data = []
    for d in rng:
        for r in regions:
            n = np.random.poisson(lam=2)
            for _ in range(n):
                data.append({
                    'category': np.random.choice(categories),
                    'region': r,
                    'date': d + pd.to_timedelta(np.random.randint(0, 24), unit='h')
                })
    df = pd.DataFrame(data)
elif option == "CSV-файл":
    uploaded = st.sidebar.file_uploader("Завантажте CSV", type="csv")
    if uploaded is None:
        st.stop()
    df = load_csv(uploaded)
else:
    url = st.sidebar.text_input("URL до CSV")
    if not url:
        st.stop()
    df = load_csv_from_url(url)

# ---- Обробка ----
date_col = st.sidebar.text_input("Колонка з датою", "date")
cat_col = st.sidebar.text_input("Колонка з категорією", "category")
reg_col = st.sidebar.text_input("Колонка з регіоном", "region")

df = preprocess(df, date_col, cat_col, reg_col)

# ---- Фільтри ----
st.title("📊 Злочинність у регіонах України")
st.markdown("Аналітичний дашборд із візуалізацією злочинів за категоріями, регіонами та часом.")

min_d, max_d = df[date_col].min().date(), df[date_col].max().date()
sel_date = st.slider("Виберіть період:", min_d, max_d, (min_d, max_d))
sel_cat = st.multiselect("Категорії:", sorted(df[cat_col].unique()), default=sorted(df[cat_col].unique()))
sel_reg = st.multiselect("Регіони:", sorted(df[reg_col].unique()), default=sorted(df[reg_col].unique()))

mask = (
    (df[date_col].dt.date >= sel_date[0]) &
    (df[date_col].dt.date <= sel_date[1]) &
    (df[cat_col].isin(sel_cat)) &
    (df[reg_col].isin(sel_reg))
)
df_f = df[mask]

st.write(f"🔎 Відібрано {len(df_f)} записів з {len(df)}")

# ---- Динаміка злочинів ----
ts = df_f.groupby(pd.Grouper(key=date_col, freq='W'))[cat_col].count().reset_index(name='count')
fig_ts = px.line(ts, x=date_col, y='count', title="Динаміка злочинів (тижнево)")
st.plotly_chart(fig_ts, use_container_width=True)

ts_cat = df_f.groupby([pd.Grouper(key=date_col, freq='M'), cat_col]).size().reset_index(name='count')
fig_cat = px.area(ts_cat, x=date_col, y='count', color=cat_col, title="Розподіл злочинів за категоріями (місяці)")
st.plotly_chart(fig_cat, use_container_width=True)

# ---- Індекс криміногенності ----
agg = aggregate_by_region(df_f, reg_col, 'month')
index_df = compute_crime_index(agg, region_col=reg_col)
st.subheader("📈 Індекс криміногенності по регіонах")
st.dataframe(index_df.sort_values("crime_index", ascending=False), use_container_width=True)

# ---- Карта ----
geojson_url = st.sidebar.text_input("URL або шлях до GeoJSON (опціонально)")
if geojson_url:
    geojson_data = load_geojson(geojson_url)
    if geojson_data:
        m = folium.Map(location=[48.3794, 31.1656], zoom_start=6)
        folium.Choropleth(
            geo_data=geojson_data,
            data=index_df,
            columns=['region', 'crime_index'],
            key_on='feature.properties.name',  # змініть поле під ваш GeoJSON
            fill_color='YlOrRd',
            fill_opacity=0.7,
            line_opacity=0.2,
            legend_name='Індекс криміногенності (0–100)'
        ).add_to(m)
        st_folium(m, width=700, height=500)
else:
    fig_bar = px.bar(index_df, x='region', y='crime_index', color='crime_index',
                     color_continuous_scale='Reds', title="Індекс криміногенності (бар-графік)")
    st.plotly_chart(fig_bar, use_container_width=True)

# ---- Експорт ----
csv = df_f.to_csv(index=False).encode('utf-8')
st.download_button("⬇️ Завантажити відфільтровані дані CSV", data=csv, file_name="crime_filtered.csv", mime="text/csv")

st.markdown("---")
st.caption("Створено за допомогою GPT Online — https://gptonline.ai/")
