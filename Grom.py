import streamlit as st
import pandas as pd
import numpy as np
import json
import requests
import datetime
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="Злочинність у регіонах", layout="wide")

# -----------------------------
# 📦 Завантаження даних
# -----------------------------
@st.cache_data
def load_csv(filelike):
    return pd.read_csv(filelike)

@st.cache_data
def load_csv_from_url(url: str):
    return pd.read_csv(url)

@st.cache_data
def load_geojson(url_or_path: str):
    try:
        if str(url_or_path).startswith("http"):
            r = requests.get(url_or_path, timeout=10)
            r.raise_for_status()
            return r.json()
        else:
            with open(url_or_path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        st.error(f"Не вдалося завантажити GeoJSON: {e}")
        return None

# -----------------------------
# ⚙️ Попередня обробка
# -----------------------------
def preprocess(df, date_col='date', category_col='category', region_col='region'):
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col])
    df[category_col] = df[category_col].astype(str)
    df[region_col] = df[region_col].astype(str)
    df['month'] = df[date_col].dt.to_period('M').dt.to_timestamp()
    return df

def compute_crime_index(df, region_col='region', date_col='month'):
    agg = df.groupby([region_col, date_col]).size().reset_index(name='count')
    total = agg.groupby(region_col)['count'].sum().reset_index()
    minv, maxv = total['count'].min(), total['count'].max()
    total['crime_index'] = (
        (total['count'] - minv) / (maxv - minv) * 100 if maxv != minv else 50
    )
    return total

# -----------------------------
# 🧭 Інтерфейс користувача
# -----------------------------
st.sidebar.header("📊 Джерело даних")
source = st.sidebar.radio("Оберіть:", ["Прикладні дані", "Завантажити CSV", "Завантажити з URL"])

if source == "Прикладні дані":
    # генерація синтетичних даних
    rng = pd.date_range(end=pd.Timestamp.today(), periods=365)
    cats = ['Крадіжка', 'Шахрайство', 'Насильство', 'Наркотики']
    regions = ['Київська', 'Львівська', 'Одеська', 'Харківська', 'Дніпропетровська']
    np.random.seed(42)
    data = []
    for d in rng:
        for r in regions:
            n = np.random.poisson(lam=2)
            for _ in range(n):
                data.append({
                    'category': np.random.choice(cats),
                    'region': r,
                    'date': d + pd.to_timedelta(np.random.randint(0, 24), unit='h')
                })
    df = pd.DataFrame(data)

elif source == "Завантажити CSV":
    uploaded = st.sidebar.file_uploader("Завантажте CSV (category, region, date)", type=["csv"])
    if not uploaded:
        st.stop()
    df = load_csv(uploaded)

else:
    url = st.sidebar.text_input("Введіть URL до CSV")
    if not url:
        st.stop()
    df = load_csv_from_url(url)

# -----------------------------
# 🧹 Підготовка
# -----------------------------
st.sidebar.markdown("---")
st.sidebar.header("Налаштування стовпців")
date_col = st.sidebar.text_input("Колонка з датою", "date")
category_col = st.sidebar.text_input("Колонка з категорією", "category")
region_col = st.sidebar.text_input("Колонка з регіоном", "region")

df = preprocess(df, date_col, category_col, region_col)

# -----------------------------
# 🧩 Фільтри
# -----------------------------
st.title("🚓 Злочинність у регіонах")
min_date, max_date = df[date_col].min().date(), df[date_col].max().date()
date_range = st.date_input("Діапазон дат", (min_date, max_date))

selected_categories = st.multiselect("Категорії", df[category_col].unique(), df[category_col].unique())
selected_regions = st.multiselect("Регіони", df[region_col].unique(), df[region_col].unique())

mask = (
    (df[date_col] >= pd.to_datetime(date_range[0])) &
    (df[date_col] <= pd.to_datetime(date_range[1])) &
    (df[category_col].isin(selected_categories)) &
    (df[region_col].isin(selected_regions))
)
df_filtered = df[mask]

st.write(f"Відфільтровано {len(df_filtered)} записів")

# -----------------------------
# 📈 Візуалізація динаміки
# -----------------------------
st.subheader("📅 Динаміка злочинів")
trend = df_filtered.groupby(df_filtered[date_col].dt.to_period('M')).size()
st.line_chart(trend)

st.subheader("📂 Розподіл за категоріями")
cat_count = df_filtered[category_col].value_counts()
st.bar_chart(cat_count)

# -----------------------------
# 🔢 Індекс криміногенності
# -----------------------------
st.subheader("💥 Індекс криміногенності по регіонах")
index_df = compute_crime_index(df_filtered, region_col, date_col)
st.dataframe(index_df.sort_values('crime_index', ascending=False), use_container_width=True)

# -----------------------------
# 🗺️ Карта
# -----------------------------
st.subheader("🗺️ Карта злочинності")
geojson_url = st.text_input("URL до GeoJSON карти (опціонально)")
if geojson_url:
    geojson_data = load_geojson(geojson_url)
    if geojson_data:
        m = folium.Map(location=[48.3794, 31.1656], zoom_start=6)
        folium.Choropleth(
            geo_data=geojson_data,
            name="choropleth",
            data=index_df,
            columns=[region_col, 'crime_index'],
            key_on="feature.properties.name",
            fill_color="YlOrRd",
            fill_opacity=0.7,
            line_opacity=0.2,
            legend_name="Індекс криміногенності (0–100)"
        ).add_to(m)
        st_folium(m, width=700, height=500)
else:
    region_summary = index_df.set_index(region_col)['crime_index']
    st.bar_chart(region_summary)

# -----------------------------
# 💾 Експорт
# -----------------------------
st.subheader("💾 Завантажити результати")
csv_data = df_filtered.to_csv(index=False).encode('utf-8')
st.download_button("⬇️ Завантажити CSV", data=csv_data, file_name="filtered_crime_data.csv", mime="text/csv")

st.markdown("---")
st.caption("Згенеровано з допомогою GPT Online — [https://gptonline.ai/](https://gptonline.ai/)")
