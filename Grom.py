import streamlit as st
import pandas as pd
import numpy as np
import json
import requests
import io
import datetime

# візуалізація
import matplotlib.pyplot as plt

# карта
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="Злочинність у регіонах", layout="wide")

# ---------- Кешовані завантажувачі ----------
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

# ---------- Допоміжні функції ----------
def preprocess(df, date_col='date', category_col='category', region_col='region'):
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    # якщо колонка з датою має іншу назву — користувач може змінити у UI
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col])
    df[category_col] = df[category_col].astype(str)
    df[region_col] = df[region_col].astype(str)
    df['month'] = df[date_col].dt.to_period('M').dt.to_timestamp()
    return df

def aggregate_by_region(df, region_col='region', date_col='month'):
    return df.groupby([region_col, date_col]).size().reset_index(name='count')

def compute_crime_index(agg_df, population_df=None, region_col='region', count_col='count', scale=100000):
    total = agg_df.groupby(region_col)[count_col].sum().reset_index().rename(columns={count_col:'total_crimes'})
    if population_df is not None:
        pop = population_df.copy()
        pop.columns = [c.strip() for c in pop.columns]
        pop = pop.rename(columns={pop.columns[0]:'region', pop.columns[1]:'population'})
        merged = total.merge(pop, on='region', how='left')
        # уникати ділення на 0
        merged['population'] = merged['population'].replace({0: np.nan})
        merged['crime_rate_per_100k'] = merged['total_crimes'] / merged['population'] * scale
        minv = merged['crime_rate_per_100k'].min()
        maxv = merged['crime_rate_per_100k'].max()
        if pd.isna(minv) or pd.isna(maxv) or minv == maxv:
            merged['crime_index'] = np.nan
        else:
            merged['crime_index'] = (merged['crime_rate_per_100k'] - minv) / (maxv - minv) * 100
        return merged
    else:
        minv = total['total_crimes'].min()
        maxv = total['total_crimes'].max()
        if minv == maxv:
            total['crime_index'] = 50.0
        else:
            total['crime_index'] = (total['total_crimes'] - minv) / (maxv - minv) * 100
        return total

def plot_time_series(df, date_col='date', freq='W'):
    ts = df.groupby(pd.Grouper(key=date_col, freq=freq)).size().reset_index(name='count')
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(ts[date_col], ts['count'], marker='o', linewidth=1)
    ax.set_title("Динаміка злочинів")
    ax.set_xlabel("Дата")
    ax.set_ylabel("Кількість")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig

def plot_category_stack(df, date_col='month', category_col='category'):
    pivot = df.groupby([date_col, category_col]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    pivot.plot.area(ax=ax)
    ax.set_title("Розподіл по категоріях (місяць)")
    ax.set_xlabel("Місяць")
    ax.set_ylabel("Кількість")
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
    fig.tight_layout()
    return fig

# ---------- Інтерфейс ----------
st.sidebar.header("Джерело даних")
source = st.sidebar.radio("Оберіть:", ["Прикладні дані", "Завантажити CSV", "Завантажити з URL"])

if source == "Прикладні дані":
    # синтетика для швидкого тесту
    rng = pd.date_range(end=pd.Timestamp.today(), periods=365)
    cats = ['Крадіжка', 'Шахрайство', 'Насильство', 'Наркотики']
    regions = ['Київська','Львівська','Одеська','Харківська','Дніпропетровська']
    np.random.seed(1)
    rows = []
    for d in rng:
        for r in regions:
            n = np.random.poisson(lam=1.8)
            for _ in range(n):
                rows.append({'category': np.random.choice(cats),
                             'region': r,
                             'date': d + pd.to_timedelta(np.random.randint(0,24), unit='h')})
    df = pd.DataFrame(rows)
elif source == "Завантажити CSV":
    uploaded = st.sidebar.file_uploader("CSV файл (category, region, date)", type=["csv"])
    if uploaded is None:
        st.stop()
    df = load_csv(uploaded)
else:
    url = st.sidebar.text_input("Пряме посилання на CSV (URL)")
    if not url:
        st.stop()
    df = load_csv_from_url(url)

# назви колонок (користувач може змінити)
st.sidebar.markdown("---")
st.sidebar.header("Налаштування стовпців")
date_col = st.sidebar.text_input("Колонка з датою:", value='date')
category_col = st.sidebar.text_input("Колонка з категорією:", value='category')
region_col = st.sidebar.text_input("Колонка з регіоном:", value='region')

# опційно населення/geojson
st.sidebar.markdown("---")
pop_file = st.sidebar.file_uploader("Опційно: CSV з населенням (region,population)", type=["csv"])
geojson_source = st.sidebar.text_input("Опційно: URL або шлях до GeoJSON (контури регіонів)")

# preprocess
try:
    df = preprocess(df, date_col=date_col, category_col=category_col, region_col=region_col)
except Exception as e:
    st.error(f"Помилка обробки даних: {e}")
    st.stop()

# фільтри
st.title("🔎 Злочинність у регіонах — дашборд")
min_date = df[date_col].min().date()
max_date = df[date_col].max().date()
date_range = st.date_input("Діапазон дат:", value=(min_date, max_date), min_value=min_date, max_value=max_date)

selected_categories = st.multiselect("Категорії:", options=sorted(df[category_col].unique()), default=sorted(df[category_col].unique()))
selected_regions = st.multiselect("Регіони:", options=sorted(df[region_col].unique()), default=sorted(df[region_col].unique()))

start_dt = pd.to_datetime(date_range[0])
end_dt = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

mask = (
    (df[date_col] >= start_dt) &
    (df[date_col] <= end_dt) &
    (df[category_col].isin(selected_categories)) &
    (df[region_col].isin(selected_regions))
)
df_f = df[mask].copy()
st.write(f"Показано записів: **{len(df_f)}** з {len(df)}")

# візуалізації — matplotlib
st.subheader("Динаміка злочинів")
fig_ts = plot_time_series(df_f, date_col=date_col, freq='W')
st.pyplot(fig_ts)

st.subheader("Розподіл за категоріями")
fig_cat = plot_category_stack(df_f, date_col='month', category_col=category_col)
st.pyplot(fig_cat)

# індекс криміногенності
agg = aggregate_by_region(df_f, region_col, date_col='month')
pop_df = None
if pop_file is not None:
    try:
        pop_df = pd.read_csv(pop_file)
    except Exception as e:
        st.warning(f"Не вдалося прочитати population CSV: {e}")

index_df = compute_crime_index(agg, population_df=pop_df, region_col=region_col, count_col='count')
st.subheader("Індекс криміногенності по регіонах")
st.dataframe(index_df.sort_values('crime_index', ascending=False).reset_index(drop=True), use_container_width=True)

# карта
st.subheader("Карта / хлороплет")
if geojson_source:
    geojson = load_geojson(geojson_source)
    if geojson:
        # Пробуємо виявити яке поле у feature.properties містить назву (варианти: name, NAME, region, region_name)
        # Якщо ключ не знайдено — користувач має відредагувати key_on у коді під свій GeoJSON.
        example_props = geojson.get('features', [{}])[0].get('properties', {})
        prop_keys = list(example_props.keys())
        # За замовчуванням підставимо 'name' якщо є, інакше перший ключ
        key_name = 'name' if 'name' in prop_keys else (prop_keys[0] if prop_keys else None)
        if key_name is None:
            st.warning("Не вдалося знайти поле з назвою регіону у GeoJSON (properties пусті).")
        else:
            m = folium.Map(location=[48.3794, 31.1656], zoom_start=6)
            # Для Choropleth потрібна таблиця у форматі [region, crime_index]
            if 'crime_index' in index_df.columns and region_col in index_df.columns:
                # Переконаємося, що назви збігаються: користувач має перевірити поле key_on
                try:
                    folium.Choropleth(
                        geo_data=geojson,
                        name='choropleth',
                        data=index_df,
                        columns=[region_col, 'crime_index'],
                        key_on=f'feature.properties.{key_name}',
                        fill_opacity=0.7,
                        line_opacity=0.2,
                        legend_name='Індекс криміногенності (0-100)'
                    ).add_to(m)
                    st_folium(m, width=700, height=500)
                except Exception as e:
                    st.error(f"Не вдалося побудувати хлороплет: {e}. Можливо, поле для зв'язку назв регіонів ('feature.properties.{key_name}') не збігається з колонкою region у таблиці.")
            else:
                st.info("Немає даних індексу для карти.")
else:
    # Якщо GeoJSON відсутній — показуємо просту кругову карту за координатами, якщо вони є
    if {'latitude','longitude'}.issubset(df_f.columns) or {'lat','lon'}.issubset(df_f.columns):
        lat_col = 'latitude' if 'latitude' in df_f.columns else 'lat'
        lon_col = 'longitude' if 'longitude' in df_f.columns else 'lon'
        pts = df_f.groupby([lat_col, lon_col]).size().reset_index(name='count')
        m = folium.Map(location=[pts[lat_col].mean(), pts[lon_col].mean()], zoom_start=6)
        for _, r in pts.iterrows():
            folium.CircleMarker(location=[r[lat_col], r[lon_col]],
                                radius=3 + np.log1p(r['count']),
                                popup=f"count: {r['count']}").add_to(m)
        st_folium(m, width=700, height=500)
    else:
        st.info("Щоб побачити карту з кордонами, додайте GeoJSON у бічній панелі, або додайте координати (latitude/longitude) у CSV.")
        # показати бар по регіонах (matplotlib)
        region_totals = agg.groupby(region_col)['count'].sum().reset_index().sort_values('count', ascending=False)
        fig, ax = plt.subplots(figsize=(8,3.5))
        ax.bar(region_totals[region_col], region_totals['count'])
        ax.set_title("Загальна кількість злочинів по регіонах")
        ax.set_xlabel("Регіон")
        ax.set_ylabel("Кількість")
        plt.xticks(rotation=45, ha='right')
        fig.tight_layout()
        st.pyplot(fig)

# експорт
st.subheader("Експорт відфільтрованих даних")
csv_bytes = df_f.to_csv(index=False).encode('utf-8')
st.download_button("Завантажити CSV", data=csv_bytes, file_name="filtered_crime_data.csv", mime="text/csv")

st.markdown("---")
st.caption("Згенеровано з допомогою GPT Online — https://gptonline.ai/")
