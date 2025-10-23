import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Злочинність у регіонах", layout="wide")

# ---- Завантаження даних ----
@st.cache_data
def load_csv(filelike):
    return pd.read_csv(filelike)

@st.cache_data
def load_csv_from_url(url: str):
    return pd.read_csv(url)

# ---- Попередня обробка ----
def preprocess(df, date_col='date', category_col='category', region_col='region'):
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    if date_col not in df.columns:
        raise KeyError(f"Колонка '{date_col}' не знайдена у CSV.")
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col])
    # створюємо текстові колонки, якщо відсутні
    if category_col not in df.columns:
        df[category_col] = 'Unknown'
    if region_col not in df.columns:
        df[region_col] = 'Unknown'
    df[category_col] = df[category_col].astype(str)
    df[region_col] = df[region_col].astype(str)
    df['month'] = df[date_col].dt.to_period('M').dt.to_timestamp()
    return df

def compute_crime_index(df, region_col='region'):
    agg = df.groupby(region_col).size().reset_index(name='total_crimes')
    minv = agg['total_crimes'].min()
    maxv = agg['total_crimes'].max()
    if pd.isna(minv) or pd.isna(maxv) or minv == maxv:
        agg['crime_index'] = 50.0
    else:
        agg['crime_index'] = (agg['total_crimes'] - minv) / (maxv - minv) * 100
    return agg

# ---- UI: джерело даних ----
st.sidebar.header("Джерело даних")
source = st.sidebar.radio("Оберіть:", ["Прикладні дані", "Завантажити CSV", "Завантажити з URL"])

if source == "Прикладні дані":
    # синтетика для тесту
    rng = pd.date_range(end=pd.Timestamp.today(), periods=365)
    cats = ['Крадіжка', 'Шахрайство', 'Насильство', 'Наркотики']
    regions = ['Київська', 'Львівська', 'Одеська', 'Харківська', 'Дніпропетровська']
    np.random.seed(0)
    rows = []
    for d in rng:
        for r in regions:
            n = np.random.poisson(lam=1.8)
            for _ in range(n):
                rows.append({
                    'category': np.random.choice(cats),
                    'region': r,
                    'date': d + pd.to_timedelta(np.random.randint(0,24), unit='h')
                })
    df = pd.DataFrame(rows)

elif source == "Завантажити CSV":
    uploaded = st.sidebar.file_uploader("Завантажте CSV (має бути колонка з датою)", type=["csv"])
    if not uploaded:
        st.stop()
    df = load_csv(uploaded)

else:  # URL
    url = st.sidebar.text_input("URL до CSV (пряме посилання)")
    if not url:
        st.stop()
    df = load_csv_from_url(url)

# ---- Налаштування назв стовпців ----
st.sidebar.markdown("---")
st.sidebar.header("Налаштування стовпців")
date_col = st.sidebar.text_input("Колонка з датою", "date")
category_col = st.sidebar.text_input("Колонка з категорією", "category")
region_col = st.sidebar.text_input("Колонка з регіоном", "region")

# ---- Обробка даних і фільтри ----
try:
    df = preprocess(df, date_col=date_col, category_col=category_col, region_col=region_col)
except Exception as e:
    st.error(f"Помилка при обробці: {e}")
    st.stop()

st.title("📊 Злочинність у регіонах — дашборд")
min_date = df[date_col].min().date()
max_date = df[date_col].max().date()
date_range = st.date_input("Діапазон дат", value=(min_date, max_date), min_value=min_date, max_value=max_date)

selected_cats = st.multiselect("Категорії", options=sorted(df[category_col].unique()), default=sorted(df[category_col].unique()))
selected_regs = st.multiselect("Регіони", options=sorted(df[region_col].unique()), default=sorted(df[region_col].unique()))

start_dt = pd.to_datetime(date_range[0])
end_dt = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

mask = (
    (df[date_col] >= start_dt) &
    (df[date_col] <= end_dt) &
    (df[category_col].isin(selected_cats)) &
    (df[region_col].isin(selected_regs))
)
df_f = df[mask].copy()
st.write(f"Показано записів: **{len(df_f)}** з {len(df)}")

# ---- Динаміка (вбудовані графіки streamlit) ----
st.subheader("Динаміка злочинів (по місяцях)")
trend = df_f.groupby(df_f['month']).size()
if len(trend) == 0:
    st.info("Немає даних у вибраному діапазоні.")
else:
    st.line_chart(trend)

st.subheader("Розподіл по категоріях")
cat_counts = df_f[category_col].value_counts()
st.bar_chart(cat_counts)

# ---- Індекс криміногенності ----
st.subheader("Індекс криміногенності по регіонах")
index_df = compute_crime_index(df_f, region_col=region_col)
st.dataframe(index_df.sort_values('crime_index', ascending=False).reset_index(drop=True), use_container_width=True)

# ---- Таблиця з прикладами записів ----
st.subheader("Приклади записів")
st.dataframe(df_f.head(200), use_container_width=True)

# ---- Експорт ----
st.subheader("Експорт відфільтрованих даних")
csv = df_f.to_csv(index=False).encode('utf-8')
st.download_button("⬇️ Завантажити CSV", data=csv, file_name="filtered_crime_data.csv", mime="text/csv")

st.markdown("---")
st.caption("Згенеровано з допомогою GPT Online — https://gptonline.ai/")
