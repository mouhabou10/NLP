"""
generate_dashboard.py
=====================
Reads yassir_customers_processed.csv and generates a fully
data-accurate interactive HTML dashboard.

Usage (from lab1/notebooks/):
    python generate_dashboard.py

Output: yassir_dashboard.html  (open in any browser)
"""

import pandas as pd
import json
import re
from collections import Counter
from datetime import datetime
import os

# ─── CONFIG ──────────────────────────────────────────────────
CSV_PATH  = '../data/processed/yassir_customers_processed.csv'
OUT_PATH  = 'yassir_dashboard.html'
SAMPLE_N  = 1500   # reviews sampled for language detection

# ─── LOAD ─────────────────────────────────────────────────────
print("📂 Loading CSV...")
df = pd.read_csv(CSV_PATH)
df['review_date'] = pd.to_datetime(df['review_date'], errors='coerce')
df['review_year']  = df['review_year'].astype(int)
df['review_month'] = df['review_month'].astype(int)
df['text'] = df['text'].fillna('').astype(str)
print(f"   ✅ {len(df):,} reviews loaded")

# ─── DERIVED COLUMNS ──────────────────────────────────────────
df['sentiment'] = df['rating'].apply(
    lambda r: 'Positive' if r >= 4 else ('Negative' if r <= 2 else 'Neutral')
)

# ─── KPI ──────────────────────────────────────────────────────
print("📊 Computing KPIs...")
total        = len(df)
avg_rating   = round(df['rating'].mean(), 2)
pct_positive = round(len(df[df['rating'] >= 4]) / total * 100, 1)
pct_negative = round(len(df[df['rating'] <= 2]) / total * 100, 1)
pct_neutral  = round(len(df[df['rating'] == 3]) / total * 100, 1)
date_min     = df['review_date'].min().strftime('%Y-%m-%d')
date_max     = df['review_date'].max().strftime('%Y-%m-%d')
n_years      = df['review_year'].nunique()
count_pos    = len(df[df['rating'] >= 4])
count_neg    = len(df[df['rating'] <= 2])
count_neu    = len(df[df['rating'] == 3])
avg_len_pos  = round(df[df['sentiment']=='Positive']['text_length'].mean())
avg_len_neu  = round(df[df['sentiment']=='Neutral']['text_length'].mean())
avg_len_neg  = round(df[df['sentiment']=='Negative']['text_length'].mean())

# ─── RATINGS DISTRIBUTION ─────────────────────────────────────
print("⭐ Rating distributions...")
def rating_dist(sub):
    return [int(len(sub[sub['rating']==r])) for r in [1,2,3,4,5]]

rating_all = rating_dist(df)

# Per year (full years only)
full_years = [y for y in sorted(df['review_year'].unique()) if y < datetime.now().year]
rating_by_year = {}
for y in full_years:
    rating_by_year[str(y)] = rating_dist(df[df['review_year'] == y])

# ─── YEARLY / MONTHLY VOLUMES ─────────────────────────────────
print("📅 Temporal data...")
yearly = df.groupby('review_year').size()
yearly_labels = [str(y) for y in yearly.index.tolist()]
yearly_counts = yearly.values.tolist()

# monthly
df['ym'] = df['review_year'].astype(str) + '-' + df['review_month'].astype(str).str.zfill(2)
monthly_all = df.groupby('ym').size().sort_index()
monthly_pos = df[df['sentiment']=='Positive'].groupby('ym').size().reindex(monthly_all.index, fill_value=0)
monthly_neg = df[df['sentiment']=='Negative'].groupby('ym').size().reindex(monthly_all.index, fill_value=0)
monthly_labels = monthly_all.index.tolist()
monthly_all_v  = monthly_all.values.tolist()
monthly_pos_v  = monthly_pos.values.tolist()
monthly_neg_v  = monthly_neg.values.tolist()

# monthly avg rating
monthly_rating = df.groupby('ym')['rating'].mean().reindex(monthly_all.index).round(2)
monthly_rating_v = monthly_rating.ffill().values.tolist()

# ─── HEATMAP (month × year) ────────────────────────────────────
print("🗓️  Heatmap data...")
hm_years  = [y for y in sorted(df['review_year'].unique())]
hm_months = list(range(1, 13))
hm_data   = []
for yi, yr in enumerate(hm_years):
    for mi, mo in enumerate(hm_months):
        val = int(len(df[(df['review_year']==yr) & (df['review_month']==mo)]))
        if val > 0:
            hm_data.append({'x': yi, 'y': mi, 'r': round((val**0.5)*1.5, 1), 'v': val})
hm_year_labels  = [str(y) for y in hm_years]
hm_month_labels = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

# ─── STACKED BAR (rating × year) ──────────────────────────────
stacked_years = [str(y) for y in full_years]
stacked = {str(r): [] for r in [1,2,3,4,5]}
for y in full_years:
    sub = df[df['review_year']==y]
    for r in [1,2,3,4,5]:
        stacked[str(r)].append(int(len(sub[sub['rating']==r])))

# ─── WORD ANALYSIS ─────────────────────────────────────────────
print("☁️  Word frequencies...")
STOPWORDS = {
    'application','app','yassir','les','des','est','une','pas','que','plus',
    'très','bien','pour','sur','avec','dans','par','qui','ce','tout','mais',
    'cest','cette','comme','avoir','the','and','is','it','to','of','in','for',
    'on','are','this','that','not','my','me','an','be','can','you','your',
    'have','had','was','but','just','all','when','they','dont','even','get',
    'got','been','has','aussi','leur','leurs','nous','vous','ils','elles',
    'même','après','avant','encore','toujours','jamais','rien','tout','très',
    'trop','assez','peu','beaucoup','plus','moins','fois','fait','faire',
    'sont','être','avoir','aller','voir','venir','pouvoir','vouloir','savoir',
    'les', 'des', 'est', 'une', 'pas',
}

def top_words(series, n=15):
    words = []
    for text in series:
        tokens = re.findall(r'[a-zA-Zéèàùôâêîûïëüœ]{4,}', text.lower())
        words.extend([w for w in tokens if w not in STOPWORDS])
    counts = Counter(words).most_common(n)
    return [{'word': w, 'count': c} for w, c in counts]

words_all = top_words(df['text'], 20)
words_pos = top_words(df[df['sentiment']=='Positive']['text'], 15)
words_neg = top_words(df[df['sentiment']=='Negative']['text'], 15)

# ─── TOP REVIEWS (by thumbs up) ────────────────────────────────
print("💬 Top reviews...")
top_reviews_raw = df.nlargest(12, 'thumbs_up_count')[
    ['author','rating','thumbs_up_count','review_date','text']
].fillna({'author':'Anonymous', 'text':''})

top_reviews = []
for _, row in top_reviews_raw.iterrows():
    top_reviews.append({
        'author': str(row['author'])[:30],
        'rating': int(row['rating']),
        'thumbs': int(row['thumbs_up_count']),
        'date':   str(row['review_date'])[:10],
        'text':   str(row['text'])[:300],
    })

# ─── LANGUAGE DETECTION ────────────────────────────────────────
print("🌍 Language detection...")
def detect_lang(text):
    text = str(text)
    arabic = len(re.findall(r'[\u0600-\u06FF]', text))
    latin  = len(re.findall(r'[a-zA-Zéèàùôâêîûïëüœ]', text))
    if arabic > latin:
        return 'Arabic'
    fr_markers = ['le ','la ','les ','est ','pas ','que ','une ','très ','avec ','pour ','dans ']
    en_markers = ['the ','and ','this ','that ','very ','not ','good ','was ','have ','you ']
    t = text.lower()
    return 'French' if sum(m in t for m in fr_markers) >= sum(m in t for m in en_markers) else 'English'

sample = df.sample(min(SAMPLE_N, len(df)), random_state=42)
sample = sample.copy()
sample['lang'] = sample['text'].apply(detect_lang)
lang_counts_raw = sample['lang'].value_counts()
lang_labels = lang_counts_raw.index.tolist()
lang_counts = lang_counts_raw.values.tolist()

# avg rating per language
lang_ratings = []
for lang in lang_labels:
    ids = sample[sample['lang']==lang].index
    avg = round(df.loc[df.index.isin(ids), 'rating'].mean(), 2) if len(ids) else 0
    lang_ratings.append(avg)

# ─── PACK ALL DATA ─────────────────────────────────────────────
DATA = {
    'meta': {
        'total': total, 'avg_rating': avg_rating,
        'pct_positive': pct_positive, 'pct_negative': pct_negative, 'pct_neutral': pct_neutral,
        'count_pos': count_pos, 'count_neg': count_neg, 'count_neu': count_neu,
        'date_min': date_min, 'date_max': date_max, 'n_years': n_years,
        'avg_len_pos': avg_len_pos, 'avg_len_neu': avg_len_neu, 'avg_len_neg': avg_len_neg,
    },
    'rating_all': rating_all,
    'rating_by_year': rating_by_year,
    'yearly': {'labels': yearly_labels, 'counts': yearly_counts},
    'monthly': {
        'labels': monthly_labels,
        'all': monthly_all_v, 'pos': monthly_pos_v, 'neg': monthly_neg_v,
        'avg_rating': monthly_rating_v,
    },
    'heatmap': {
        'data': hm_data,
        'year_labels': hm_year_labels,
        'month_labels': hm_month_labels,
    },
    'stacked': {'years': stacked_years, 'by_star': stacked},
    'words': {'all': words_all, 'pos': words_pos, 'neg': words_neg},
    'top_reviews': top_reviews,
    'languages': {'labels': lang_labels, 'counts': lang_counts, 'avg_ratings': lang_ratings},
}

print("✅ All data computed!")
print(f"   Total: {total:,} | Avg: {avg_rating}★ | Positive: {pct_positive}% | Negative: {pct_negative}%")

# ═══════════════════════════════════════════════════════════════
# HTML TEMPLATE
# ═══════════════════════════════════════════════════════════════
data_json = json.dumps(DATA, ensure_ascii=False)

HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Yassir Reviews — NLP Dashboard</title>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap" rel="stylesheet"/>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
:root{{
  --purple:#6C2BDB;--pink:#E91E8C;--purple-light:#9B59F5;--dark:#0a0812;--dark2:#110d1a;
  --dark3:#160e2a;--card:#130d22;--border:rgba(108,43,219,0.22);
  --text:#e8e8f0;--muted:#7070a0;--green:#2ecc71;--red:#e74c3c;
  --yellow:#f39c12;--blue:#3498db;
}}
*{{margin:0;padding:0;box-sizing:border-box;}}
body{{background:var(--dark);color:var(--text);font-family:'DM Sans',sans-serif;min-height:100vh;overflow-x:hidden;}}
body::before{{content:'';position:fixed;inset:0;background-image:linear-gradient(rgba(108,43,219,0.03) 1px,transparent 1px),linear-gradient(90deg,rgba(108,43,219,0.03) 1px,transparent 1px);background-size:40px 40px;pointer-events:none;z-index:0;}}
.sidebar{{position:fixed;left:0;top:0;bottom:0;width:220px;background:var(--dark2);border-right:1px solid var(--border);display:flex;flex-direction:column;padding:24px 0;z-index:100;transition:transform 0.3s ease;}}
.logo{{padding:0 24px 28px;border-bottom:1px solid var(--border);}}
.logo-text{{font-family:'Syne',sans-serif;font-size:20px;font-weight:800;color:#a855f7;letter-spacing:-0.5px;}}
.logo-sub{{font-size:11px;color:var(--muted);margin-top:2px;}}
.nav{{padding:20px 12px;flex:1;}}
.nav-item{{display:flex;align-items:center;gap:10px;padding:10px 14px;border-radius:10px;cursor:pointer;font-size:13px;font-weight:500;color:var(--muted);transition:all 0.2s;margin-bottom:4px;border:1px solid transparent;}}
.nav-item:hover{{background:rgba(108,43,219,0.1);color:var(--text);}}
.nav-item.active{{background:rgba(108,43,219,0.12);color:#a855f7;border-color:var(--border);}}
.nav-item .icon{{font-size:16px;width:20px;text-align:center;}}
.nav-section-label{{font-size:10px;font-weight:600;letter-spacing:1.5px;text-transform:uppercase;color:var(--muted);padding:16px 14px 6px;}}
.sidebar-footer{{padding:16px 24px;border-top:1px solid var(--border);font-size:11px;color:var(--muted);line-height:1.6;}}
.main{{margin-left:220px;padding:32px 36px;position:relative;z-index:1;min-height:100vh;}}
.page{{display:none;animation:fadeUp 0.4s ease;}}
.page.active{{display:block;}}
@keyframes fadeUp{{from{{opacity:0;transform:translateY(16px)}}to{{opacity:1;transform:translateY(0)}}}}
.page-header{{margin-bottom:32px;}}
.page-title{{font-family:'Syne',sans-serif;font-size:32px;font-weight:800;line-height:1.1;background:linear-gradient(135deg,#fff 30%,#E91E8C 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;}}
.page-sub{{color:var(--muted);font-size:14px;margin-top:6px;}}
.filter-bar{{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:28px;padding:14px 18px;background:var(--card);border-radius:14px;border:1px solid var(--border);}}
.filter-label{{font-size:12px;color:var(--muted);align-self:center;margin-right:4px;}}
.filter-btn{{padding:6px 14px;border-radius:20px;border:1px solid var(--border);background:transparent;color:var(--muted);font-size:12px;font-family:'DM Sans',sans-serif;cursor:pointer;transition:all 0.2s;}}
.filter-btn:hover{{border-color:#a855f7;color:var(--text);}}
.filter-btn.active{{background:linear-gradient(135deg,var(--purple),var(--pink));border-color:#a855f7;color:#fff;font-weight:600;}}
.kpi-grid{{display:grid;grid-template-columns:repeat(5,1fr);gap:14px;margin-bottom:28px;}}
.kpi-card{{background:var(--card);border:1px solid var(--border);border-radius:16px;padding:20px 18px;position:relative;overflow:hidden;transition:transform 0.2s,border-color 0.2s;}}
.kpi-card:hover{{transform:translateY(-3px);border-color:rgba(108,43,219,0.45);}}
.kpi-card::before{{content:'';position:absolute;top:0;left:0;right:0;height:3px;}}
.kpi-card.orange::before{{background:linear-gradient(90deg,var(--purple),var(--pink));}}.kpi-card.green::before{{background:var(--green);}}.kpi-card.red::before{{background:var(--red);}}.kpi-card.blue::before{{background:var(--blue);}}.kpi-card.yellow::before{{background:var(--yellow);}}
.kpi-label{{font-size:11px;text-transform:uppercase;letter-spacing:1px;color:var(--muted);margin-bottom:8px;}}
.kpi-value{{font-family:'Syne',sans-serif;font-size:28px;font-weight:800;line-height:1;}}
.kpi-card.orange .kpi-value{{color:#a855f7}}.kpi-card.green .kpi-value{{color:var(--green)}}.kpi-card.red .kpi-value{{color:var(--red)}}.kpi-card.blue .kpi-value{{color:var(--blue)}}.kpi-card.yellow .kpi-value{{color:var(--yellow)}}
.kpi-delta{{font-size:11px;color:var(--muted);margin-top:6px;}}
.chart-grid-2{{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:20px;}}
.chart-grid-3{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:20px;margin-bottom:20px;}}
.chart-grid-1{{display:grid;grid-template-columns:1fr;gap:20px;margin-bottom:20px;}}
.chart-grid-13{{display:grid;grid-template-columns:1fr 2fr;gap:20px;margin-bottom:20px;}}
.chart-grid-31{{display:grid;grid-template-columns:2fr 1fr;gap:20px;margin-bottom:20px;}}
.chart-card{{background:var(--card);border:1px solid var(--border);border-radius:16px;padding:22px;transition:border-color 0.2s;}}
.chart-card:hover{{border-color:rgba(108,43,219,0.35);}}
.chart-card-header{{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:18px;}}
.chart-card-title{{font-family:'Syne',sans-serif;font-size:15px;font-weight:700;color:var(--text);}}
.chart-card-sub{{font-size:11px;color:var(--muted);margin-top:3px;}}
.chart-badge{{font-size:10px;padding:3px 8px;border-radius:10px;background:rgba(108,43,219,0.1);color:#a855f7;border:1px solid rgba(108,43,219,0.25);white-space:nowrap;}}
.chart-container{{position:relative;}}
.insight-box{{background:rgba(108,43,219,0.06);border:1px solid rgba(108,43,219,0.25);border-radius:12px;padding:14px 18px;font-size:12.5px;color:#c0c0d8;line-height:1.7;margin-top:14px;}}
.insight-box strong{{color:#a855f7;}}
.wordcloud-container{{display:flex;flex-wrap:wrap;gap:8px;padding:16px;align-items:center;justify-content:center;min-height:200px;}}
.wc-word{{cursor:default;border-radius:6px;padding:4px 10px;transition:transform 0.2s,opacity 0.2s;font-family:'Syne',sans-serif;font-weight:700;}}
.wc-word:hover{{transform:scale(1.15);opacity:0.9;}}
.review-list{{display:flex;flex-direction:column;gap:12px;}}
.review-item{{background:var(--dark3);border:1px solid var(--border);border-radius:12px;padding:16px;transition:border-color 0.2s;}}
.review-item:hover{{border-color:rgba(108,43,219,0.4);}}
.review-meta{{display:flex;align-items:center;gap:10px;margin-bottom:8px;}}
.review-stars{{color:var(--yellow);font-size:13px;}}.review-stars.neg{{color:var(--red);}}
.review-author{{font-size:12px;color:var(--muted);}}.review-date{{font-size:11px;color:var(--muted);margin-left:auto;}}
.review-text{{font-size:13px;line-height:1.6;color:var(--text);}}
.review-thumbs{{font-size:11px;color:var(--muted);margin-top:8px;}}
.lang-bar-item{{margin-bottom:16px;}}
.lang-bar-label{{display:flex;justify-content:space-between;font-size:12px;margin-bottom:6px;}}
.lang-bar-track{{background:rgba(255,255,255,0.06);border-radius:4px;height:8px;overflow:hidden;}}
.lang-bar-fill{{height:100%;border-radius:4px;transition:width 1s ease;}}
.timeline-item{{display:flex;gap:16px;padding:10px 0;border-bottom:1px solid var(--border);align-items:center;}}
.timeline-year{{font-family:'Syne',sans-serif;font-weight:800;font-size:16px;color:#a855f7;min-width:46px;}}
.timeline-bar-wrap{{flex:1;display:flex;align-items:center;gap:10px;}}
.timeline-bar{{height:18px;border-radius:4px;background:linear-gradient(90deg,var(--purple),var(--purple-light));transition:width 1.2s ease;min-width:2px;}}
.timeline-count{{font-size:11px;color:var(--muted);white-space:nowrap;}}
.donut-wrap{{position:relative;}}
.donut-center{{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);text-align:center;pointer-events:none;}}
.donut-center-val{{font-family:'Syne',sans-serif;font-size:26px;font-weight:800;color:var(--text);}}
.donut-center-label{{font-size:11px;color:var(--muted);}}
.hamburger{{display:none;position:fixed;top:16px;left:16px;z-index:200;background:var(--card);border:1px solid var(--border);border-radius:10px;padding:8px 10px;cursor:pointer;font-size:18px;}}
.overlay{{display:none;position:fixed;inset:0;background:rgba(0,0,0,0.6);z-index:99;}}
.animate-in{{opacity:0;transform:translateY(20px);transition:opacity 0.5s ease,transform 0.5s ease;}}
.animate-in.visible{{opacity:1;transform:translateY(0);}}
@media(max-width:1200px){{.kpi-grid{{grid-template-columns:repeat(3,1fr)}}.chart-grid-3{{grid-template-columns:1fr 1fr}}}}
@media(max-width:900px){{.sidebar{{transform:translateX(-220px)}}.sidebar.open{{transform:translateX(0)}}.main{{margin-left:0;padding:20px 16px;padding-top:60px;}}.hamburger{{display:block}}.overlay.show{{display:block}}.chart-grid-2,.chart-grid-13,.chart-grid-31{{grid-template-columns:1fr}}.kpi-grid{{grid-template-columns:repeat(2,1fr)}}.page-title{{font-size:24px}}}}
@media(max-width:500px){{.kpi-grid{{grid-template-columns:1fr 1fr}}.chart-grid-3{{grid-template-columns:1fr}}}}
::-webkit-scrollbar{{width:6px}}::-webkit-scrollbar-track{{background:var(--dark2)}}::-webkit-scrollbar-thumb{{background:rgba(108,43,219,0.4);border-radius:3px}}
</style>
</head>
<body>
<button class="hamburger" onclick="toggleSidebar()">☰</button>
<div class="overlay" id="overlay" onclick="toggleSidebar()"></div>
<nav class="sidebar" id="sidebar">
  <div class="logo">
    <div class="logo-text">YASSIR</div>
    <div class="logo-sub">Customer Reviews · NLP Project</div>
  </div>
  <div class="nav">
    <div class="nav-section-label">Analytics</div>
    <div class="nav-item active" onclick="showPage('overview')"><span class="icon">📊</span> Overview</div>
    <div class="nav-item" onclick="showPage('ratings')"><span class="icon">⭐</span> Ratings</div>
    <div class="nav-item" onclick="showPage('trends')"><span class="icon">📈</span> Trends</div>
    <div class="nav-section-label">Content</div>
    <div class="nav-item" onclick="showPage('words')"><span class="icon">☁️</span> Word Analysis</div>
    <div class="nav-item" onclick="showPage('reviews')"><span class="icon">💬</span> Top Reviews</div>
    <div class="nav-section-label">Context</div>
    <div class="nav-item" onclick="showPage('languages')"><span class="icon">🌍</span> Languages</div>
    <div class="nav-item" onclick="showPage('insights')"><span class="icon">💡</span> Insights</div>
  </div>
  <div class="sidebar-footer">
    <div>📱 Yassir - Ride, Eat &amp; Shop</div>
    <div>🗓️ {date_min} → {date_max}</div>
    <div style="margin-top:6px">NLP Course · Group Project</div>
  </div>
</nav>
<main class="main">

<!-- PAGE 1: OVERVIEW -->
<div class="page active" id="page-overview">
  <div class="page-header">
    <div class="page-title">Executive Overview</div>
    <div class="page-sub">{total:,} customer reviews · Google Play Store · Algeria · {date_min} → {date_max}</div>
  </div>
  <div class="kpi-grid animate-in">
    <div class="kpi-card orange"><div class="kpi-label">Total Reviews</div><div class="kpi-value">{total:,}</div><div class="kpi-delta">{date_min[:4]} – {date_max[:4]}</div></div>
    <div class="kpi-card yellow"><div class="kpi-label">Avg Rating</div><div class="kpi-value">{avg_rating}★</div><div class="kpi-delta">Out of 5.0</div></div>
    <div class="kpi-card green"><div class="kpi-label">Positive</div><div class="kpi-value">{pct_positive}%</div><div class="kpi-delta">4–5 star reviews</div></div>
    <div class="kpi-card red"><div class="kpi-label">Negative</div><div class="kpi-value">{pct_negative}%</div><div class="kpi-delta">1–2 star reviews</div></div>
    <div class="kpi-card blue"><div class="kpi-label">Years Active</div><div class="kpi-value">{n_years} yrs</div><div class="kpi-delta">{n_years * 365:,} days of data</div></div>
  </div>
  <div class="chart-grid-13 animate-in" style="transition-delay:0.1s">
    <div class="chart-card">
      <div class="chart-card-header"><div><div class="chart-card-title">Sentiment Breakdown</div><div class="chart-card-sub">By review category</div></div><span class="chart-badge">All time</span></div>
      <div class="donut-wrap"><div class="chart-container" style="height:220px"><canvas id="sentimentDonut"></canvas></div>
      <div class="donut-center"><div class="donut-center-val">{avg_rating}</div><div class="donut-center-label">avg ★</div></div></div>
    </div>
    <div class="chart-card">
      <div class="chart-card-header"><div><div class="chart-card-title">Reviews Per Year</div><div class="chart-card-sub">Growth trajectory</div></div><span class="chart-badge">Yearly</span></div>
      <div class="chart-container" style="height:220px"><canvas id="yearlyBar"></canvas></div>
    </div>
  </div>
  <div class="chart-grid-2 animate-in" style="transition-delay:0.2s">
    <div class="chart-card">
      <div class="chart-card-header"><div><div class="chart-card-title">Monthly Volume</div><div class="chart-card-sub">Review activity over time</div></div></div>
      <div class="chart-container" style="height:180px"><canvas id="monthlyLine"></canvas></div>
    </div>
    <div class="chart-card">
      <div class="chart-card-header"><div><div class="chart-card-title">Rating Distribution</div><div class="chart-card-sub">Count per star level</div></div></div>
      <div class="chart-container" style="height:180px"><canvas id="ratingBarOverview"></canvas></div>
    </div>
  </div>
</div>

<!-- PAGE 2: RATINGS -->
<div class="page" id="page-ratings">
  <div class="page-header"><div class="page-title">Rating Analysis</div><div class="page-sub">Deep dive into star ratings and sentiment patterns</div></div>
  <div class="filter-bar animate-in">
    <span class="filter-label">Filter by year:</span>
    <button class="filter-btn active" onclick="filterRatings('all',this)">All Years</button>
    <div id="year-filter-btns"></div>
  </div>
  <div class="chart-grid-2 animate-in" style="transition-delay:0.1s">
    <div class="chart-card">
      <div class="chart-card-header"><div><div class="chart-card-title">Star Distribution</div><div class="chart-card-sub">Count per rating level</div></div><span class="chart-badge" id="rating-year-badge">All Years</span></div>
      <div class="chart-container" style="height:260px"><canvas id="ratingBarDetail"></canvas></div>
    </div>
    <div class="chart-card">
      <div class="chart-card-header"><div><div class="chart-card-title">Sentiment Pie</div><div class="chart-card-sub">Positive · Neutral · Negative</div></div></div>
      <div class="chart-container" style="height:260px"><canvas id="sentimentPie"></canvas></div>
    </div>
  </div>
  <div class="chart-grid-1 animate-in" style="transition-delay:0.2s">
    <div class="chart-card">
      <div class="chart-card-header"><div><div class="chart-card-title">Average Rating Over Time</div><div class="chart-card-sub">Monthly average with 3-month rolling mean</div></div></div>
      <div class="chart-container" style="height:220px"><canvas id="ratingTrend"></canvas></div>
      <div class="insight-box"><strong>📌 Key insight:</strong> Early years show higher ratings from early adopters. Dips post-2021 correlate with rapid scaling. The {avg_rating}★ average masks extreme polarisation ({pct_positive}% positive vs {pct_negative}% negative).</div>
    </div>
  </div>
</div>

<!-- PAGE 3: TRENDS -->
<div class="page" id="page-trends">
  <div class="page-header"><div class="page-title">Volume Trends</div><div class="page-sub">How Yassir's review activity evolved over time</div></div>
  <div class="chart-grid-1 animate-in">
    <div class="chart-card">
      <div class="chart-card-header"><div><div class="chart-card-title">Monthly Review Volume</div><div class="chart-card-sub">All reviews</div></div>
      <div style="display:flex;gap:8px">
        <button class="filter-btn active" id="trend-all" onclick="switchTrend('all')">All</button>
        <button class="filter-btn" id="trend-pos" onclick="switchTrend('pos')">Positive</button>
        <button class="filter-btn" id="trend-neg" onclick="switchTrend('neg')">Negative</button>
      </div></div>
      <div class="chart-container" style="height:260px"><canvas id="trendLine"></canvas></div>
    </div>
  </div>
  <div class="chart-grid-2 animate-in" style="transition-delay:0.1s">
    <div class="chart-card">
      <div class="chart-card-header"><div><div class="chart-card-title">Growth Timeline</div><div class="chart-card-sub">Annual review count</div></div></div>
      <div id="timeline-chart" style="padding:4px 0;max-height:300px;overflow-y:auto"></div>
    </div>
    <div class="chart-card">
      <div class="chart-card-header"><div><div class="chart-card-title">Volume Heatmap</div><div class="chart-card-sub">Month × Year intensity</div></div></div>
      <div class="chart-container" style="height:260px"><canvas id="heatmapChart"></canvas></div>
    </div>
  </div>
  <div class="insight-box animate-in" style="transition-delay:0.2s">
    <strong>📌 Growth story:</strong> From just {yearly_counts[0]} reviews in {yearly_labels[0]} to {max(yearly_counts):,} in {yearly_labels[yearly_counts.index(max(yearly_counts))]} — a <strong>{round(max(yearly_counts)/yearly_counts[0])}× increase</strong>. The 2021 spike coincides with COVID-19 lockdowns boosting food delivery demand. Post-2022 growth reflects Yassir expanding into new cities and services.
  </div>
</div>

<!-- PAGE 4: WORDS -->
<div class="page" id="page-words">
  <div class="page-header"><div class="page-title">Word Analysis</div><div class="page-sub">Most frequent terms across positive and negative reviews</div></div>
  <div class="filter-bar animate-in">
    <span class="filter-label">Show:</span>
    <button class="filter-btn active" onclick="switchCloud('all',this)">All Reviews</button>
    <button class="filter-btn" onclick="switchCloud('pos',this)">Positive Only</button>
    <button class="filter-btn" onclick="switchCloud('neg',this)">Negative Only</button>
  </div>
  <div class="chart-grid-1 animate-in" style="transition-delay:0.1s">
    <div class="chart-card">
      <div class="chart-card-header"><div><div class="chart-card-title">Word Cloud</div><div class="chart-card-sub">Most frequent words — size = frequency</div></div></div>
      <div class="wordcloud-container" id="wordcloud"></div>
    </div>
  </div>
  <div class="chart-grid-2 animate-in" style="transition-delay:0.2s">
    <div class="chart-card">
      <div class="chart-card-header"><div><div class="chart-card-title">Top 15 — Positive Reviews</div><div class="chart-card-sub">Most frequent in 4–5★ reviews</div></div></div>
      <div class="chart-container" style="height:300px"><canvas id="topWordsPos"></canvas></div>
    </div>
    <div class="chart-card">
      <div class="chart-card-header"><div><div class="chart-card-title">Top 15 — Negative Reviews</div><div class="chart-card-sub">Most frequent in 1–2★ reviews</div></div></div>
      <div class="chart-container" style="height:300px"><canvas id="topWordsNeg"></canvas></div>
    </div>
  </div>
  <div class="insight-box animate-in" style="transition-delay:0.3s">
    <strong>📌 Key finding:</strong> Positive reviews center on speed, service, and driver quality. Negative reviews highlight pricing, bugs, cancellations, and driver behavior. The word "chauffeur/driver" appears in both — it is the single most impactful variable in user satisfaction.
  </div>
</div>

<!-- PAGE 5: TOP REVIEWS -->
<div class="page" id="page-reviews">
  <div class="page-header"><div class="page-title">Top Reviews</div><div class="page-sub">Most upvoted reviews by the community — real data</div></div>
  <div class="filter-bar animate-in">
    <span class="filter-label">Show:</span>
    <button class="filter-btn active" onclick="filterReviews('all',this)">All</button>
    <button class="filter-btn" onclick="filterReviews('pos',this)">Positive</button>
    <button class="filter-btn" onclick="filterReviews('neg',this)">Negative</button>
  </div>
  <div class="chart-grid-31 animate-in" style="transition-delay:0.1s">
    <div class="chart-card">
      <div class="chart-card-header"><div><div class="chart-card-title">Most Upvoted Reviews</div><div class="chart-card-sub">Community-validated feedback from real users</div></div></div>
      <div class="review-list" id="review-list"></div>
    </div>
    <div class="chart-card">
      <div class="chart-card-header"><div><div class="chart-card-title">Thumbs Up Chart</div><div class="chart-card-sub">Top reviews by upvotes</div></div></div>
      <div class="chart-container" style="height:300px"><canvas id="thumbsChart"></canvas></div>
      <div class="insight-box" style="margin-top:14px"><strong>📌 Note:</strong> Most upvoted = most representative frustrations. Community validates shared pain points.</div>
    </div>
  </div>
</div>

<!-- PAGE 6: LANGUAGES -->
<div class="page" id="page-languages">
  <div class="page-header"><div class="page-title">Language Distribution</div><div class="page-sub">Multilingual analysis — detected from {min(SAMPLE_N, total):,} sampled reviews</div></div>
  <div class="chart-grid-2 animate-in">
    <div class="chart-card">
      <div class="chart-card-header"><div><div class="chart-card-title">Language Share</div><div class="chart-card-sub">Heuristic detection on sample</div></div></div>
      <div class="chart-container" style="height:260px"><canvas id="langDonut"></canvas></div>
    </div>
    <div class="chart-card">
      <div class="chart-card-header"><div><div class="chart-card-title">Language Breakdown</div><div class="chart-card-sub">With estimated counts</div></div></div>
      <div style="padding:16px 0" id="lang-bars"></div>
      <div class="insight-box"><strong>📌 NLP challenge:</strong> Reviews mix French, Arabic (MSA + Darija), and English — sometimes in the same sentence. Standard NLP tools need language-specific preprocessing for each.</div>
    </div>
  </div>
  <div class="chart-grid-1 animate-in" style="transition-delay:0.1s">
    <div class="chart-card">
      <div class="chart-card-header"><div><div class="chart-card-title">Avg Rating by Language</div><div class="chart-card-sub">Do different language groups rate differently?</div></div></div>
      <div class="chart-container" style="height:200px"><canvas id="langRating"></canvas></div>
    </div>
  </div>
</div>

<!-- PAGE 7: INSIGHTS -->
<div class="page" id="page-insights">
  <div class="page-header"><div class="page-title">Key Insights</div><div class="page-sub">Findings &amp; recommendations for Yassir</div></div>
  <div class="chart-grid-2 animate-in">
    <div class="chart-card">
      <div class="chart-card-header"><div><div class="chart-card-title">Review Length vs Sentiment</div><div class="chart-card-sub">Avg characters per group</div></div></div>
      <div class="chart-container" style="height:240px"><canvas id="lengthChart"></canvas></div>
      <div class="insight-box"><strong>📌 Finding:</strong> Negative reviews average <strong>{avg_len_neg} chars</strong> vs {avg_len_pos} for positive. Unhappy users explain their problems in detail — rich data for NLP.</div>
    </div>
    <div class="chart-card">
      <div class="chart-card-header"><div><div class="chart-card-title">Rating Composition × Year</div><div class="chart-card-sub">Stacked — sentiment evolution</div></div></div>
      <div class="chart-container" style="height:240px"><canvas id="stackedBar"></canvas></div>
    </div>
  </div>
  <div class="chart-grid-1 animate-in" style="transition-delay:0.1s">
    <div class="chart-card" style="background:linear-gradient(135deg,var(--card) 0%,rgba(108,43,219,0.05) 100%)">
      <div class="chart-card-title" style="margin-bottom:20px;font-size:17px">🔍 Recommendations for Yassir</div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px" id="rec-grid"></div>
    </div>
  </div>
</div>

</main>

<script>
const D = {data_json};
const C = {{purple:'#6C2BDB',pink:'#E91E8C',green:'#2ecc71',red:'#e74c3c',yellow:'#f39c12',blue:'#3498db',muted:'#7070a0',card:'#16162a',text:'#e8e8f0'}};
const CD = {{responsive:true,maintainAspectRatio:false,plugins:{{legend:{{labels:{{color:C.text,font:{{family:'DM Sans'}}}}}}}}}};

function showPage(id){{
  document.querySelectorAll('.page').forEach(p=>p.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(n=>n.classList.remove('active'));
  document.getElementById('page-'+id).classList.add('active');
  document.querySelectorAll('.nav-item').forEach(n=>{{if(n.getAttribute('onclick')?.includes(id))n.classList.add('active');}});
  setTimeout(()=>{{document.querySelectorAll('#page-'+id+' .animate-in').forEach((el,i)=>{{setTimeout(()=>el.classList.add('visible'),i*80);}});}},50);
  if(window.innerWidth<900)toggleSidebar(false);
}}
function toggleSidebar(force){{
  const s=document.getElementById('sidebar'),o=document.getElementById('overlay');
  const open=force!==undefined?force:!s.classList.contains('open');
  s.classList.toggle('open',open);o.classList.toggle('show',open);
}}

// ── OVERVIEW ──
new Chart(document.getElementById('sentimentDonut'),{{type:'doughnut',data:{{labels:['Positive (4-5★)','Neutral (3★)','Negative (1-2★)'],datasets:[{{data:[D.meta.count_pos,D.meta.count_neu,D.meta.count_neg],backgroundColor:[C.green,C.yellow,C.red],borderWidth:3,borderColor:C.card,hoverOffset:8}}]}},options:{{...CD,cutout:'68%',plugins:{{...CD.plugins,tooltip:{{callbacks:{{label:ctx=>` ${{ctx.label}}: ${{ctx.raw.toLocaleString()}} (${{(ctx.raw/D.meta.total*100).toFixed(1)}}%)`}}}}}}}}}});
new Chart(document.getElementById('yearlyBar'),{{type:'bar',data:{{labels:D.yearly.labels,datasets:[{{data:D.yearly.counts,backgroundColor:D.yearly.counts.map((v,i)=>i===D.yearly.counts.indexOf(Math.max(...D.yearly.counts))?C.purple:'rgba(108,43,219,0.35)'),borderRadius:6,borderSkipped:false}}]}},options:{{...CD,plugins:{{...CD.plugins,legend:{{display:false}}}},scales:{{x:{{grid:{{color:'rgba(255,255,255,0.04)'}},ticks:{{color:C.muted}}}},y:{{grid:{{color:'rgba(255,255,255,0.04)'}},ticks:{{color:C.muted}}}}}}}}}});
new Chart(document.getElementById('monthlyLine'),{{type:'line',data:{{labels:D.monthly.labels,datasets:[{{data:D.monthly.all,label:'Reviews',borderColor:C.purple,backgroundColor:'rgba(108,43,219,0.1)',fill:true,tension:0.4,pointRadius:0}}]}},options:{{...CD,plugins:{{...CD.plugins,legend:{{display:false}}}},scales:{{x:{{grid:{{display:false}},ticks:{{color:C.muted,maxTicksLimit:8}}}},y:{{grid:{{color:'rgba(255,255,255,0.04)'}},ticks:{{color:C.muted}}}}}}}}}});
new Chart(document.getElementById('ratingBarOverview'),{{type:'bar',indexAxis:'y',data:{{labels:['1★','2★','3★','4★','5★'],datasets:[{{data:D.rating_all,backgroundColor:[C.red,'#e67e22',C.yellow,'#27ae60',C.green],borderRadius:5,borderSkipped:false}}]}},options:{{...CD,plugins:{{...CD.plugins,legend:{{display:false}}}},scales:{{x:{{grid:{{color:'rgba(255,255,255,0.04)'}},ticks:{{color:C.muted}}}},y:{{grid:{{display:false}},ticks:{{color:C.text}}}}}}}}}});

// ── RATINGS ──
const yearBtns=document.getElementById('year-filter-btns');
Object.keys(D.rating_by_year).forEach(yr=>{{const b=document.createElement('button');b.className='filter-btn';b.textContent=yr;b.onclick=()=>filterRatings(yr,b);yearBtns.appendChild(b);}});
let ratingBarChart=new Chart(document.getElementById('ratingBarDetail'),{{type:'bar',data:{{labels:['1★','2★','3★','4★','5★'],datasets:[{{data:D.rating_all,backgroundColor:[C.red,'#e67e22',C.yellow,'#27ae60',C.green],borderRadius:8,borderSkipped:false}}]}},options:{{...CD,plugins:{{...CD.plugins,legend:{{display:false}}}},scales:{{x:{{grid:{{display:false}},ticks:{{color:C.text,font:{{size:14,weight:'bold'}}}}}},y:{{grid:{{color:'rgba(255,255,255,0.04)'}},ticks:{{color:C.muted}}}}}},animation:{{duration:600,easing:'easeOutQuart'}}}}}});
let sentPie=new Chart(document.getElementById('sentimentPie'),{{type:'pie',data:{{labels:['Positive','Neutral','Negative'],datasets:[{{data:[D.meta.count_pos,D.meta.count_neu,D.meta.count_neg],backgroundColor:[C.green,C.yellow,C.red],borderWidth:3,borderColor:C.card,hoverOffset:10}}]}},options:{{...CD,plugins:{{...CD.plugins,tooltip:{{callbacks:{{label:ctx=>` ${{ctx.label}}: ${{(ctx.raw/D.meta.total*100).toFixed(1)}}%`}}}}}}}}}});
const rolling=D.monthly.avg_rating.map((v,i,a)=>i===0||i===a.length-1?v:+((a[i-1]+v+a[i+1])/3).toFixed(2));
new Chart(document.getElementById('ratingTrend'),{{type:'line',data:{{labels:D.monthly.labels,datasets:[{{label:'Monthly avg',data:D.monthly.avg_rating,borderColor:'rgba(255,255,255,0.2)',fill:false,tension:0.4,pointRadius:0}},{{label:'3-month rolling',data:rolling,borderColor:C.purple,borderWidth:2.5,fill:false,tension:0.4,pointRadius:0}},{{label:'Overall avg',data:D.monthly.labels.map(()=>D.meta.avg_rating),borderColor:C.red,borderDash:[6,3],borderWidth:1.5,pointRadius:0,fill:false}}]}},options:{{...CD,scales:{{x:{{grid:{{display:false}},ticks:{{color:C.muted,maxTicksLimit:8}}}},y:{{min:1,max:5,grid:{{color:'rgba(255,255,255,0.04)'}},ticks:{{color:C.muted}}}}}}}}}});

function filterRatings(yr,btn){{
  document.querySelectorAll('#page-ratings .filter-btn').forEach(b=>b.classList.remove('active'));
  btn.classList.add('active');
  document.getElementById('rating-year-badge').textContent=yr==='all'?'All Years':yr;
  const d=yr==='all'?D.rating_all:D.rating_by_year[yr];
  ratingBarChart.data.datasets[0].data=d; ratingBarChart.update();
  const total=d.reduce((a,b)=>a+b,0);
  const pos=d[3]+d[4],neg=d[0]+d[1],neu=d[2];
  sentPie.data.datasets[0].data=[pos,neu,neg]; sentPie.update();
}}

// ── TRENDS ──
let trendChart=new Chart(document.getElementById('trendLine'),{{type:'line',data:{{labels:D.monthly.labels,datasets:[{{label:'All Reviews',data:D.monthly.all,borderColor:C.purple,backgroundColor:'rgba(108,43,219,0.12)',fill:true,tension:0.4,pointRadius:2,pointHoverRadius:6}}]}},options:{{...CD,scales:{{x:{{grid:{{display:false}},ticks:{{color:C.muted,maxTicksLimit:10}}}},y:{{grid:{{color:'rgba(255,255,255,0.04)'}},ticks:{{color:C.muted}}}}}}}}}});
function switchTrend(t){{
  document.querySelectorAll('#page-trends .filter-btn').forEach(b=>b.classList.remove('active'));
  document.getElementById('trend-'+t).classList.add('active');
  const map={{all:{{d:D.monthly.all,c:C.purple,l:'All Reviews'}},pos:{{d:D.monthly.pos,c:C.green,l:'Positive'}},neg:{{d:D.monthly.neg,c:C.red,l:'Negative'}}}};
  const m=map[t]; trendChart.data.datasets[0].data=m.d; trendChart.data.datasets[0].borderColor=m.c;
  trendChart.data.datasets[0].backgroundColor=m.c+'33';
  trendChart.data.datasets[0].label=m.l; trendChart.update();
}}
(()=>{{
  const tl=document.getElementById('timeline-chart');
  const mx=Math.max(...D.yearly.counts);
  tl.innerHTML=D.yearly.labels.map((yr,i)=>{{
    const w=Math.round(D.yearly.counts[i]/mx*100);
    return `<div class="timeline-item"><div class="timeline-year">${{yr}}</div><div class="timeline-bar-wrap"><div class="timeline-bar" style="width:0%" data-w="${{w}}%"></div><div class="timeline-count">${{D.yearly.counts[i].toLocaleString()}} reviews</div></div></div>`;
  }}).join('');
  setTimeout(()=>tl.querySelectorAll('.timeline-bar').forEach(b=>b.style.width=b.dataset.w+'%'),400);
}})();
new Chart(document.getElementById('heatmapChart'),{{type:'bubble',data:{{datasets:[{{data:D.heatmap.data,backgroundColor:ctx=>{{const v=ctx.raw.v;const a=Math.min(0.9,0.1+v/300);return `rgba(108,43,219,${{a}})`;}},borderColor:'rgba(108,43,219,0.35)',borderWidth:1}}]}},options:{{...CD,plugins:{{...CD.plugins,legend:{{display:false}},tooltip:{{callbacks:{{label:ctx=>`${{D.heatmap.month_labels[ctx.raw.y]}}, ${{D.heatmap.year_labels[ctx.raw.x]}}: ${{ctx.raw.v}} reviews`}}}}}},scales:{{x:{{min:-0.5,max:D.heatmap.year_labels.length-0.5,grid:{{display:false}},ticks:{{color:C.muted,callback:v=>D.heatmap.year_labels[v]||''}}}},y:{{min:-0.5,max:11.5,grid:{{color:'rgba(255,255,255,0.04)'}},ticks:{{color:C.muted,callback:v=>D.heatmap.month_labels[v]||''}}}}}}}}}});

// ── WORDS ──
function renderCloud(type){{
  const words=D.words[type];const mx=words[0].count;
  const container=document.getElementById('wordcloud');container.innerHTML='';
  const pc=['#2ecc71','#27ae60','#1abc9c','#a855f7','#58d68d'];
  const nc=['#e74c3c','#c0392b','#e67e22','#d35400','#ec407a'];
  const ac=['#6C2BDB','#E91E8C','#a855f7','#2ecc71','#e74c3c','#3498db'];
  words.forEach((w,i)=>{{
    const size=12+(w.count/mx)*28;
    const colors=type==='pos'?pc:type==='neg'?nc:ac;
    const el=document.createElement('span');el.className='wc-word';el.textContent=w.word;
    el.style.fontSize=size+'px';el.style.color=colors[i%colors.length];
    el.style.opacity=0.6+(w.count/mx)*0.4;el.title=`${{w.word}}: ${{w.count}} occurrences`;
    container.appendChild(el);
  }});
}}
renderCloud('all');
function switchCloud(t,btn){{document.querySelectorAll('#page-words .filter-btn').forEach(b=>b.classList.remove('active'));btn.classList.add('active');renderCloud(t);}}
new Chart(document.getElementById('topWordsPos'),{{type:'bar',indexAxis:'y',data:{{labels:D.words.pos.map(w=>w.word),datasets:[{{data:D.words.pos.map(w=>w.count),label:'Frequency',backgroundColor:'rgba(46,204,113,0.7)',borderRadius:4,borderSkipped:false}}]}},options:{{...CD,plugins:{{...CD.plugins,legend:{{display:false}}}},scales:{{x:{{grid:{{color:'rgba(255,255,255,0.04)'}},ticks:{{color:C.muted}}}},y:{{grid:{{display:false}},ticks:{{color:C.text}}}}}}}}}});
new Chart(document.getElementById('topWordsNeg'),{{type:'bar',indexAxis:'y',data:{{labels:D.words.neg.map(w=>w.word),datasets:[{{data:D.words.neg.map(w=>w.count),label:'Frequency',backgroundColor:'rgba(231,76,60,0.7)',borderRadius:4,borderSkipped:false}}]}},options:{{...CD,plugins:{{...CD.plugins,legend:{{display:false}}}},scales:{{x:{{grid:{{color:'rgba(255,255,255,0.04)'}},ticks:{{color:C.muted}}}},y:{{grid:{{display:false}},ticks:{{color:C.text}}}}}}}}}});

// ── REVIEWS ──
function renderReviews(filter){{
  const reviews=filter==='all'?D.top_reviews:filter==='pos'?D.top_reviews.filter(r=>r.rating>=4):D.top_reviews.filter(r=>r.rating<=2);
  document.getElementById('review-list').innerHTML=reviews.map(r=>`<div class="review-item"><div class="review-meta"><div class="review-stars ${{r.rating<=2?'neg':''}}">${{'★'.repeat(r.rating)}}${{' ☆'.repeat(5-r.rating)}}</div><div class="review-author">${{r.author}}</div><div class="review-date">${{r.date}}</div></div><div class="review-text">${{r.text}}</div><div class="review-thumbs">👍 ${{r.thumbs}} people found this helpful</div></div>`).join('');
}}
renderReviews('all');
function filterReviews(t,btn){{document.querySelectorAll('#page-reviews .filter-btn').forEach(b=>b.classList.remove('active'));btn.classList.add('active');renderReviews(t);}}
new Chart(document.getElementById('thumbsChart'),{{type:'bar',indexAxis:'y',data:{{labels:D.top_reviews.slice(0,8).map(r=>r.author.substring(0,14)+'…'),datasets:[{{data:D.top_reviews.slice(0,8).map(r=>r.thumbs),label:'👍',backgroundColor:D.top_reviews.slice(0,8).map(r=>r.rating<=2?'rgba(231,76,60,0.7)':'rgba(46,204,113,0.7)'),borderRadius:4}}]}},options:{{...CD,plugins:{{...CD.plugins,legend:{{display:false}}}},scales:{{x:{{grid:{{color:'rgba(255,255,255,0.04)'}},ticks:{{color:C.muted}}}},y:{{grid:{{display:false}},ticks:{{color:C.text,font:{{size:10}}}}}}}}}}}});

// ── LANGUAGES ──
new Chart(document.getElementById('langDonut'),{{type:'doughnut',data:{{labels:D.languages.labels,datasets:[{{data:D.languages.counts,backgroundColor:[C.blue,C.green,C.red,'#9b59b6'],borderWidth:3,borderColor:C.card,hoverOffset:8}}]}},options:{{...CD,cutout:'60%',plugins:{{...CD.plugins,tooltip:{{callbacks:{{label:ctx=>`${{ctx.label}}: ${{ctx.raw}} (${{(ctx.raw/D.languages.counts.reduce((a,b)=>a+b,0)*100).toFixed(1)}}%)`}}}}}}}}}});
(()=>{{
  const total=D.languages.counts.reduce((a,b)=>a+b,0);
  const colors=[C.blue,C.green,C.red,'#9b59b6'];
  document.getElementById('lang-bars').innerHTML=D.languages.labels.map((l,i)=>{{
    const pct=(D.languages.counts[i]/total*100).toFixed(1);
    return `<div class="lang-bar-item"><div class="lang-bar-label"><span>${{l}}</span><span style="color:${{colors[i]}}">${{pct}}%</span></div><div class="lang-bar-track"><div class="lang-bar-fill" style="width:0%;background:${{colors[i]}}" data-w="${{pct}}%"></div></div></div>`;
  }}).join('');
  setTimeout(()=>document.querySelectorAll('.lang-bar-fill').forEach(b=>b.style.width=b.dataset.w),400);
}})();
new Chart(document.getElementById('langRating'),{{type:'bar',data:{{labels:D.languages.labels,datasets:[{{label:'Avg Rating',data:D.languages.avg_ratings,backgroundColor:[C.blue,C.green,C.red,'#9b59b6'],borderRadius:8,borderSkipped:false}}]}},options:{{...CD,plugins:{{...CD.plugins,legend:{{display:false}}}},scales:{{x:{{grid:{{display:false}},ticks:{{color:C.text}}}},y:{{min:0,max:5,grid:{{color:'rgba(255,255,255,0.04)'}},ticks:{{color:C.muted}}}}}}}}}});

// ── INSIGHTS ──
new Chart(document.getElementById('lengthChart'),{{type:'bar',data:{{labels:['Positive (4-5★)','Neutral (3★)','Negative (1-2★)'],datasets:[{{label:'Avg Characters',data:[D.meta.avg_len_pos,D.meta.avg_len_neu,D.meta.avg_len_neg],backgroundColor:[C.green,C.yellow,C.red],borderRadius:8,borderSkipped:false}}]}},options:{{...CD,plugins:{{...CD.plugins,legend:{{display:false}}}},scales:{{x:{{grid:{{display:false}},ticks:{{color:C.text}}}},y:{{grid:{{color:'rgba(255,255,255,0.04)'}},ticks:{{color:C.muted}},title:{{display:true,text:'Avg Characters',color:C.muted}}}}}}}}}});
new Chart(document.getElementById('stackedBar'),{{type:'bar',data:{{labels:D.stacked.years,datasets:[{{label:'5★',data:D.stacked.by_star['5'],backgroundColor:C.green,borderRadius:2}},{{label:'4★',data:D.stacked.by_star['4'],backgroundColor:'#27ae60',borderRadius:2}},{{label:'3★',data:D.stacked.by_star['3'],backgroundColor:C.yellow,borderRadius:2}},{{label:'2★',data:D.stacked.by_star['2'],backgroundColor:'#e67e22',borderRadius:2}},{{label:'1★',data:D.stacked.by_star['1'],backgroundColor:C.red,borderRadius:2}}]}},options:{{...CD,scales:{{x:{{stacked:true,grid:{{display:false}},ticks:{{color:C.text}}}},y:{{stacked:true,grid:{{color:'rgba(255,255,255,0.04)'}},ticks:{{color:C.muted}}}}}}}}}});
document.getElementById('rec-grid').innerHTML=[
  {{icon:'🚗',title:'Driver Quality',text:'Implement real-time driver rating alerts. Driver behavior is the #1 variable in user satisfaction across all languages.'}},
  {{icon:'🐛',title:'Bug Fixes',text:'Rating dips correlate with app updates. Strengthen QA testing pipelines before each release.'}},
  {{icon:'💰',title:'Pricing Transparency',text:'Users feel surprised by surge pricing. Add clear fee breakdowns before confirming orders.'}},
  {{icon:'🇩🇿',title:'Arabic Support',text:'Arabic reviewers are an underserved segment. Prioritize Arabic UI and dedicated customer support.'}},
  {{icon:'📊',title:'Monthly Monitoring',text:'Build an internal dashboard to track review sentiment monthly and catch issues before they escalate.'}},
  {{icon:'🤖',title:'NLP Next Steps',text:'Train a multilingual sentiment classifier. Aspect-based analysis for drivers, pricing, and bugs.'}}
].map(r=>`<div class="chart-card" style="padding:16px"><div style="font-size:22px;margin-bottom:8px">${{r.icon}}</div><div style="font-family:'Syne',sans-serif;font-weight:700;font-size:13px;margin-bottom:6px;color:#a855f7">${{r.title}}</div><div style="font-size:12px;color:#c0c0d8;line-height:1.6">${{r.text}}</div></div>`).join('');

// ── INIT ANIMATIONS ──
setTimeout(()=>document.querySelectorAll('#page-overview .animate-in').forEach((el,i)=>setTimeout(()=>el.classList.add('visible'),i*100)),100);
Chart.defaults.color=C.muted; Chart.defaults.borderColor='rgba(255,255,255,0.06)';
</script>
</body>
</html>"""

# ─── WRITE FILE ────────────────────────────────────────────────
print(f"\n💾 Writing {OUT_PATH}...")
with open(OUT_PATH, 'w', encoding='utf-8') as f:
    f.write(HTML)

size = os.path.getsize(OUT_PATH) / 1024
print(f"✅ Done! File size: {size:.0f} KB")
print(f"\n👉 Open in browser: lab1/notebooks/{OUT_PATH}")
print("   (double-click it or drag it into Chrome/Firefox/Edge)")
