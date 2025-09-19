import streamlit as st
import pandas as pd
import requests
import re
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from hybrid import hybrid_recommendation, get_profile_from_recommendations, get_explanations, df_movies

# ---------- CONFIG ----------
OMDB_API_KEY = "da8f830a"   # your OMDb key
PLACEHOLDER = "https://via.placeholder.com/150"
OMDB_CACHE_FILE = "omdb_cache.json"
MAX_WORKERS = 8  # threads for concurrent prefetching

# ---------- Load or init disk cache ----------
try:
    if os.path.exists(OMDB_CACHE_FILE):
        with open(OMDB_CACHE_FILE, "r", encoding="utf-8") as f:
            _poster_cache = json.load(f)
    else:
        _poster_cache = {}
except Exception:
    _poster_cache = {}

# ---------- HELPERS ----------
def save_cache_to_disk():
    try:
        with open(OMDB_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(_poster_cache, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def extract_imdb_id(imdb_url):
    if not isinstance(imdb_url, str):
        return None
    imdb_url = imdb_url.split('?')[0].strip().rstrip('/')
    for p in imdb_url.split('/')[::-1]:
        if p.startswith("tt"):
            return p
    return None

def clean_title(title):
    if not isinstance(title, str):
        return title
    return re.sub(r'\s*[\(\[]\d{4}[\)\]]\s*$', '', title).strip()

def fetch_omdb_json_by_id(imdb_id):
    try:
        resp = requests.get("http://www.omdbapi.com/", params={"i": imdb_id, "apikey": OMDB_API_KEY}, timeout=6)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        return None
    return None

def fetch_omdb_json_by_title(title):
    try:
        resp = requests.get("http://www.omdbapi.com/", params={"t": title, "apikey": OMDB_API_KEY}, timeout=6)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        return None
    return None

def build_metadata_from_omdb(data):
    """Normalize OMDb json into our metadata dict"""
    result = {
        "Poster": PLACEHOLDER,
        "imdbRating": "N/A",
        "Year": "N/A",
        "Genre": "N/A",
        "Runtime": "N/A",
        "Director": "N/A",
        "Language": "N/A"
    }
    if not data or data.get("Response") == "False":
        return result
    if data.get("Poster") and data["Poster"] != "N/A":
        result["Poster"] = data["Poster"]
    if data.get("imdbRating"):
        result["imdbRating"] = data["imdbRating"]
    if data.get("Year"):
        result["Year"] = data["Year"]
    if data.get("Genre"):
        result["Genre"] = data["Genre"]
    if data.get("Runtime"):
        result["Runtime"] = data["Runtime"]
    if data.get("Director"):
        result["Director"] = data["Director"]
    if data.get("Language"):
        result["Language"] = data["Language"]
    return result

def get_movie_metadata(title, imdb_url=None):
    """Return poster, rating, year, genre, runtime, director, language for a movie.
       Uses in-memory cache which is persisted to disk on updates.
    """
    cache_key = f"{title}||{imdb_url}"
    if cache_key in _poster_cache:
        return _poster_cache[cache_key]

    imdb_id = extract_imdb_id(imdb_url) if imdb_url else None
    data = None
    if imdb_id:
        data = fetch_omdb_json_by_id(imdb_id)
    if not data or data.get("Response") == "False":
        ctitle = clean_title(title)
        data = fetch_omdb_json_by_title(ctitle)

    result = build_metadata_from_omdb(data)
    _poster_cache[cache_key] = result
    try:
        save_cache_to_disk()
    except Exception:
        pass
    return result

# ---------- Batch prefetch for candidates ----------
def batch_prefetch_metadata(recs_df, max_workers=MAX_WORKERS):
    """
    Given a DataFrame of candidate recommendations (with 'processed_title' and 'IMDb_URL'),
    prefetch metadata for all items using a ThreadPoolExecutor and cache results to disk.
    """
    if recs_df is None or recs_df.empty:
        return

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {}
        for _, row in recs_df.iterrows():
            title = row.get("processed_title") or row.get("movie_title") or ""
            imdb_url = row.get("IMDb_URL")
            key = f"{title}||{imdb_url}"
            if key in _poster_cache:
                continue  # already cached
            futures[ex.submit(_fetch_and_store, title, imdb_url)] = key

        for future in as_completed(list(futures.keys())):
            try:
                _ = future.result()
            except Exception:
                pass

    try:
        save_cache_to_disk()
    except Exception:
        pass

def _fetch_and_store(title, imdb_url):
    """Helper used by threadpool to fetch metadata and store in cache"""
    try:
        imdb_id = extract_imdb_id(imdb_url) if imdb_url else None
        data = None
        if imdb_id:
            data = fetch_omdb_json_by_id(imdb_id)
        if not data or data.get("Response") == "False":
            ctitle = clean_title(title)
            data = fetch_omdb_json_by_title(ctitle)
        result = build_metadata_from_omdb(data)
        cache_key = f"{title}||{imdb_url}"
        _poster_cache[cache_key] = result
        return True
    except Exception:
        return False

# ---------- STREAMLIT UI ----------
st.set_page_config(page_title="Hybrid Movie Recommender", layout="wide")
st.title("🎬 Personalized Movie Recommendation Engine")
st.markdown("Hybrid recommendations with advanced filters, per-user taste summary, brief explanations, and export.")

# INPUTS
user_id_input = st.number_input("Enter User ID:", min_value=1, max_value=943, value=1, step=1)
top_n_input = st.slider("Number of recommendations to display:", min_value=5, max_value=20, value=10)

# FILTERS (sidebar)
st.sidebar.header("🔍 Filter Options (optional)")
genre_filter = st.sidebar.text_input("Filter by Genre (e.g. Comedy, Action)", "")
year_filter = st.sidebar.slider("Release Year After", 1900, 2025, 1900)
rating_filter = st.sidebar.slider("Minimum IMDb Rating", 0.0, 10.0, 0.0, step=0.1)

# NEW advanced filters
st.sidebar.subheader("Advanced Filters")
runtime_option = st.sidebar.selectbox("Runtime", options=["Any", "Short (<90 min)", "Medium (90–150 min)", "Long (>150 min)"], index=0)
director_filter = st.sidebar.text_input("Director (contains)", "")
language_filter = st.sidebar.text_input("Language (contains)", "")

# helper: check if any filter is active
def any_filter_active():
    defaults = {
        "genre": "",
        "year": 1900,
        "rating": 0.0,
        "runtime": "Any",
        "director": "",
        "language": ""
    }
    if genre_filter.strip() != defaults["genre"]:
        return True
    if year_filter != defaults["year"]:
        return True
    if rating_filter != defaults["rating"]:
        return True
    if runtime_option != defaults["runtime"]:
        return True
    if director_filter.strip() != defaults["director"]:
        return True
    if language_filter.strip() != defaults["language"]:
        return True
    return False

# MAIN: Get recommendations button
if st.button("Get Recommendations"):
    candidate_pool_size = max(top_n_input * 5, 50)  # ensure a decent pool

    try:
        candidates = hybrid_recommendation(int(user_id_input), top_n=candidate_pool_size)
    except Exception as e:
        st.error(f"Error generating recommendations: {e}")
        st.stop()

    if candidates is None or candidates.shape[0] == 0:
        st.info("No recommendations returned for this user.")
        st.stop()

    # Build profile from candidate recommendations (dynamic taste summary)
    recs_profile = get_profile_from_recommendations(candidates)

    # Show user taste profile only if NO filters active
    if not any_filter_active():
        st.markdown("### 🔎 What kind of movies we're recommending you")
        st.info(recs_profile.get("summary", "We couldn't form a profile from the recommendations."))
    else:
        # show active filters summary
        tags = []
        if genre_filter:
            tags.append(f"Genre: {genre_filter}")
        if year_filter and year_filter != 1900:
            tags.append(f"Year after: {year_filter}")
        if rating_filter and rating_filter > 0:
            tags.append(f"Min IMDb: {rating_filter}")
        if runtime_option and runtime_option != "Any":
            tags.append(f"Runtime: {runtime_option}")
        if director_filter:
            tags.append(f"Director contains: {director_filter}")
        if language_filter:
            tags.append(f"Language contains: {language_filter}")

        if tags:
            st.markdown("### 🔖 Active Filters")
            st.write(", ".join(tags))

    # Batch prefetch OMDb metadata for candidates (so filters run quickly)
    with st.spinner("Fetching metadata for candidate movies (cached locally)..."):
        batch_prefetch_metadata(candidates)

    # Prepare explanations map for the candidate pool
    rec_ids = candidates["movie_id"].tolist()
    explanations_map = get_explanations(int(user_id_input), rec_ids, top_k=2)

    # Now apply filters one-by-one and collect displayed items
    displayed = 0
    displayed_rows = []
    st.subheader(f"Top {top_n_input} Recommendations (after filters) for User {user_id_input}")

    for i, row in candidates.iterrows():
        if displayed >= top_n_input:
            break

        title = row['movie_title']
        imdb_url = row.get('IMDb_URL')
        processed_title = row.get('processed_title', title)
        movie_id = row.get('movie_id')

        meta = get_movie_metadata(processed_title, imdb_url)
        poster = meta["Poster"]
        imdb_rating = meta["imdbRating"]
        year = meta["Year"]
        genre = meta["Genre"]
        runtime = meta["Runtime"]
        director = meta["Director"]
        language = meta["Language"]

        # APPLY FILTERS (genre/year/imdb already present)
        if genre_filter and genre_filter.lower() not in (genre or "").lower():
            continue

        # year filter
        try:
            if year != "N/A" and re.search(r'\d{4}', str(year)):
                y = int(re.search(r'(\d{4})', str(year)).group(1))
                if y < year_filter:
                    continue
            else:
                if year_filter and year_filter > 1900:
                    continue
        except Exception:
            pass

        # imdb rating filter
        try:
            if imdb_rating != "N/A" and float(imdb_rating) < rating_filter:
                continue
        except Exception:
            pass

        # director filter (case-insensitive substring)
        if director_filter and director_filter.strip():
            if director == "N/A" or director_filter.lower() not in director.lower():
                continue

        # language filter
        if language_filter and language_filter.strip():
            if language == "N/A" or language_filter.lower() not in language.lower():
                continue

        # runtime filter
        if runtime_option != "Any":
            minutes = None
            try:
                if runtime and runtime != "N/A":
                    m = re.search(r'(\d+)', runtime)
                    if m:
                        minutes = int(m.group(1))
            except Exception:
                minutes = None

            if runtime_option == "Short (<90 min)":
                if minutes is None or minutes >= 90:
                    continue
            elif runtime_option == "Medium (90–150 min)":
                if minutes is None or not (90 <= minutes <= 150):
                    continue
            elif runtime_option == "Long (>150 min)":
                if minutes is None or minutes <= 150:
                    continue

        # Passed all filters — display
        col1, col2 = st.columns([1, 4])
        with col1:
            st.image(poster, width=140)
        with col2:
            st.markdown(f"**{displayed+1}. {title}**")
            st.write(f"⭐ IMDb Rating: {imdb_rating}    |    📅 Year: {year}")
            st.write(f"🎭 Genre: {genre}")
            st.write(f"⏱️ Runtime: {runtime}    |    🎬 Director: {director}    |    🗣️ Language: {language}")

            # Explanation line (because you liked X and Y)
            expl = explanations_map.get(movie_id, [])
            if expl:
                expl_text = " and ".join([str(x) for x in expl])
                st.write(f"🔎 Because you liked: {expl_text}")
            else:
                st.write("")  # empty line if none

            if imdb_url and isinstance(imdb_url, str):
                st.write(f"[IMDb Link]({imdb_url})")

        displayed_rows.append({
            "rank": displayed + 1,
            "movie_title": title,
            "movie_id": movie_id,
            "imdb_rating": imdb_rating,
            "year": year,
            "genre": genre,
            "runtime": runtime,
            "director": director,
            "language": language,
            "explanation": ("; ".join(expl) if expl else "")
        })

        displayed += 1

    if displayed == 0:
        st.info("No recommended movies matched your filters. Try relaxing the filters.")

    # ---------- Export: build a TXT for download ----------
    if displayed > 0:
        def build_export_text():
            lines = []
            lines.append(f"Recommendations for User ID: {user_id_input}")
            lines.append("")
            # include profile summary (always include the recs-based profile so interviewer sees system's view)
            lines.append("Taste Summary (from recommendations):")
            lines.append(recs_profile.get("summary", "N/A"))
            lines.append("")
            lines.append("Final displayed recommendations:")
            lines.append("")
            for r in displayed_rows:
                lines.append(f"{r['rank']}. {r['movie_title']} (Year: {r['year']}, IMDb: {r['imdb_rating']})")
                lines.append(f"   Genre: {r['genre']}")
                lines.append(f"   Runtime: {r['runtime']}, Director: {r['director']}, Language: {r['language']}")
                if r['explanation']:
                    lines.append(f"   Why: Because you liked {r['explanation']}")
                lines.append("")
            return "\n".join(lines)

        export_text = build_export_text()
        st.download_button("📥 Download recommendations & profile (TXT)", data=export_text, file_name=f"recs_user_{user_id_input}.txt", mime="text/plain")
