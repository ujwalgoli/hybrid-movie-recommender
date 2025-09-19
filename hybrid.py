import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re

# ---------- Load MovieLens Metadata ----------
df_movies = pd.read_csv(
    "u.item",
    sep="|",
    header=None,
    encoding="latin-1",
    usecols=[0, 1, 2, 4] + list(range(5, 24)),
    names=["movie_id", "movie_title", "release_date", "IMDb_URL"] + [
        "unknown", "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
        "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery",
        "Romance", "Sci-Fi", "Thriller", "War", "Western"
    ]
)

# Create "genres" column
genre_cols = df_movies.columns[4:]
df_movies["genres"] = df_movies[genre_cols].apply(
    lambda row: ", ".join([g for g, val in row.items() if val == 1]), axis=1
)

# ---------- Preprocess Titles ----------
def preprocess_title(title):
    if not isinstance(title, str):
        return title
    title = re.sub(r'\s*[\(\[]\d{4}[\)\]]\s*$', '', title).strip()
    m = re.match(r'^(.*), (The|A|An)$', title)
    if m:
        title = f"{m.group(2)} {m.group(1)}"
    title = re.sub(r'\s+', ' ', title).strip()
    return title

df_movies["processed_title"] = df_movies["movie_title"].apply(preprocess_title)

# ---------- Load Ratings ----------
df_ratings = pd.read_csv(
    "u.data",
    sep="\t",
    header=None,
    names=["user_id", "movie_id", "rating", "timestamp"]
)

# ---------- User-Item Matrix & Similarities ----------
user_item_matrix = df_ratings.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)
user_similarity_df = pd.DataFrame(cosine_similarity(user_item_matrix), index=user_item_matrix.index, columns=user_item_matrix.index)
item_similarity_df = pd.DataFrame(cosine_similarity(user_item_matrix.T), index=user_item_matrix.columns, columns=user_item_matrix.columns)

genre_matrix = df_movies[genre_cols].values
genre_similarity_df = pd.DataFrame(cosine_similarity(genre_matrix), index=df_movies['movie_id'], columns=df_movies['movie_id'])

# Popularity (number of ratings per movie) normalized
popularity_counts = df_ratings.groupby("movie_id").size()
if not popularity_counts.empty:
    popularity_norm = (popularity_counts - popularity_counts.min()) / (popularity_counts.max() - popularity_counts.min())
else:
    popularity_norm = pd.Series(dtype=float)

# Internal default weights (kept in backend only)
_DEFAULT_WEIGHTS = {
    "alpha": 0.35,   # user CF
    "beta": 0.25,    # item CF
    "gamma": 0.15,   # genre
    "delta": 0.10,   # year
    "epsilon": 0.10, # imdb placeholder (app may not use it)
    "zeta": 0.05     # popularity
}

# ---------- Helper: decade bucket ----------
def decade_label(year):
    try:
        y = int(year)
    except Exception:
        return None
    decade = (y // 10) * 10
    return f"{decade}s"

# ---------- User taste profile function (historical) ----------
def get_user_profile(user_id, top_k_genres=2):
    """
    Analyze the user's rated movies and return a short taste summary.
    (Kept for compatibility — uses historical high-rated movies.)
    Returns a dict: {"summary": str, "top_genres": [...], "decade": "1990s"}
    """
    profile = {"summary": "No data available.", "top_genres": [], "decade": None}
    if user_id not in user_item_matrix.index:
        return profile

    user_ratings = user_item_matrix.loc[user_id]
    watched = user_ratings[user_ratings > 0]
    if watched.empty:
        return profile

    high_rated = watched[watched >= 4]
    if high_rated.empty:
        high_rated = watched

    genre_scores = {}
    for g in genre_cols:
        movies_with_g = df_movies[df_movies[g] == 1]["movie_id"].astype(int)
        count = len(set(movies_with_g).intersection(set(high_rated.index)))
        if count > 0:
            genre_scores[g] = count

    if genre_scores:
        sorted_genres = sorted(genre_scores.items(), key=lambda x: x[1], reverse=True)
        top_genres = [g for g, _ in sorted_genres[:top_k_genres]]
        profile["top_genres"] = top_genres
    else:
        profile["top_genres"] = []

    years = []
    for mid in high_rated.index:
        try:
            row = df_movies[df_movies["movie_id"] == mid]
            if not row.empty:
                rdate = row.iloc[0]["release_date"]
                if isinstance(rdate, str) and rdate.strip() != "":
                    m = re.search(r'(\d{4})', rdate)
                    if m:
                        years.append(int(m.group(1)))
        except Exception:
            continue

    if years:
        avg_year = int(np.mean(years))
        dec = decade_label(avg_year)
        profile["decade"] = dec
    else:
        profile["decade"] = None

    parts = []
    if profile["top_genres"]:
        g_text = " and ".join(profile["top_genres"])
        parts.append(f"you tend to enjoy {g_text} movies")
    if profile["decade"]:
        parts.append(f"especially from the {profile['decade']}")
    if parts:
        summary = "You " + " ".join(parts) + "."
    else:
        summary = "We couldn't detect a clear taste profile from your ratings."
    profile["summary"] = summary.capitalize()
    return profile

# ---------- New: Taste profile based on recommendations ----------
def get_profile_from_recommendations(recs_df, top_k_genres=2):
    """
    Given a recommendations DataFrame (must include 'genres' and optional 'movie_title'/'release_date'),
    return a short profile summary of the recommendations (top genres + decade).
    This is used to describe the type of movies the system is recommending to the user.
    """
    profile = {"summary": "No data available from recommendations.", "top_genres": [], "decade": None}
    if recs_df is None or recs_df.empty:
        return profile

    # aggregate genres from recs
    genre_count = {}
    for idx, row in recs_df.iterrows():
        gstr = row.get("genres", "")
        if not isinstance(gstr, str) or gstr.strip() == "":
            continue
        for g in [x.strip() for x in gstr.split(",") if x.strip()]:
            genre_count[g] = genre_count.get(g, 0) + 1

    if genre_count:
        sorted_genres = sorted(genre_count.items(), key=lambda x: x[1], reverse=True)
        top_genres = [g for g, _ in sorted_genres[:top_k_genres]]
        profile["top_genres"] = top_genres

    # decade: try to extract years from recs (if release_date column exists)
    years = []
    if "release_date" in recs_df.columns:
        for idx, row in recs_df.iterrows():
            rdate = row.get("release_date")
            if isinstance(rdate, str) and rdate.strip() != "":
                m = re.search(r'(\d{4})', rdate)
                if m:
                    try:
                        years.append(int(m.group(1)))
                    except Exception:
                        continue
    # fallback: if not in recs_df, try to map movie_id to df_movies release_date
    if not years and "movie_id" in recs_df.columns:
        for mid in recs_df["movie_id"].tolist():
            try:
                row = df_movies[df_movies["movie_id"] == mid]
                if not row.empty:
                    rdate = row.iloc[0].get("release_date", "")
                    if isinstance(rdate, str) and rdate.strip() != "":
                        m = re.search(r'(\d{4})', rdate)
                        if m:
                            years.append(int(m.group(1)))
            except Exception:
                continue

    if years:
        avg_year = int(np.mean(years))
        dec = decade_label(avg_year)
        profile["decade"] = dec

    parts = []
    if profile["top_genres"]:
        g_text = " and ".join(profile["top_genres"])
        parts.append(f"these recommendations are heavy on {g_text} movies")
    if profile["decade"]:
        parts.append(f"mostly from the {profile['decade']}")
    if parts:
        summary = "We recommend " + " and ".join(parts) + "."
    else:
        summary = "We couldn't form a clear recommendation profile."
    profile["summary"] = summary.capitalize()
    return profile

# ---------- New: explanation generator ----------
def get_explanations(user_id, rec_movie_ids, top_k=2):
    """
    For each recommended movie id in rec_movie_ids, find top_k watched movies
    (from user's history) that contributed most to the recommendation based on
    item similarity * user's rating. Return dict: movie_id -> [title1, title2...]
    """
    explanations = {}
    if user_id not in user_item_matrix.index:
        for mid in rec_movie_ids:
            explanations[mid] = []
        return explanations

    user_ratings = user_item_matrix.loc[user_id]
    watched = user_ratings[user_ratings > 0]
    watched_ids = watched.index.tolist()
    if len(watched_ids) == 0:
        for mid in rec_movie_ids:
            explanations[mid] = []
        return explanations

    # For each recommended movie, compute contribution score from each watched movie:
    # contribution = item_similarity(rec, watched) * (user_rating / 5.0)  (normalize rating)
    for mid in rec_movie_ids:
        try:
            if mid not in item_similarity_df.index:
                explanations[mid] = []
                continue
            sims = item_similarity_df.loc[mid, watched_ids]
            # watched ratings normalized (0-1)
            ratings_norm = user_ratings[watched_ids] / 5.0
            contrib = sims * ratings_norm
            contrib = contrib.sort_values(ascending=False)
            top = contrib.head(top_k)
            titles = []
            for wid in top.index:
                # map watched id to title (if exists)
                row = df_movies[df_movies["movie_id"] == wid]
                if not row.empty:
                    titles.append(row.iloc[0]["movie_title"])
                else:
                    titles.append(f"Movie {wid}")
            explanations[mid] = titles
        except Exception:
            explanations[mid] = []
    return explanations

# ---------- Hybrid Recommendation ----------
def hybrid_recommendation(user_id, top_n=10,
                          weights=None):
    """
    Returns top-N recommended movies for user_id as a DataFrame with columns:
    ['movie_id', 'movie_title', 'processed_title', 'IMDb_URL', 'genres', 'release_date']
    weights: optional dict to override internal defaults (alpha,beta,gamma,delta,epsilon,zeta)
    """
    # ensure user exists
    if user_id not in user_item_matrix.index:
        return pd.DataFrame(columns=['movie_id', 'movie_title', 'processed_title', 'IMDb_URL', 'genres', 'release_date'])

    # merge weights
    if weights is None:
        w = _DEFAULT_WEIGHTS.copy()
    else:
        w = _DEFAULT_WEIGHTS.copy()
        for k, v in (weights.items() if isinstance(weights, dict) else []):
            if k in w:
                w[k] = float(v)

    alpha, beta, gamma, delta, epsilon, zeta = w["alpha"], w["beta"], w["gamma"], w["delta"], w["epsilon"], w["zeta"]

    # --- User CF ---
    sim_scores = user_similarity_df[user_id]
    similar_users = sim_scores.sort_values(ascending=False).drop(user_id).index
    weighted_ratings = user_item_matrix.loc[similar_users].T.dot(sim_scores[similar_users])
    sim_sums = sim_scores[similar_users].sum() if sim_scores[similar_users].sum() != 0 else 1.0
    user_cf_scores = weighted_ratings / sim_sums

    # --- Item CF ---
    user_ratings = user_item_matrix.loc[user_id]
    unrated_movies = user_ratings[user_ratings == 0].index
    item_cf_scores = {}
    for movie in unrated_movies:
        sim_scores_item = item_similarity_df[movie]
        weighted_sum = np.dot(user_ratings, sim_scores_item)
        sim_sum = sim_scores_item.sum()
        item_cf_scores[movie] = weighted_sum / sim_sum if sim_sum != 0 else 0
    item_cf_scores = pd.Series(item_cf_scores)

    # --- Genre similarity ---
    watched_movies = user_ratings[user_ratings > 0].index
    if len(watched_movies) == 0:
        genre_scores = pd.Series(0, index=unrated_movies)
    else:
        genre_scores = genre_similarity_df.loc[unrated_movies, watched_movies].sum(axis=1)

    # --- Year similarity ---
    year_series = pd.to_datetime(df_movies.set_index('movie_id')['release_date'], errors='coerce').dt.year
    watched_years = year_series[watched_movies].dropna()
    if watched_years.empty:
        user_avg_year = np.nan
    else:
        user_avg_year = watched_years.mean()
    year_similarity = 1 - (abs(year_series[unrated_movies] - user_avg_year) / 50)
    year_similarity = year_similarity.fillna(0)

    # --- IMDb rating placeholder (app handles OMDb) ---
    imdb_scores = pd.Series(0.5, index=unrated_movies)

    # --- Popularity ---
    popularity_scores = popularity_norm.reindex(unrated_movies).fillna(0)

    # --- Final hybrid score ---
    hybrid_scores = (
        alpha * user_cf_scores[unrated_movies] +
        beta * item_cf_scores +
        gamma * genre_scores +
        delta * year_similarity +
        epsilon * imdb_scores +
        zeta * popularity_scores
    )

    # Top-N recommendations
    # We return exactly top_n candidate movies here (caller can request a larger pool)
    recommendations = hybrid_scores.sort_values(ascending=False).head(top_n)
    rec_movies = df_movies[df_movies['movie_id'].isin(recommendations.index)][
        ['movie_id', 'movie_title', 'processed_title', 'IMDb_URL', 'genres', 'release_date']
    ].copy()

    # keep original ordering by score
    rec_movies['score'] = rec_movies['movie_id'].map(recommendations.to_dict())
    rec_movies = rec_movies.sort_values(by='score', ascending=False).drop(columns=['score'])
    rec_movies = rec_movies.reset_index(drop=True)

    return rec_movies
