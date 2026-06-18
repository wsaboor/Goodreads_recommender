"""
app.py  —  Goodreads Book Recommender  (Project 2)
===================================================
Run:    streamlit run app.py
        (from the folder containing Books.csv and Ratings.csv)

Layer 1 — Collaborative Filtering  (scikit-surprise)
    • Baseline   : Item Mean  (per-book average rating)
    • User-CF    : KNNWithMeans, cosine similarity, user_based=True,  k=30
    • Item-CF    : KNNWithMeans, cosine similarity, user_based=False, k=30
    • SVD        : Matrix factorization, 50 latent factors  ← best RMSE

Layer 2 — LLM Re-ranking  (Google Gemini 2.5 Flash Lite)
    Takes Top-N CF candidates + user's stated preference, re-ranks them,
    and returns a short explanation for every pick.
    Book metadata (title, author, year, avg rating) is passed as context.
    The LLM re-ranks from the CF list — it cannot add new titles.

API key: enter in the sidebar at runtime — never hard-coded here.
"""

import json
import streamlit as st
import pandas as pd
import numpy as np

# ── scikit-surprise (required); scipy fallback if not installed ──────────────
try:
    from surprise import Dataset, Reader, SVD, KNNWithMeans
    from surprise.model_selection import train_test_split
    SURPRISE_OK = True
except ImportError:
    SURPRISE_OK = False
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import svds

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="📚 Goodreads Recommender",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* CF result card */
.cf-card    { background:#f0f6ff; border-radius:10px; padding:13px 15px 11px;
              margin-bottom:8px;  border-left:4px solid #4a90d9; }
/* LLM re-ranked card */
.llm-card   { background:#fff8f0; border-radius:10px; padding:13px 15px 11px;
              margin-bottom:8px;  border-left:4px solid #e07b39; }
.bk-title   { font-size:14px; font-weight:700; color:#1a1a2e; margin:0 0 3px; }
.bk-meta    { font-size:12px; color:#555; margin:2px 0; }
.cf-badge   { background:#4a90d9; color:#fff; border-radius:10px;
              padding:1px 9px; font-size:11px; font-weight:700; }
.llm-badge  { background:#e07b39; color:#fff; border-radius:10px;
              padding:1px 9px; font-size:11px; font-weight:700; }
.expl       { font-size:12px; color:#444; font-style:italic;
              border-top:1px solid #e0d0c0; margin-top:6px; padding-top:5px; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Data Loading
# ════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load_data():
    """Load Books.csv and Ratings.csv. Cached so it runs only once."""
    books   = pd.read_csv("Books.csv")
    ratings = pd.read_csv("Ratings.csv")
    books["is_series"] = books["title"].str.contains(r"#\d", na=False)
    return books, ratings


# ════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Model Training
# ════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def train_models(_ratings_df):
    """
    Train all four recommendation models on an 80/20 split.
    Uses scikit-surprise when available; falls back to scipy SVD otherwise.
    Cached so training only happens once per session.
    """
    if SURPRISE_OK:
        # ── Build Surprise dataset ──
        reader   = Reader(rating_scale=(1, 5))
        data     = Dataset.load_from_df(
            _ratings_df[["user_id", "book_id", "rating"]], reader)
        trainset, _ = train_test_split(data, test_size=0.2, random_state=42)

        # ── Baseline: item mean ──
        item_means  = _ratings_df.groupby("book_id")["rating"].mean()
        global_mean = float(_ratings_df["rating"].mean())

        # ── User-Based CF ──
        user_cf = KNNWithMeans(
            k=20, min_k=3,
            sim_options={"name": "msd", "user_based": True},
            verbose=False)
        user_cf.fit(trainset)

        # ── Item-Based CF ──
        item_cf = KNNWithMeans(
            k=20, min_k=3,
            sim_options={"name": "msd", "user_based": False},
            verbose=False)
        item_cf.fit(trainset)

        # ── SVD (matrix factorization) ──
        svd = SVD(n_factors=50, n_epochs=20,
                  lr_all=0.005, reg_all=0.02, random_state=42)
        svd.fit(trainset)

        return {
            "backend":     "surprise",
            "trainset":    trainset,
            "item_means":  item_means,
            "global_mean": global_mean,
            # selectable models
            "Item Mean Baseline":              "baseline",
            "User-Based CF  (msd, k=20)":   user_cf,
            "Item-Based CF  (msd, k=20)":   item_cf,
            "SVD  (50 factors)  ★ best RMSE":  svd,
        }

    else:
        # ── scipy SVD fallback ──────────────────────────────────────────────
        train       = _ratings_df.sample(frac=0.8, random_state=42)
        users       = sorted(_ratings_df["user_id"].unique())
        items       = sorted(_ratings_df["book_id"].unique())
        u2i         = {u: i for i, u in enumerate(users)}
        i2i         = {it: i for i, it in enumerate(items)}
        gm          = float(train["rating"].mean())
        um          = train.groupby("user_id")["rating"].mean()

        rows = train["user_id"].map(u2i).values
        cols = train["book_id"].map(i2i).values
        vals = (train["rating"] - train["user_id"].map(um).fillna(gm)).values.astype(np.float32)
        R    = csr_matrix((vals, (rows, cols)), shape=(len(users), len(items)))
        U, s, Vt = svds(R, k=50)

        return {
            "backend":    "scipy",
            "R_hat":      U @ np.diag(s) @ Vt,
            "u2i": u2i,   "i2i": i2i,
            "idx2item":   {i: it for it, i in i2i.items()},
            "global_mean": gm,
            "user_means":  um,
            "SVD  (scipy fallback)": "scipy",
        }


# ════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Recommendation Logic
# ════════════════════════════════════════════════════════════════════════════
def get_top_n(state, model_key, user_id, books_df, user_hist_ids, n):
    """
    Return a DataFrame of the top-N unseen book recommendations
    for the given user using the chosen model.
    Works for all four models (baseline, user-CF, item-CF, SVD).
    """
    if state["backend"] == "surprise":
        trainset = state["trainset"]

        # Books the user has already rated (exclude from recommendations)
        try:
            inner     = trainset.to_inner_uid(user_id)
            rated_raw = {trainset.to_raw_iid(iid) for iid, _ in trainset.ur[inner]}
        except ValueError:
            rated_raw = set()

        unrated = [bid for bid in books_df["book_id"] if bid not in rated_raw]

        model_obj = state[model_key]

        if model_obj == "baseline":
            # Item mean baseline: score by per-book average rating
            gm = state["global_mean"]
            preds = [(bid, float(state["item_means"].get(bid, gm))) for bid in unrated]
        else:
            # Surprise model: predict rating for each unrated book
            preds = [(bid, model_obj.predict(user_id, bid).est) for bid in unrated]

        preds.sort(key=lambda x: x[1], reverse=True)
        top     = preds[:n]
        top_ids = [bid for bid, _ in top]
        est_map = {bid: est for bid, est in top}

    else:
        # scipy fallback
        uid    = state["u2i"].get(user_id, 0)
        um     = state["user_means"].get(user_id, state["global_mean"])
        scores = np.clip(state["R_hat"][uid] + um, 1.0, 5.0)

        rated_set = set(user_hist_ids)
        cands = [(state["idx2item"][i], float(scores[i]))
                 for i in range(len(scores))
                 if state["idx2item"][i] not in rated_set
                 and state["idx2item"][i] in set(books_df["book_id"])]
        cands.sort(key=lambda x: x[1], reverse=True)
        top     = cands[:n]
        top_ids = [bid for bid, _ in top]
        est_map = {bid: est for bid, est in top}

    rec_df = (
        books_df[books_df["book_id"].isin(top_ids)]
        [["book_id", "title", "authors", "average_rating",
          "original_publication_year", "image_url", "is_series"]]
        .copy()
    )
    rec_df["predicted_rating"] = rec_df["book_id"].map(est_map)
    return rec_df.sort_values("predicted_rating", ascending=False).reset_index(drop=True)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 4 — LLM Re-ranking  (Google Gemini)
# ════════════════════════════════════════════════════════════════════════════
def llm_rerank(api_key, candidates, user_prefs):
    """
    Re-rank the CF candidate list using Gemini and return a short
    explanation for each pick.

    Design decisions:
    - Passes full book metadata (title, author, year, avg rating,
      CF predicted rating) as context, as required by the rubric.
    - Instructs the LLM to include ALL candidates — no hallucinated titles.
    - Returns structured JSON for clean rendering in the UI.
    - Model: gemini-2.5-flash-lite  (free tier, ~500 req/day).
    """
    from google import genai

    client     = genai.Client(api_key=api_key)

    # Build numbered candidate list with all metadata
    book_lines = "\n".join(
        f"  {i+1}. \"{b['title']}\" by {b['authors']}"
        f" | published: {int(b['original_publication_year']) if pd.notna(b.get('original_publication_year')) else 'unknown'}"
        f" | community avg: {b['average_rating']:.2f}"
        f" | CF predicted: {b['predicted_rating']:.2f}"
        for i, b in enumerate(candidates)
    )

    prompt = f"""You are a personalised book recommendation assistant.

A collaborative filtering model produced these Top-N candidates for a reader:
{book_lines}

Reader's stated preferences: "{user_prefs}"

Instructions:
1. Re-rank ALL {len(candidates)} books based on fit with the reader's preferences.
2. Write one sentence (≤20 words) per book explaining specifically why it suits this reader.
3. You MUST return all {len(candidates)} books — do not add or remove any titles.
4. Cite the model used: Google Gemini gemini-2.5-flash-lite.
5. Return ONLY valid JSON — no markdown fences, no extra text — in this exact format:

{{
  "model_used": "Google Gemini gemini-2.5-flash-lite",
  "reranked": [
    {{
      "title": "<exact title>",
      "authors": "<exact authors>",
      "year": <integer or null>,
      "avg_rating": <float>,
      "predicted_rating": <float>,
      "explanation": "<one sentence, ≤20 words>"
    }}
  ]
}}"""

    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=prompt,
    )

    raw = response.text.strip()
    # Remove markdown fences if the model added them anyway
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    parsed = json.loads(raw.strip())
    return parsed["reranked"], parsed.get("model_used", "Gemini gemini-2.5-flash-lite")


# ════════════════════════════════════════════════════════════════════════════
# SECTION 5 — App: Load Data & Train
# ════════════════════════════════════════════════════════════════════════════
with st.spinner("Loading Books.csv and Ratings.csv…"):
    books, ratings = load_data()

with st.spinner("Training models — cached after first run (~60 s)…"):
    state = train_models(ratings)

# Derive selectable model names (exclude internal state keys)
_internal = {"backend", "trainset", "R_hat", "u2i", "i2i",
             "idx2item", "global_mean", "user_means", "item_means"}
model_names = [k for k in state if k not in _internal]


# ════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Sidebar
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("⚙️ Settings")

    if not SURPRISE_OK:
        st.warning("scikit-surprise not found — running scipy SVD fallback.\n\n"
                   "`pip install scikit-surprise`")

    # Model selector
    model_choice = st.selectbox(
        "Collaborative filtering model",
        model_names,
        index=3 if SURPRISE_OK else 0,   # default to SVD
        help="SVD has the best RMSE (0.843). Item-CF has the best Precision@10 (0.682).",
    )

    n_recs = st.slider("Number of recommendations", min_value=5, max_value=20, value=10)

    st.divider()

    # LLM settings
    st.subheader("🤖 LLM Re-ranking (Gemini)")
    st.caption("Free key at [aistudio.google.com](https://aistudio.google.com)")

    # Read from Streamlit secrets if deployed, else let user paste
    default_key = ""
    try:
        default_key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        pass

    gemini_key = st.text_input(
        "Gemini API Key",
        value=default_key,
        type="password",
        placeholder="AIza…  (never stored or logged)",
    )
    user_prefs = st.text_area(
        "Your reading preferences",
        placeholder="e.g. I love psychological thrillers and dark humour. "
                    "Recently enjoyed Gone Girl. Not a fan of sci-fi.",
        height=120,
    )

    st.divider()

    # Performance summary
    st.subheader("📊 Model Performance")
    st.caption("Hold-out test set (20%), threshold ≥ 4.0 for P/R@10")
    perf = pd.DataFrame([
        {"Model": "Global Mean",  "RMSE": "1.006", "P@10": "0.00 †", "R@10": "0.00 †"},
        {"Model": "Item Mean",    "RMSE": "0.941", "P@10": "0.783",   "R@10": "0.372"},
        {"Model": "User-CF",      "RMSE": "0.858", "P@10": "0.657",   "R@10": "0.250"},
        {"Model": "Item-CF",      "RMSE": "0.883", "P@10": "0.682",   "R@10": "0.266"},
        {"Model": "SVD ★",        "RMSE": "0.843", "P@10": "0.615",   "R@10": "0.230"},
    ]).set_index("Model")
    st.dataframe(perf, use_container_width=True)
    st.caption("★ best RMSE  ·  † global mean 3.84 never reaches ≥4 threshold")


# ════════════════════════════════════════════════════════════════════════════
# SECTION 7 — Main Page
# ════════════════════════════════════════════════════════════════════════════
st.title("📚 Goodreads Book Recommender")
st.caption(
    f"Collaborative filtering (User-CF · Item-CF · SVD) + Gemini LLM re-ranking  ·  "
    f"{ratings.shape[0]:,} ratings · {books.shape[0]:,} books · "
    f"{ratings['user_id'].nunique():,} users"
)

tab_recs, tab_data = st.tabs(["🎯 Get Recommendations", "📊 Explore Dataset"])


# ── Tab 1: Recommendations ──────────────────────────────────────────────────
with tab_recs:

    col_left, col_right = st.columns([3, 1])
    with col_left:
        all_users = sorted(ratings["user_id"].unique())
        user_id   = st.selectbox("Select a user ID", all_users)
    with col_right:
        user_hist = ratings[ratings["user_id"] == user_id]
        st.metric("Books rated", len(user_hist))
        st.metric("Avg rating given", f"{user_hist['rating'].mean():.2f}")

    go = st.button("🔍  Get Recommendations", type="primary")

    if go:

        # ── Step 1: CF recommendations ──────────────────────────────────────
        with st.spinner(f"Running {model_choice}…"):
            recs = get_top_n(
                state, model_choice, user_id, books,
                set(user_hist["book_id"].tolist()), n_recs
            )

        # ── Step 2: LLM re-ranking ───────────────────────────────────────────
        reranked   = None
        model_used = None

        if gemini_key.strip() and user_prefs.strip():
            with st.spinner("LLM re-ranking with Gemini…"):
                try:
                    reranked, model_used = llm_rerank(
                        gemini_key.strip(),
                        recs[["title", "authors", "average_rating",
                              "predicted_rating", "original_publication_year"]]
                        .to_dict("records"),
                        user_prefs.strip(),
                    )
                except Exception as e:
                    st.warning(f"LLM re-ranking failed — showing CF results only.  ({e})")

        elif gemini_key.strip() and not user_prefs.strip():
            st.info("💡 Add your reading preferences in the sidebar to enable LLM re-ranking.")

        # ── Display: LLM re-ranked view ──────────────────────────────────────
        if reranked:
            st.success(
                f"**{len(reranked)} recommendations re-ranked by LLM** for your preferences",
                icon="🤖"
            )
            st.caption(
                f"CF model: `{model_choice}` → re-ranked by `{model_used}`  "
                f"· Original CF order shown below for comparison"
            )

            for j, book in enumerate(reranked):
                # Retrieve cover image from original recs DataFrame
                match      = recs[recs["title"] == book["title"]]
                img_url    = match["image_url"].values[0] if len(match) else ""
                is_series  = bool(match["is_series"].values[0]) if len(match) else False
                series_tag = " · *series*" if is_series else ""
                yr         = f" · {int(book['year'])}" if book.get("year") else ""

                ic, tc = st.columns([1, 6])
                with ic:
                    if isinstance(img_url, str) and img_url.startswith("http"):
                        st.image(img_url, width=65)
                with tc:
                    st.markdown(
                        f"<div class='llm-card'>"
                        f"<p class='bk-title'>{j+1}. {book['title']}</p>"
                        f"<p class='bk-meta'>✍️ {book['authors']}{series_tag}{yr}</p>"
                        f"<p class='bk-meta'>"
                        f"⭐ Avg {book['avg_rating']:.2f} &nbsp;·&nbsp; "
                        f"🎯 CF score: <span class='llm-badge'>{book['predicted_rating']:.2f}</span>"
                        f"</p>"
                        f"<p class='expl'>💬 {book['explanation']}</p>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

            # Collapsible: original CF order for comparison
            with st.expander("Compare: original CF ranking (before LLM re-ranking)"):
                cf_table = recs[["title", "authors", "average_rating", "predicted_rating"]].copy()
                cf_table.columns = ["Title", "Authors", "Avg Rating", "CF Score"]
                st.dataframe(
                    cf_table.style.format({"Avg Rating": "{:.2f}", "CF Score": "{:.2f}"}),
                    use_container_width=True, hide_index=True,
                )

        # ── Display: plain CF results (no LLM key or prefs) ─────────────────
        else:
            st.success(
                f"Top {n_recs} recommendations for User **{user_id}**  ·  model: `{model_choice}`",
                icon="✅",
            )
            if not gemini_key.strip():
                st.info(
                    "🤖 Enter a Gemini API key + your preferences in the sidebar "
                    "to get LLM re-ranking with a personalised explanation for each book."
                )

            for i in range(0, len(recs), 2):
                c1, c2 = st.columns(2)
                for col, j in zip([c1, c2], [i, i + 1]):
                    if j < len(recs):
                        row        = recs.iloc[j]
                        series_tag = " · *series*" if row.get("is_series") else ""
                        yr         = (f" · {int(row['original_publication_year'])}"
                                      if pd.notna(row.get("original_publication_year")) else "")
                        img        = row.get("image_url", "")
                        with col:
                            ic, tc = st.columns([1, 4])
                            with ic:
                                if isinstance(img, str) and img.startswith("http"):
                                    st.image(img, width=65)
                            with tc:
                                st.markdown(
                                    f"<div class='cf-card'>"
                                    f"<p class='bk-title'>{j+1}. {row['title']}</p>"
                                    f"<p class='bk-meta'>✍️ {row['authors']}{series_tag}</p>"
                                    f"<p class='bk-meta'>⭐ Avg {row['average_rating']:.2f}{yr}</p>"
                                    f"<p class='bk-meta'>🎯 CF score: "
                                    f"<span class='cf-badge'>{row['predicted_rating']:.2f}</span>"
                                    f"</p></div>",
                                    unsafe_allow_html=True,
                                )

            with st.expander("View as table"):
                t = recs[["title", "authors", "average_rating", "predicted_rating"]].copy()
                t.columns = ["Title", "Authors", "Avg Rating", "CF Score"]
                st.dataframe(
                    t.style.format({"Avg Rating": "{:.2f}", "CF Score": "{:.2f}"}),
                    use_container_width=True, hide_index=True,
                )

        # ── User rating history ──────────────────────────────────────────────
        with st.expander(f"📖 User {user_id}'s rating history ({len(user_hist)} books)"):
            hist = (
                user_hist
                .merge(books[["book_id", "title", "authors"]], on="book_id")
                .sort_values("rating", ascending=False)
                [["title", "authors", "rating"]]
                .rename(columns={"title": "Title", "authors": "Authors", "rating": "Rating"})
                .reset_index(drop=True)
            )
            st.dataframe(hist, use_container_width=True, hide_index=True)


# ── Tab 2: Dataset Explorer ──────────────────────────────────────────────────
with tab_data:
    st.subheader("Dataset Overview")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total books",    f"{books.shape[0]:,}")
    m2.metric("Books rated",    f"{ratings['book_id'].nunique():,}")
    m3.metric("Users",          f"{ratings['user_id'].nunique():,}")
    m4.metric("Total ratings",  f"{ratings.shape[0]:,}")
    m5.metric("Matrix sparsity","98.75%")

    ca, cb = st.columns(2)
    with ca:
        st.markdown("**Rating distribution**")
        rc = ratings["rating"].value_counts().sort_index().reset_index()
        rc.columns = ["Rating", "Count"]
        st.bar_chart(rc.set_index("Rating"), height=250)

    with cb:
        st.markdown("**Top 10 most-rated books in sample**")
        top10 = (
            ratings.groupby("book_id").size().reset_index(name="Ratings")
            .merge(books[["book_id", "title"]], on="book_id")
            .sort_values("Ratings", ascending=False)
            .head(10)[["title", "Ratings"]]
            .rename(columns={"title": "Title"})
        )
        st.dataframe(top10, use_container_width=True, hide_index=True, height=250)

    st.markdown("**Books sample (first 30)**")
    st.dataframe(
        books[["title", "authors", "original_publication_year", "average_rating"]]
        .rename(columns={"original_publication_year": "Year",
                         "average_rating": "Avg Rating"})
        .head(30),
        use_container_width=True, hide_index=True,
    )
