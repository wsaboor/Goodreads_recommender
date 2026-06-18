"""
app.py  —  Goodreads Book Recommender  (Project 2)
===================================================
Run:    streamlit run app.py
        (launch from the folder containing Books.csv and Ratings.csv)

Architecture
------------
Layer 1 – Collaborative Filtering (scikit-surprise)
    • User-Based CF   : KNNWithMeans, cosine similarity, user_based=True
    • Item-Based CF   : KNNWithMeans, cosine similarity, user_based=False
    • SVD             : Matrix factorization, 50 latent factors
    • Baseline        : Item mean (per-book average rating)

Layer 2 – LLM Re-ranking (Google Gemini)
    Takes the Top-N CF candidates + user's stated preferences,
    re-ranks them and returns a short explanation for each pick.
    Book metadata (title, author, year, avg rating) is passed as context.
    The LLM re-ranks from the CF list — it does not generate new titles.
"""

import json
import streamlit as st
import pandas as pd
import numpy as np
from collections import defaultdict

# ── Try surprise; fall back to scipy SVD if unavailable ─────────────────────
try:
    from surprise import Dataset, Reader, SVD, KNNWithMeans
    from surprise.model_selection import train_test_split
    SURPRISE_AVAILABLE = True
except ImportError:
    SURPRISE_AVAILABLE = False
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import svds

# ─────────────────────────── Page config ────────────────────────────────────
st.set_page_config(
    page_title="📚 Book Recommender",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.book-card   { background:#f8f9fa; border-radius:10px; padding:14px 16px;
               margin-bottom:10px; border-left:4px solid #4a90d9; }
.rerank-card { background:#fff8f0; border-radius:10px; padding:14px 16px;
               margin-bottom:10px; border-left:4px solid #e07b39; }
.book-title  { font-size:15px; font-weight:600; color:#1a1a2e; margin:0 0 3px 0; }
.book-meta   { font-size:12px; color:#555; margin:2px 0; }
.pred-badge  { background:#4a90d9; color:white; border-radius:12px;
               padding:2px 10px; font-size:12px; font-weight:600; }
.llm-badge   { background:#e07b39; color:white; border-radius:12px;
               padding:2px 10px; font-size:12px; font-weight:600; }
.expl-text   { font-size:12px; color:#333; font-style:italic; margin-top:5px;
               border-top:1px solid #ddd; padding-top:5px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────── Data loading ───────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    books   = pd.read_csv("Books.csv")
    ratings = pd.read_csv("Ratings.csv")
    books['is_series'] = books['title'].str.contains(r'#\d', na=False)
    return books, ratings

# ─────────────────────────── Model training ─────────────────────────────────
@st.cache_resource(show_spinner=False)
def train_models(_ratings_df):
    """
    Train User-CF, Item-CF, and SVD using scikit-surprise (preferred),
    or scipy SVD as a fallback if surprise is not installed.
    Returns a dict of {model_name: model_object} plus the trainset.
    """
    if SURPRISE_AVAILABLE:
        reader   = Reader(rating_scale=(1, 5))
        data     = Dataset.load_from_df(
            _ratings_df[['user_id', 'book_id', 'rating']], reader)
        # Serve from a model trained on ALL ratings. Accuracy metrics come from the
        # notebook's separate 20% hold-out; for live recommendations we want every
        # rating in the model, and we want trainset.ur to contain every book each
        # user rated so we never recommend one back to them.
        trainset = data.build_full_trainset()

        user_cf = KNNWithMeans(
            k=30, min_k=3,
            sim_options={'name': 'cosine', 'user_based': True},
            verbose=False)
        item_cf = KNNWithMeans(
            k=30, min_k=3,
            sim_options={'name': 'cosine', 'user_based': False},
            verbose=False)
        svd = SVD(n_factors=50, n_epochs=20,
                  lr_all=0.005, reg_all=0.02, random_state=42)

        user_cf.fit(trainset)
        item_cf.fit(trainset)
        svd.fit(trainset)

        return {
            'type': 'surprise',
            'trainset': trainset,
            'User-Based CF (cosine, k=30)': user_cf,
            'Item-Based CF (cosine, k=30)': item_cf,
            'SVD (50 factors) ★ best RMSE': svd,
        }
    else:
        # scipy fallback — SVD only, trained on all ratings (see note above)
        train  = _ratings_df
        users  = sorted(_ratings_df['user_id'].unique())
        items  = sorted(_ratings_df['book_id'].unique())
        u2i    = {u: i for i, u in enumerate(users)}
        i2i    = {it: i for i, it in enumerate(items)}
        gm     = train['rating'].mean()
        um     = train.groupby('user_id')['rating'].mean()

        rows = train['user_id'].map(u2i).values
        cols = train['book_id'].map(i2i).values
        vals = (train['rating'] - train['user_id'].map(um).fillna(gm)).values.astype(np.float32)
        R = csr_matrix((vals, (rows, cols)), shape=(len(users), len(items)))
        U, s, Vt = svds(R, k=50)
        R_hat = U @ np.diag(s) @ Vt

        return {
            'type': 'scipy',
            'R_hat': R_hat, 'u2i': u2i, 'i2i': i2i,
            'idx2item': {i: it for it, i in i2i.items()},
            'global_mean': float(gm), 'user_means': um,
            'SVD (scipy fallback)': 'scipy',
        }


# ─────────────────────────── Recommendation helpers ─────────────────────────
def get_top_n_surprise(model_obj, trainset, user_id, books_df, n):
    """Generate Top-N recommendations using a trained surprise model."""
    try:
        inner_uid = trainset.to_inner_uid(user_id)
        rated_raw = {trainset.to_raw_iid(iid)
                     for (iid, _) in trainset.ur[inner_uid]}
    except ValueError:
        rated_raw = set()

    unrated = set(books_df['book_id'].astype(str)) - {str(r) for r in rated_raw}
    preds   = sorted(
        [(int(bid), model_obj.predict(user_id, int(bid)).est) for bid in unrated],
        key=lambda x: x[1], reverse=True
    )[:n]

    top_ids = [bid for bid, _ in preds]
    est_map  = {bid: est for bid, est in preds}

    rec_df = (books_df[books_df['book_id'].isin(top_ids)]
              [['book_id', 'title', 'authors', 'average_rating',
                'original_publication_year', 'image_url', 'is_series']]
              .copy())
    rec_df['predicted_rating'] = rec_df['book_id'].map(est_map)
    return rec_df.sort_values('predicted_rating', ascending=False).reset_index(drop=True)


def get_top_n_scipy(state, user_id, books_df, rated_ids, n):
    """Generate Top-N recommendations using the scipy SVD fallback."""
    uid  = state['u2i'].get(user_id, 0)
    um   = state['user_means'].get(user_id, state['global_mean'])
    scores = np.clip(state['R_hat'][uid] + um, 1.0, 5.0)

    rated_set = set(rated_ids)
    candidates = [
        (state['idx2item'][i], float(scores[i]))
        for i in range(len(scores))
        if state['idx2item'][i] not in rated_set
           and state['idx2item'][i] in set(books_df['book_id'])
    ]
    candidates.sort(key=lambda x: x[1], reverse=True)
    top      = candidates[:n]
    top_ids  = [bid for bid, _ in top]
    est_map  = {bid: est for bid, est in top}

    rec_df = (books_df[books_df['book_id'].isin(top_ids)]
              [['book_id', 'title', 'authors', 'average_rating',
                'original_publication_year', 'image_url', 'is_series']]
              .copy())
    rec_df['predicted_rating'] = rec_df['book_id'].map(est_map)
    return rec_df.sort_values('predicted_rating', ascending=False).reset_index(drop=True)


# ─────────────────────────── LLM Re-ranking ─────────────────────────────────
def llm_rerank(api_key: str, candidates: list[dict], user_prefs: str) -> list[dict]:
    """
    Re-rank the CF Top-N with Gemini, using the structured-output pattern from the
    Week 4 demo: a `system_instruction` defines the model's behaviour, and a
    `response_schema` (a Pydantic model) forces typed JSON output instead of free text.
    This is the same generate_content + GenerateContentConfig structure shown in class.

    Robustness choice: the schema is {id, reason} — NOT {title, rating, ...}. The model
    only chooses an ordering (by id) and writes a reason; every number and title is then
    looked up from the authoritative candidate list, so the LLM can never alter a
    predicted score or invent a title that wasn't in the CF output.

    Reproducibility choice: temperature is low and the seed is fixed, so the same
    reader + preferences yield a stable ranking (ties to the lecture's reproducibility
    discussion — useful when an LLM sits inside an otherwise deterministic pipeline).

    Returns the candidate dicts in re-ranked order, each with an added 'explanation'.
    """
    from google import genai
    from google.genai import types
    from pydantic import BaseModel

    # --- Output schema: one entry per book, identified by its position id ----------
    class RerankedBook(BaseModel):
        id: int       # 0-based index into the candidate list
        reason: str   # one-sentence justification tied to the reader's preferences

    system_instruction = (
        "You are a personalised book concierge. The candidate books below were already "
        "chosen for this reader by a collaborative-filtering model. Re-rank ALL of them "
        "by how well they fit the reader's stated preferences, best first. Refer to each "
        "book only by its id. Give every book a one-sentence reason (max 25 words) tied "
        "to the reader's preferences. Keep every candidate exactly once; never add, drop, "
        "or invent a book."
    )

    # --- Candidate catalogue passed as context (data, not instructions) ------------
    catalogue = "\n".join(
        f"id={i}: \"{b['title']}\" by {b['authors']} "
        f"(published {int(b['original_publication_year']) if pd.notna(b.get('original_publication_year')) else 'unknown'}, "
        f"avg community rating {b['average_rating']:.2f}, "
        f"CF-predicted {b['predicted_rating']:.2f})"
        for i, b in enumerate(candidates)
    )
    prompt = (
        f"Reader's stated preferences: \"{user_prefs}\"\n\n"
        f"Candidate books:\n{catalogue}"
    )

    client   = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_schema=list[RerankedBook],
            temperature=0.3,   # low → focused, near-deterministic ranking
            seed=42,           # fixed → reproducible output for identical inputs
        ),
    )

    # With a response_schema set, the SDK can return parsed objects directly; fall
    # back to raw-JSON parsing if .parsed is unavailable for any reason.
    try:
        items = [{"id": it.id, "reason": it.reason} for it in (response.parsed or [])]
    except Exception:
        items = json.loads(response.text)

    # Map the model's ordering back onto the authoritative candidate rows. All numbers
    # and titles come from `candidates`, never from the model's response.
    reranked, seen = [], set()
    for it in items:
        idx = it.get("id") if isinstance(it, dict) else None
        if not isinstance(idx, int) or idx in seen or not (0 <= idx < len(candidates)):
            continue
        seen.add(idx)
        row = dict(candidates[idx])
        row["explanation"] = str(it.get("reason", "")).strip()
        reranked.append(row)
    # Safety net: if the model dropped any ids, append them in original CF order so the
    # reader still sees the full Top-N.
    for i, b in enumerate(candidates):
        if i not in seen:
            row = dict(b)
            row["explanation"] = ""
            reranked.append(row)
    return reranked


# ─────────────────────────── Load & train ───────────────────────────────────
with st.spinner("Loading data…"):
    books, ratings = load_data()

with st.spinner("Training models — this takes ~60 s the first time, then cached…"):
    state = train_models(ratings)

model_names = [k for k in state if k not in ('type', 'trainset', 'R_hat', 'u2i',
                                               'i2i', 'idx2item', 'global_mean', 'user_means')]

# ─────────────────────────── Sidebar ────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")

    if not SURPRISE_AVAILABLE:
        st.warning("scikit-surprise not installed. Running scipy SVD only.\n\n"
                   "Install with: `pip install scikit-surprise`")

    model_choice = st.selectbox(
        "Collaborative filtering model",
        model_names,
        help="All three models are trained. SVD has the best RMSE. Among the "
             "personalised models, Item-CF leads on Precision@10 — though the "
             "non-personalised Item-Mean baseline scores higher still (see table below)."
    )
    n_recs = st.slider("Number of CF candidates / recommendations", 5, 20, 10)

    st.divider()
    st.subheader("🤖 LLM Re-ranking")
    st.caption("Gemini re-ranks the CF list and explains each pick.\n"
               "Free key at [aistudio.google.com](https://aistudio.google.com)")

    default_key = ""
    try:
        default_key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        pass

    gemini_key = st.text_input("Gemini API Key", value=default_key,
                                type="password", placeholder="AIza…")
    user_prefs = st.text_area(
        "Your reading preferences",
        placeholder="e.g. I love slow-burn psychological thrillers and recently "
                    "enjoyed Gone Girl. Not a fan of sci-fi.",
        height=110,
    )

    st.divider()
    st.subheader("📊 Model Performance")
    st.caption("Evaluated on 20% held-out test set (surprise)")
    perf = pd.DataFrame([
        {"Model": "Global Mean",   "RMSE": 1.006, "P@10": "0.000†", "R@10": "0.000†"},
        {"Model": "Item Mean",     "RMSE": 0.941, "P@10": "0.783",  "R@10": "0.372"},
        {"Model": "User-CF",       "RMSE": 0.858, "P@10": "0.657",  "R@10": "0.250"},
        {"Model": "Item-CF",       "RMSE": 0.883, "P@10": "0.682",  "R@10": "0.266"},
        {"Model": "SVD ★",         "RMSE": 0.843, "P@10": "0.615",  "R@10": "0.230"},
    ]).set_index("Model")
    st.dataframe(perf, use_container_width=True)
    st.caption("★ best RMSE  · † global mean (3.84) never exceeds ≥4 threshold")

# ─────────────────────────── Main area ──────────────────────────────────────
st.title("📚 Goodreads Book Recommender")
st.caption(
    f"Collaborative filtering (User-CF · Item-CF · SVD) + Gemini LLM re-ranking  ·  "
    f"{ratings.shape[0]:,} ratings · {books.shape[0]:,} books · "
    f"{ratings['user_id'].nunique():,} users"
)

tab_recs, tab_explore = st.tabs(["🎯 Recommendations", "📊 Dataset Explorer"])

# ── Recommendations tab ──────────────────────────────────────────────────────
with tab_recs:
    col_sel, col_info = st.columns([3, 1])

    with col_sel:
        all_users = sorted(ratings['user_id'].unique())
        user_id   = st.selectbox("Select a user", all_users)

    user_hist = ratings[ratings['user_id'] == user_id]
    with col_info:
        st.metric("Ratings by user", len(user_hist))
        st.metric("Their avg rating", f"{user_hist['rating'].mean():.2f}")

    run_btn = st.button("Get Recommendations", type="primary")

    if run_btn:
        # ── Step 1: CF recommendations ──
        with st.spinner(f"Running {model_choice}…"):
            if state['type'] == 'surprise':
                recs = get_top_n_surprise(
                    state[model_choice], state['trainset'], user_id, books, n_recs)
            else:
                rated_ids = set(user_hist['book_id'].tolist())
                recs = get_top_n_scipy(state, user_id, books, rated_ids, n_recs)

        # ── Step 2: LLM re-ranking ──
        reranked = None
        if gemini_key.strip() and user_prefs.strip():
            with st.spinner("LLM re-ranking candidates…"):
                try:
                    reranked = llm_rerank(
                        gemini_key.strip(),
                        recs[['title', 'authors', 'average_rating',
                              'predicted_rating', 'original_publication_year']]
                        .to_dict('records'),
                        user_prefs.strip()
                    )
                except Exception as e:
                    st.warning(f"LLM re-ranking failed: {e}")

        elif gemini_key.strip() and not user_prefs.strip():
            st.info("💡 Enter your reading preferences in the sidebar to enable LLM re-ranking.")

        # ── Display ──
        if reranked:
            # Show LLM reranked results with per-book explanations
            st.success(f"Top {n_recs} recommendations — **LLM re-ranked** for your preferences", icon="🤖")
            st.caption(f"CF model: {model_choice}  ·  Re-ranked by: Gemini gemini-2.5-flash-lite")

            for j, book in enumerate(reranked):
                # Cover image + series flag come from the original recs (looked up by
                # the authoritative title, which the LLM cannot have changed).
                orig = recs[recs['title'] == book['title']]
                img  = orig['image_url'].values[0] if len(orig) > 0 else ""
                is_series = bool(orig['is_series'].values[0]) if len(orig) > 0 else False
                series_tag = " · *series*" if is_series else ""
                yr = book.get('original_publication_year')
                year_str = f" · {int(yr)}" if pd.notna(yr) else ""

                img_col, txt_col = st.columns([1, 5])
                with img_col:
                    if isinstance(img, str) and img.startswith('http'):
                        st.image(img, width=70)
                with txt_col:
                    st.markdown(
                        f"<div class='rerank-card'>"
                        f"<p class='book-title'>{j+1}. {book['title']}</p>"
                        f"<p class='book-meta'>✍️ {book['authors']}{series_tag}{year_str}</p>"
                        f"<p class='book-meta'>"
                        f"⭐ Avg {book['average_rating']:.2f} &nbsp;·&nbsp; "
                        f"🎯 CF predicted: <span class='llm-badge'>{book['predicted_rating']:.2f}</span></p>"
                        f"<p class='expl-text'>💬 {book.get('explanation', '')}</p>"
                        f"</div>",
                        unsafe_allow_html=True
                    )

        else:
            # Show raw CF results
            st.success(
                f"Top {n_recs} CF recommendations for User **{user_id}** · model: **{model_choice}**",
                icon="✅"
            )
            if not gemini_key.strip():
                st.info("🤖 Add a Gemini API key in the sidebar to enable LLM re-ranking with per-book explanations.")

            for i in range(0, len(recs), 2):
                c1, c2 = st.columns(2)
                for col, j in zip([c1, c2], [i, i + 1]):
                    if j < len(recs):
                        row = recs.iloc[j]
                        series_tag = " · *series*" if row.get('is_series') else ""
                        pub_year   = (f" · {int(row['original_publication_year'])}"
                                      if pd.notna(row.get('original_publication_year')) else "")
                        with col:
                            ic, tc = st.columns([1, 4])
                            with ic:
                                img = row.get('image_url', '')
                                if isinstance(img, str) and img.startswith('http'):
                                    st.image(img, width=70)
                            with tc:
                                st.markdown(
                                    f"<div class='book-card'>"
                                    f"<p class='book-title'>{j+1}. {row['title']}</p>"
                                    f"<p class='book-meta'>✍️ {row['authors']}{series_tag}</p>"
                                    f"<p class='book-meta'>⭐ Avg {row['average_rating']:.2f}{pub_year}</p>"
                                    f"<p class='book-meta'>🎯 CF predicted: "
                                    f"<span class='pred-badge'>{row['predicted_rating']:.2f}</span></p>"
                                    f"</div>",
                                    unsafe_allow_html=True
                                )

        # ── Table view ──
        with st.expander("View CF results as table"):
            display = recs[['title', 'authors', 'average_rating', 'predicted_rating']].copy()
            display.columns = ['Title', 'Authors', 'Avg Rating', 'CF Predicted']
            st.dataframe(
                display.style.format({'Avg Rating': '{:.2f}', 'CF Predicted': '{:.2f}'}),
                use_container_width=True, hide_index=True
            )

        # ── User history ──
        with st.expander(f"Books user {user_id} has already rated ({len(user_hist)})"):
            hist_df = (
                user_hist
                .merge(books[['book_id', 'title', 'authors']], on='book_id')
                .sort_values('rating', ascending=False)
                [['title', 'authors', 'rating']]
                .rename(columns={'title': 'Title', 'authors': 'Authors', 'rating': 'Rating'})
                .reset_index(drop=True)
            )
            st.dataframe(hist_df, use_container_width=True, hide_index=True)

# ── Explorer tab ─────────────────────────────────────────────────────────────
with tab_explore:
    st.subheader("Dataset Overview")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Books",          f"{books.shape[0]:,}")
    m2.metric("Rated books",    f"{ratings['book_id'].nunique():,}")
    m3.metric("Users",          f"{ratings['user_id'].nunique():,}")
    m4.metric("Total ratings",  f"{ratings.shape[0]:,}")
    sparsity = 1 - len(ratings) / (ratings['user_id'].nunique() * ratings['book_id'].nunique())
    m5.metric("Sparsity",       f"{sparsity:.2%}")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Rating distribution**")
        rc = ratings['rating'].value_counts().sort_index().reset_index()
        rc.columns = ['Rating', 'Count']
        st.bar_chart(rc.set_index('Rating'), height=260)

    with col_b:
        st.markdown("**Top 10 most-rated books in sample**")
        top10 = (
            ratings.groupby('book_id').size().reset_index(name='Ratings')
            .merge(books[['book_id', 'title']], on='book_id')
            .sort_values('Ratings', ascending=False)
            .head(10)[['title', 'Ratings']]
            .rename(columns={'title': 'Title'})
        )
        st.dataframe(top10, use_container_width=True, hide_index=True, height=260)

    st.markdown("**Books sample**")
    st.dataframe(
        books[['title', 'authors', 'original_publication_year', 'average_rating']]
        .rename(columns={'original_publication_year': 'Year', 'average_rating': 'Avg Rating'})
        .head(30),
        use_container_width=True, hide_index=True
    )
