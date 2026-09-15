import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------

st.set_page_config(
    page_title="CinePredict AI",
    page_icon="🎬",
    layout="wide"
)

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

@st.cache_data
def load_data():

    df = pd.read_csv("data/tmdb_5000_movies.csv")

    # Convert columns to numbers
    numeric_columns = [
        "budget",
        "revenue",
        "runtime",
        "popularity",
        "vote_average",
        "vote_count"
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # Release year
    df["release_date"] = pd.to_datetime(
        df["release_date"],
        errors="coerce"
    )

    df["release_year"] = df["release_date"].dt.year

    # Keep valid movies
    df = df[
        (df["budget"] > 0) &
        (df["revenue"] > 0)
    ].copy()

    # Calculate ROI
    df["roi"] = (
        (df["revenue"] - df["budget"])
        / df["budget"]
    )

    # Same classification method used in Jupyter
    q25 = df["roi"].quantile(0.25)
    q75 = df["roi"].quantile(0.75)

    def classify_success(roi):

        if roi <= q25:
            return "Flop"

        elif roi >= q75:
            return "Hit"

        else:
            return "Average"

    df["success"] = df["roi"].apply(
        classify_success
    )

    return df


# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------

@st.cache_resource
def load_model():

    return joblib.load(
        "models/cinepredict_model.pkl"
    )


df = load_data()
model = load_model()


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

st.sidebar.title("🎬 CinePredict AI")

st.sidebar.write(
    "Movie Success Prediction Platform"
)

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "📊 Explore & EDA",
        "🤖 Predict Movie",
        "📈 Model Insights"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info(
    """
    **Machine Learning**

    Random Forest Classifier

    Prediction Classes:
    • Flop
    • Average
    • Hit
    """
)


# =========================================================
# HOME
# =========================================================

if page == "🏠 Home":

    st.title("🎬 CinePredict AI")

    st.subheader(
        "Movie Success Prediction & Analytics"
    )

    st.write(
        """
        CinePredict AI uses machine learning to analyze
        historical movie data and classify movies as
        **Flop, Average, or Hit**.
        """
    )

    st.markdown("---")

    # Metrics

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "🎬 Movies Analyzed",
            f"{len(df):,}"
        )

    with col2:
        st.metric(
            "⭐ Average Rating",
            f"{df['vote_average'].mean():.2f}"
        )

    with col3:
        st.metric(
            "🔥 Avg Popularity",
            f"{df['popularity'].mean():.2f}"
        )

    with col4:
        st.metric(
            "🏆 Categories",
            "3"
        )

    st.markdown("---")

    st.subheader(
        "🚀 What Can CinePredict AI Do?"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(
            """
            ### 📊 Explore

            Explore movie budgets, revenue,
            ratings and popularity using
            interactive visualizations.
            """
        )

    with col2:

        st.markdown(
            """
            ### 🔍 Analyze

            Discover patterns and factors
            associated with movie success.
            """
        )

    with col3:

        st.markdown(
            """
            ### 🤖 Predict

            Enter movie characteristics and
            generate an AI-based prediction.
            """
        )

    st.markdown("---")

    st.subheader("🧠 How It Works")

    st.write(
        """
        Historical Movie Data
        ↓
        Data Cleaning
        ↓
        Exploratory Data Analysis
        ↓
        ROI-based Success Classification
        ↓
        Machine Learning
        ↓
        Random Forest
        ↓
        Movie Success Prediction
        """
    )


# =========================================================
# EXPLORE & EDA
# =========================================================

elif page == "📊 Explore & EDA":

    st.title("📊 Explore Movie Data")

    st.write(
        "Use the filter and interactive charts to explore the dataset."
    )

    st.subheader("🔎 Filter Movies")

    min_rating = st.slider(
        "Minimum Movie Rating",
        min_value=0.0,
        max_value=10.0,
        value=0.0,
        step=0.1
    )

    filtered_df = df[
        df["vote_average"] >= min_rating
    ]

    st.write(
        f"Showing **{len(filtered_df):,} movies**"
    )

    st.markdown("---")

    # Metrics

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Movies",
            f"{len(filtered_df):,}"
        )

    with col2:
        st.metric(
            "Avg Rating",
            f"{filtered_df['vote_average'].mean():.2f}"
        )

    with col3:
        st.metric(
            "Avg Budget",
            f"${filtered_df['budget'].mean():,.0f}"
        )

    with col4:
        st.metric(
            "Avg Revenue",
            f"${filtered_df['revenue'].mean():,.0f}"
        )

    st.markdown("---")

    # Budget vs Revenue

    st.subheader("💰 Budget vs Revenue")

    fig = px.scatter(
        filtered_df,
        x="budget",
        y="revenue",
        size="popularity",
        hover_name="title",
        title="Movie Budget vs Revenue"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # Rating Distribution

    st.subheader("⭐ Rating Distribution")

    fig = px.histogram(
        filtered_df,
        x="vote_average",
        nbins=20,
        title="Distribution of Movie Ratings"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # Popularity vs Revenue

    st.subheader("🔥 Popularity vs Revenue")

    fig = px.scatter(
        filtered_df,
        x="popularity",
        y="revenue",
        size="vote_count",
        hover_name="title",
        title="Popularity vs Revenue"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # Success Distribution

    st.subheader("🏆 Movie Success Distribution")

    success_counts = (
        filtered_df["success"]
        .value_counts()
        .reset_index()
    )

    success_counts.columns = [
        "Success",
        "Movies"
    ]

    fig = px.bar(
        success_counts,
        x="Success",
        y="Movies",
        title="Flop vs Average vs Hit"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # Budget by success

    st.subheader(
        "💵 Budget Distribution by Success"
    )

    fig = px.box(
        filtered_df,
        x="success",
        y="budget",
        title="Budget by Movie Success"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# =========================================================
# PREDICTION
# =========================================================

elif page == "🤖 Predict Movie":

    st.title("🤖 Movie Success Predictor")

    st.write(
        """
        Enter movie characteristics below and
        CinePredict AI will classify the movie as
        **Flop, Average, or Hit**.
        """
    )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("🎥 Movie Information")

        budget = st.number_input(
            "💰 Budget ($)",
            min_value=0.0,
            value=50_000_000.0,
            step=1_000_000.0
        )

        runtime = st.number_input(
            "⏱️ Runtime (minutes)",
            min_value=1.0,
            max_value=300.0,
            value=120.0,
            step=1.0
        )

        popularity = st.number_input(
            "🔥 Popularity",
            min_value=0.0,
            value=50.0,
            step=1.0
        )

    with col2:

        st.subheader("⭐ Audience Information")

        vote_average = st.slider(
            "⭐ Average Rating",
            min_value=0.0,
            max_value=10.0,
            value=7.0,
            step=0.1
        )

        vote_count = st.number_input(
            "👥 Vote Count",
            min_value=0,
            value=5000,
            step=100
        )

        release_year = st.number_input(
            "📅 Release Year",
            min_value=1900,
            max_value=2035,
            value=2026,
            step=1
        )

    st.markdown("---")

    if st.button(
        "🔮 PREDICT MOVIE SUCCESS",
        use_container_width=True
    ):

        new_movie = pd.DataFrame({
            "budget": [budget],
            "runtime": [runtime],
            "popularity": [popularity],
            "vote_average": [vote_average],
            "vote_count": [vote_count],
            "release_year": [release_year]
        })

        prediction = model.predict(
            new_movie
        )[0]

        probabilities = model.predict_proba(
            new_movie
        )[0]

        classes = model.classes_

        confidence = (
            max(probabilities) * 100
        )

        st.markdown("---")

        # Result

        if prediction == "Hit":

            st.success(
                f"🟢 LIKELY HIT\n\n"
                f"Confidence: {confidence:.1f}%"
            )

        elif prediction == "Average":

            st.warning(
                f"🟡 LIKELY AVERAGE\n\n"
                f"Confidence: {confidence:.1f}%"
            )

        else:

            st.error(
                f"🔴 LIKELY FLOP\n\n"
                f"Confidence: {confidence:.1f}%"
            )

        # Probability chart

        probability_df = pd.DataFrame({
            "Category": classes,
            "Probability": probabilities
        })

        fig = px.bar(
            probability_df,
            x="Category",
            y="Probability",
            text_auto=".1%",
            title="Prediction Probability"
        )

        fig.update_yaxes(
            tickformat=".0%"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.subheader("📋 Movie Input")

        st.dataframe(
            new_movie,
            use_container_width=True
        )


# =========================================================
# MODEL INSIGHTS
# =========================================================

elif page == "📈 Model Insights":

    st.title("📈 Model Insights")

    st.subheader(
        "🌲 Factors Influencing Movie Success"
    )

    rf = model.named_steps["model"]

    importance = rf.feature_importances_

    features = [
        "budget",
        "runtime",
        "popularity",
        "vote_average",
        "vote_count",
        "release_year"
    ]

    importance_df = pd.DataFrame({
        "Feature": features,
        "Importance": importance
    }).sort_values(
        "Importance",
        ascending=True
    )

    fig = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Feature Importance"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown("---")

    st.subheader("📊 Feature Importance Table")

    display_df = importance_df.sort_values(
        "Importance",
        ascending=False
    ).copy()

    display_df["Importance"] = (
        display_df["Importance"] * 100
    ).round(2)

    display_df["Importance"] = (
        display_df["Importance"].astype(str)
        + "%"
    )

    st.dataframe(
        display_df,
        use_container_width=True
    )

    st.markdown("---")

    st.subheader(
        "🎯 Success Category Analysis"
    )

    success_analysis = (
        df.groupby("success")
        .agg(
            Movies=("title", "count"),
            Average_Budget=("budget", "mean"),
            Average_Revenue=("revenue", "mean"),
            Average_Rating=("vote_average", "mean"),
            Average_Popularity=("popularity", "mean")
        )
        .reset_index()
    )

    st.dataframe(
        success_analysis,
        use_container_width=True
    )

    st.markdown("---")

    st.info(
        """
        Feature importance shows which variables were most
        useful to the Random Forest model. It does not mean
        that a feature directly causes movie success.
        """
    )