import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


# ---------- Page setup ----------
st.set_page_config(
    page_title="House Price Detector",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------- Custom styling ----------
st.markdown(
    """
    <style>
        .main {
            background: linear-gradient(180deg, #f7f9fc 0%, #eef4ff 100%);
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .hero {
            padding: 1.5rem 1.75rem;
            border-radius: 22px;
            background: linear-gradient(135deg, #1f3c88 0%, #2c77d4 55%, #6fb1ff 100%);
            color: white;
            box-shadow: 0 18px 45px rgba(31, 60, 136, 0.18);
            margin-bottom: 1.25rem;
        }
        .hero h1, .hero p {
            color: white;
            margin: 0;
        }
        .hero p {
            opacity: 0.92;
            margin-top: 0.5rem;
            font-size: 1.02rem;
        }
        .section-card {
            background: white;
            border-radius: 18px;
            padding: 1.25rem 1.25rem 1rem 1.25rem;
            box-shadow: 0 8px 28px rgba(15, 23, 42, 0.08);
            border: 1px solid rgba(148, 163, 184, 0.16);
            margin-bottom: 1rem;
        }
        .prediction-card {
            padding: 1.35rem 1.5rem;
            border-radius: 20px;
            color: white;
            background: linear-gradient(135deg, #0f766e 0%, #14b8a6 50%, #22c55e 100%);
            box-shadow: 0 18px 38px rgba(20, 184, 166, 0.22);
            border: 1px solid rgba(255, 255, 255, 0.14);
        }
        .prediction-card h2 {
            margin: 0;
            font-size: 1rem;
            font-weight: 600;
            opacity: 0.95;
        }
        .prediction-card .price {
            margin-top: 0.35rem;
            font-size: 2.2rem;
            font-weight: 800;
            letter-spacing: -0.03em;
        }
        .prediction-card .note {
            margin-top: 0.35rem;
            font-size: 0.92rem;
            opacity: 0.95;
        }
        .stMetric {
            background: white;
            border-radius: 16px;
            padding: 0.5rem 0.5rem 0.25rem 0.5rem;
            box-shadow: 0 6px 20px rgba(15, 23, 42, 0.06);
            border: 1px solid rgba(148, 163, 184, 0.16);
        }
        .sidebar .sidebar-content {
            background: linear-gradient(180deg, #f8fbff 0%, #eef6ff 100%);
        }
        .small-caption {
            color: #64748b;
            font-size: 0.92rem;
            margin-top: 0.25rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- Synthetic data generation ----------
@st.cache_data

def generate_house_data(n_samples: int = 250, seed: int = 42) -> pd.DataFrame:
    """Create a synthetic house dataset with realistic ranges and a price formula."""
    rng = np.random.default_rng(seed)

    synthetic_area = rng.integers(600, 4501, n_samples)
    synthetic_bedrooms = rng.integers(1, 7, n_samples)
    synthetic_bathrooms = rng.uniform(1.0, 4.5, n_samples).round(1)
    synthetic_location_rating = rng.integers(1, 11, n_samples)
    synthetic_house_age = rng.integers(0, 41, n_samples)

    # A simple pricing formula with controlled noise so the model can learn the pattern.
    price = (
        synthetic_area * 300
        + synthetic_bedrooms * 18000
        + synthetic_bathrooms * 25000
        + synthetic_location_rating * 35000
        - synthetic_house_age * 4000
        + rng.normal(0, 35000, n_samples)
    )

    data = pd.DataFrame(
        {
            "Area": synthetic_area,
            "Bedrooms": synthetic_bedrooms,
            "Bathrooms": synthetic_bathrooms,
            "Location_Rating": synthetic_location_rating,
            "House_Age": synthetic_house_age,
            "Price": np.maximum(price, 50000).round(0),
        }
    )
    return data


@st.cache_resource

def train_model(data: pd.DataFrame):
    """Train the Linear Regression model and return the model plus metrics."""
    features = ["Area", "Bedrooms", "Bathrooms", "Location_Rating", "House_Age"]
    X = data[features]
    y = data["Price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    trained_model = LinearRegression()
    trained_model.fit(X_train, y_train)
    predictions = trained_model.predict(X_test)

    evaluation_metrics = {
        "r2": r2_score(y_test, predictions),
        "mae": mean_absolute_error(y_test, predictions),
    }
    return trained_model, evaluation_metrics


# ---------- Data and model ----------
dataset = generate_house_data()
model, metrics = train_model(dataset)


# ---------- Helper functions ----------
def format_price(value: float) -> str:
    """Format the predicted house price as a currency string."""
    rounded_value = int(round(value))
    digits = str(abs(rounded_value))

    if len(digits) <= 3:
        formatted = digits
    else:
        last_three = digits[-3:]
        remaining = digits[:-3]
        groups = []

        while remaining:
            groups.append(remaining[-2:])
            remaining = remaining[:-2]

        formatted = ",".join(reversed(groups)) + "," + last_three

    prefix = "-" if rounded_value < 0 else ""
    return f"₹{prefix}{formatted}"


def make_scatter_chart(data: pd.DataFrame):
    """Create a simple area vs. price scatter chart."""
    fig, ax = plt.subplots(figsize=(9, 5))
    scatter = ax.scatter(
        data["Area"],
        data["Price"],
        c=data["Location_Rating"],
        cmap="viridis",
        alpha=0.75,
        edgecolors="white",
        linewidths=0.4,
    )
    ax.set_title("Area vs Price", fontsize=14, fontweight="bold")
    ax.set_xlabel("Area (sq ft)")
    ax.set_ylabel("Price (₹)")
    ax.grid(True, linestyle="--", alpha=0.25)
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("Location Rating")
    fig.tight_layout()
    return fig


# ---------- Header ----------
st.markdown(
    """
    <div class="hero">
        <h1>🏠 House Price Detector</h1>
        <p>Applied Data Science project: predict house prices from simple property details using Linear Regression.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(
    "Beginner-friendly demo app built with synthetic data, scikit-learn, and Streamlit."
)


# ---------- Sidebar inputs ----------
with st.sidebar:
    st.markdown("## 🔧 Property Details")
    st.write("Enter the house information below and click **Predict Price**.")

    with st.form("prediction_form"):
        area = st.slider("Area in square feet", 500, 5000, 1500, step=50)
        bedrooms = st.slider("Number of bedrooms", 1, 6, 3)
        bathrooms = st.slider("Number of bathrooms", 1.0, 5.0, 2.0, step=0.5)
        location_rating = st.slider("Location rating (1 = basic, 10 = premium)", 1, 10, 7)
        house_age = st.slider("House age (years)", 0, 50, 10)

        submitted = st.form_submit_button("Predict Price")

    st.markdown("---")
    st.markdown("### Model Summary")
    st.metric("R² Accuracy", f"{metrics['r2']:.2f}")
    st.metric("Mean Absolute Error", format_price(metrics["mae"]))


# ---------- Main content grid ----------
col1, col2 = st.columns([1.1, 0.9], gap="large")

with col1:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("✨ Prediction Result")
    if submitted:
        user_input = pd.DataFrame(
            [[area, bedrooms, bathrooms, location_rating, house_age]],
            columns=["Area", "Bedrooms", "Bathrooms", "Location_Rating", "House_Age"],
        )
        predicted_price = float(model.predict(user_input)[0])

        st.markdown(
            f"""
            <div class="prediction-card">
                <h2>Estimated House Price</h2>
                <div class="price">{format_price(predicted_price)}</div>
                <div class="note">Based on the values entered in the sidebar.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.info(
            "This is a machine learning estimate in ₹. Real prices can vary based on market trends, condition, and many other factors."
        )
    else:
        st.info("Use the sidebar and click **Predict Price** to see the result card here.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("📊 Area vs Price Chart")
    st.pyplot(make_scatter_chart(dataset), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("🧪 Sample Dataset")
    st.caption("A preview of the synthetic training data used by the model.")
    st.dataframe(dataset.head(10), use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("🧠 How the Model Works")
    st.write(
        """
        - We generate a small synthetic dataset with realistic house features.
        - The model learns the relationship between those features and the target price.
        - Linear Regression finds the best-fitting line in a multi-feature space.
        - The R² score tells us how well the model explains the synthetic pricing pattern.
        """
    )
    st.markdown('</div>', unsafe_allow_html=True)


# ---------- Footer ----------
st.markdown(
    f"""
    <div class="small-caption">
        Model trained on {len(dataset)} synthetic records | R² score: {metrics['r2']:.2f} | MAE: {format_price(metrics['mae'])}
    </div>
    """,
    unsafe_allow_html=True,
)
