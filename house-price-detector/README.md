# House Price Detector

A simple but polished beginner-friendly machine learning app built with **Python, Streamlit, Pandas, NumPy, Scikit-learn, and Matplotlib**.

The app creates a synthetic house price dataset, trains a **Linear Regression** model, and predicts house price from user inputs. All prices are displayed in **₹**.

## Features

- Synthetic dataset generated inside the project
- Linear Regression model trained on:
  - Area in square feet
  - Number of bedrooms
  - Number of bathrooms
  - Location rating
  - House age
- Clean Streamlit interface with sidebar controls
- Predicted price shown in a highlighted card
- Sample dataset table
- Area vs price chart
- Model accuracy score displayed in the app

## How the model works

This project uses **supervised machine learning**. The app first creates synthetic training data with house features and a target price in **₹**. A **Linear Regression** model learns how the features relate to the price.

When you enter new values in the sidebar, the model uses the learned pattern to estimate the house price in **₹**. The model is evaluated with an **R² score**, which shows how well the synthetic data pattern is explained by the model.

## Setup

1. Make sure Python is installed.
2. Open a terminal in the project folder.
3. Install the dependencies:

```bash
pip install -r requirements.txt
```

4. Run the Streamlit app:

```bash
streamlit run app.py
```

## Deployment on Streamlit Community Cloud

1. Push this project to a GitHub repository.
2. Go to [Streamlit Community Cloud](https://streamlit.io/cloud).
3. Sign in with GitHub.
4. Click **New app**.
5. Select your repository.
6. Set the main file path to `app.py`.
7. Click **Deploy**.

## Project Structure

```text
house-price-detector/
│── app.py
│── requirements.txt
│── README.md
```

## Notes

- No external dataset is required.
- The dataset is synthetic, so the price output is for demonstration and college project use.
- You can tune the synthetic pricing formula in `app.py` if you want different behavior.
