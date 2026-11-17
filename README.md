# Retail Intelligence: AI-Powered Recommendation & Customer Retention System

## Overview

An AI-powered retail intelligence platform designed to enhance the e-commerce experience through customer churn prediction, personalized product recommendations, and computer vision–based visual product search.

The project combines machine learning and computer vision to help retailers understand customer behavior, identify customers at risk of churn, recommend relevant fashion products, and enable users to discover visually similar items using images.

## Key Features

- **Customer Churn Prediction:** Predicts customers at risk of disengagement based on historical purchasing behavior.
- **Personalized Product Recommendations:** Suggests relevant fashion items based on customer preferences and shopping behavior.
- **Visual Product Search:** Enables users to search for visually similar fashion products using an image.
- **Image Captioning:** Generates textual descriptions of fashion images to support image understanding and product discovery.
- **Retail Intelligence:** Connects predictive analytics and recommendation capabilities to support data-driven retail decisions.

## Project Objectives

- Improve customer retention through proactive churn identification.
- Deliver personalized shopping experiences.
- Simplify product discovery through image-based search.
- Support customer engagement and repeat purchases.
- Integrate multiple AI capabilities into a unified retail solution.

## Dataset

The project uses the **H&M Personalized Fashion Recommendations** dataset, which contains fashion product information, customer data, transaction history, and product images.

These data sources support customer behavior analysis, churn modeling, personalized recommendations, and visual product discovery.

## Technologies

The project uses Python and machine learning and computer vision techniques. Specific libraries, frameworks, and tools will be documented as the implementation is finalized.

## Project Structure
```
app/            FastAPI app (main.py), recommender (recsys.py), churn service (churn.py)
static/         index.html, style.css, app.js, churn.js, churn_form.js
models/         churn model + metadata (random_forest_model.joblib, feature_names.json, ...)
scripts/        fetch_data.py (downloads the Kaggle dataset)
tests/          API tests, UI test, synthetic fixture generator
data/           dataset files (not committed)
emb_clip.npy, ids_resnet.csv   CLIP embeddings and article ids (project root)
```

## Future Improvements

## Deployment
FastAPI backend + plain HTML/CSS/JS frontend for:
- **Shop:** best sellers (overall and per category), similar items on click, and search by uploaded image (CLIP embeddings).
- **Churn:** dashboard, customer lookup, and an "Add Customer" form that predicts churn with a Random Forest pipeline, plus retention recommendations.

Runs locally at http://127.0.0.1:8000

### API
| Endpoint | Purpose |
|---|---|
| `GET /api/bestsellers`, `/api/bestsellers/by-category` | Best sellers |
| `GET /api/product/{article_id}` | Product + similar items |
| `POST /api/search-by-image` | Visually similar products for an upload |
| `GET /api/churn/summary` | Dashboard data |
| `GET /api/churn/customer/{id}` | Churn prediction for an existing customer |
| `GET /api/churn/schema`, `POST /api/churn/predict` | Add Customer form schema and prediction |

Interactive docs: http://127.0.0.1:8000/docs

## Application Demo
https://github.com/user-attachments/assets/1633444b-5ca2-40ad-814f-1ff422a3bcd2

## Automation Demo
https://github.com/user-attachments/assets/fd2d4348-1018-462f-a5d0-b63b8e35863c
