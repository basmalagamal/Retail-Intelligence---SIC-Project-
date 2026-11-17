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
Final app/
│
├── app/
│   ├── __init__.py
│   ├── main.py                         # FastAPI application
│   ├── recsys.py                       # Recommendation engine (CLIP similarity, best sellers)
│   └── churn.py                        # Customer churn prediction service
│
├── static/
│   ├── index.html                      # Main web interface
│   ├── style.css                       # UI styling
│   ├── app.js                          # Shop frontend logic
│   ├── churn.js                        # Churn dashboard logic
│   └── churn_form.js                   # Add Customer form
│
├── models/                             # Churn model and metadata
│   ├── random_forest_model.joblib
│   ├── feature_names.json
│   ├── feature_schema.csv
│   ├── model_parameters.json
│   ├── evaluation_metrics.json
│   ├── data_split_info.json
│   └── churn_requirements.txt
│
├── data/                               # Dataset files
│   ├── articles_sampled.csv
│   ├── customers_sampled.csv
│   ├── final_retrieval_comparison.csv
│   ├── hybrid_recommendations.csv
│   ├── transactions_sampled.csv
│   └── images/
|       └── ...
│
├── emb_clip.npy                       # CLIP image embeddings
├── ids_resnet.csv                     # Article IDs, same order as the embeddings
└── requirements.txt
```


## Deployment
Python (<strong>FastAPI</strong>) backend with a lightweight frontend built using <strong>HTML</strong>, <strong>CSS</strong>, and <strong>JavaScript</strong>, providing two main modules:
- Shop: Displays overall and category-based bestsellers, recommends similar products when an item is selected, and supports image-based product search using CLIP embeddings.
- Churn Prediction: Provides an interactive dashboard, customer lookup, and an Add Customer form that predicts churn using a Random Forest pipeline, along with personalized retention recommendations.

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

## Automation

Integrated <strong>Make.com</strong> with the <strong>FastAPI</strong> backend using <strong>ngrok</strong> to expose the local API through a public HTTPS endpoint.

The automation allows users to send product images through Telegram and receive visually similar products using CLIP-based image embeddings.

### Automation Workflow
```
User (Telegram App)
       │
       │ 1. Sends a photo
       ▼
Telegram Bot API
       │
       │ 2. Returns update with file_id
       ▼
Make.com Scenario
       │
       ├── 3. HTTP getFile → Get file_path
       │
       ├── 4. HTTP Download → Get image binary
       │
       └── 5. HTTP POST → multipart/form-data (file, k=5)
                    │
                    ▼
              ngrok Tunnel
       (Public HTTPS → localhost:8000)
                    │
                    ▼
               FastAPI App
                    │
                    ▼
       POST /api/search-by-image
                    │
                    ▼
          CLIP Image Embedding
                    │
                    ▼
           Similarity Search
                    │
                    ▼
             Top-5 Products
                    │
                    │ 6. JSON Response (results[])
                    ▼
             Make.com Scenario
                    │
                    ├── 7. Format Recommendations
                    │
                    └── 8. Send Telegram Message
                                │
                                ▼
                         Telegram Bot API
                                │
                                ▼
                       User Receives Similar
                         Product Recommendations
```

## Future Improvements
- Improve Candidate Retrieval:
Expand the candidate pool using collaborative filtering, user behavior, and hybrid retrieval techniques to capture more future purchases.

- Enhance Ranking Performance:
Experiment with advanced ranking models and additional customer-product interaction features to improve recommendation relevance.

- Optimize Image-Based Recommendations:
Fine-tune CLIP embeddings and explore multimodal approaches combining visual similarity with purchase history and product metadata.

- Personalize Recommendations:
Incorporate customer preferences, browsing behavior, and purchase history to deliver more personalized recommendations.

- Improve Evaluation:
Test on larger datasets and explore additional metrics such as NDCG@12, MAP@12, and coverage to assess recommendation quality.

## Application Demo
https://github.com/user-attachments/assets/1633444b-5ca2-40ad-814f-1ff422a3bcd2

## Automation Demo
https://github.com/user-attachments/assets/fd2d4348-1018-462f-a5d0-b63b8e35863c
