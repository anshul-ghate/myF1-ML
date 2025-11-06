# Data Science Design Document: F1 Fan Analytics & Prediction Platform

- **Version**: 2.0
- **Date**: 2025-11-06
- **Author**: David, Data Analyst

---

## 1. Overview

This document outlines the data science strategy for the F1 Fan Analytics & Prediction Platform, revised to support a **full-stack architecture using Supabase**. It details the methodologies for server-side data ingestion, the database schema, the machine learning model training and prediction pipeline, and the implementation of the "Agent" and Generative AI assistant. This design replaces the previous client-centric approach.

## 2. Data Ingestion and Storage

The platform's data integrity is maintained by a centralized PostgreSQL database on Supabase, which serves as the single source of truth. Data is populated and updated through a robust, server-side ingestion pipeline.

### 2.1. Data Sources

1.  **Ergast API**: Primary source for historical race results, driver/constructor standings, season schedules, and circuit information.
2.  **FastF1 API**: Used for detailed session data, including lap-by-lap times, positions, tire compounds, and pit stops, which are crucial for feature engineering.

### 2.2. Ingestion Pipeline

-   **Initial Load**: A dedicated **Data Ingestion Service** (running as an external compute service) will perform a one-time bulk import of historical data from the Ergast and FastF1 APIs into the Supabase database.
-   **Live Data Handling**: During live race sessions, this service will poll the F1 Data API at a high frequency (e.g., every 5 seconds). The incoming data (lap times, positions, etc.) will be broadcasted via **Supabase Realtime** to subscribed clients and persisted to the `race_results` table upon completion.
-   **Data Storage**: All data will be stored in the Supabase PostgreSQL database, following the schema defined in the System Design Document. This eliminates the need for client-side caching (`localStorage`) and ensures data consistency.

## 3. Database Schema

All data science components will interact with the central PostgreSQL database. The schema is detailed in the **System Design Document's ER Diagram**. Key tables for data science include:

-   `drivers`, `constructors`, `circuits`, `seasons`, `races`: Core entities of the F1 domain.
-   `race_results`: Stores final and in-progress results, serving as the primary source for model training features.
-   `predictions`: Stores the outputs of our ML models, such as predicted race winners and their confidence scores.
-   `profiles`: Stores user-specific information, like favorite drivers, which can be used for personalizing the user experience.

## 4. Machine Learning Model Pipeline

The ML pipeline is designed as a server-centric workflow, moving all training and prediction generation off the client. This allows for more complex models and ensures predictions are always up-to-date without client-side computation.

### 4.1. Race Winner Prediction

-   **Objective**: Predict the probability of each driver winning an upcoming race.
-   **Model**: **Gradient Boosting Classifier (e.g., XGBoost, LightGBM)**. This model offers high accuracy and can handle complex interactions between features.
-   **Features**:
    -   Driver's rolling average performance (grid position, finishing position, points) over the last 5-10 races.
    -   Constructor's rolling average performance.
    -   Driver's historical performance at the specific circuit.
    -   Qualifying position for the upcoming race.
    -   Championship standing.
    -   Engine and component usage data, if available.
-   **Training & Prediction Process**:
    1.  The **ML Model Worker** fetches training data from the `race_results` and related tables in the Supabase DB.
    2.  The model is retrained after each race to incorporate the latest results.
    3.  The trained model artifact (e.g., a `.pkl` or `.joblib` file) is versioned and saved to **Supabase Storage**.
    4.  After qualifying, the worker generates predictions for the upcoming race and stores them in the `predictions` table, linked to the `race_id` and `driver_id`.

### 4.2. Tire Strategy & Pit Stop Prediction

-   **Objective**: Predict tire degradation and optimal pit stop windows.
-   **Model**: A combination of **Linear Regression** (to model lap time drop-off per tire compound) and a **survival model** (to predict the probability of a tire lasting a certain number of laps).
-   **Features**:
    -   `lap_number` (on current tire stint)
    -   `tire_compound`
    -   `track_id` (as a categorical feature)
    -   `fuel_load_estimate` (calculated based on lap number)
    -   `track_temperature` and `air_temperature` (if available).
-   **Process**: This model will also be managed by the ML Model Worker, with its outputs potentially stored as serialized JSON objects or in a dedicated table if needed for the frontend.

## 5. The "Agent": Automated Data & Model Maintenance

The "Agent" is a collection of automated server-side processes responsible for keeping the platform's data and models current, accurate, and self-improving.

### 5.1. Data Freshness Agent

-   **Implementation**: A **Supabase Edge Function** scheduled to run periodically (e.g., daily) using `pg_cron`.
-   **Responsibilities**:
    -   **Scan for Updates**: The function will query external APIs (e.g., Ergast) to check for changes in driver lineups, constructor names, or the race calendar.
    -   **Update Database**: If discrepancies are found, the function will update the `drivers`, `constructors`, and `races` tables accordingly. For example, if a driver switches teams, their `constructor_id` for the current season will be updated.
    -   **Logging**: All actions taken by the agent will be logged to a dedicated `logs` table for monitoring and auditing.

### 5.2. Model Retraining & Drift Detection Agent (ML Worker)

-   **Implementation**: An external **ML Model Worker** service, which can be triggered by a **Supabase Database Hook**.
-   **Workflow**:
    1.  **Trigger**: A `AFTER INSERT` hook on the `race_results` table can call a webhook that initiates the ML Worker once a race's results are fully populated.
    2.  **Retraining**: The worker executes the model retraining pipeline as described in Section 4.
    3.  **Model Evaluation & Drift Detection**:
        -   After retraining, the new model is evaluated on a hold-out dataset (e.g., the last 3 races).
        -   Its performance (e.g., Brier Score, accuracy) is compared against the currently deployed model.
        -   If the new model shows a statistically significant improvement, it is promoted to "production" (i.e., its artifact is marked as the latest version in Supabase Storage).
        -   This prevents model performance from degrading over time (model drift) and ensures the platform's predictions continuously improve.
    4.  **Prediction Generation**: The newly validated model is used to generate and store predictions for the next race.

## 6. GenAI-Powered F1 Assistant Architecture

The F1 assistant is implemented as a secure and context-aware backend service, protecting API keys and centralizing logic.

-   **Implementation**: A dedicated **Supabase Edge Function** (`/functions/v1/ask-apex`).
-   **Architecture Flow**:
    1.  **User Input**: The user asks a question in the frontend chat interface.
    2.  **API Call**: The client sends the user's query to the `ask-apex` Edge Function.
    3.  **Context Assembly (Server-Side)**: The Edge Function receives the query and fetches relevant context directly from the Supabase DB. This includes race data, user profile information (e.g., favorite driver), and historical stats.
    4.  **Dynamic Prompt Engineering**: The function constructs a detailed, context-rich prompt for the LLM, similar to the v1 design but now executed securely on the backend.
    5.  **Secure LLM Call**: The Edge Function calls the third-party LLM API (e.g., OpenAI) using an API key stored securely as a Supabase secret.
    6.  **Response Handling**: The response is streamed back from the function to the client for display. This architecture prevents exposure of API keys and offloads complex logic from the client.
