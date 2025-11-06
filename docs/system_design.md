# System Design: F1 Fan Analytics & Prediction Platform

- **Version**: 2.0
- **Date**: 2025-11-06
- **Author**: Bob, Architect

---

## 1. Implementation Approach

This document outlines a full-stack architecture for the F1 Analytics and Prediction Platform, pivoting from a client-side proof-of-concept to a robust, scalable, and real-time application. The backend will be powered by **Supabase**, providing database, authentication, storage, and serverless capabilities.

This approach directly addresses the user's requirement for a "real" application with no mockups, supporting advanced features like live predictions, automated data updates, and machine learning model integration.

The implementation will be structured as follows:

1.  **Backend Setup**: Configure the Supabase project, including the PostgreSQL database schema, authentication rules (RLS), and storage buckets.
2.  **Data Ingestion**: Develop a standalone service to fetch historical and live data from the Ergast/FastF1 API and populate the Supabase database. This service will also feed the Supabase Realtime service during live events.
3.  **Backend Logic (Edge Functions)**:
    *   Create an Edge Function to act as a secure proxy for the conversational AI assistant, protecting API keys.
    *   Implement a scheduled Edge Function (`pg_cron`) to act as the "Agent," periodically checking for and applying updates to drivers, teams, and schedules.
4.  **Machine Learning Worker**: Develop an external service (e.g., running on a schedule in a separate compute environment) responsible for:
    *   Fetching training data from the Supabase DB.
    *   Retraining predictive models after each race.
    *   Storing updated model artifacts in Supabase Storage and writing new predictions to the `predictions` table.
5.  **Frontend Integration**: The Next.js frontend will communicate with Supabase using the `supabase-js` client library to handle user authentication, data queries, and real-time subscriptions.

## 2. Architecture

The system is decoupled into a frontend, a Supabase backend, external compute services for specialized tasks (data ingestion, ML), and third-party APIs.

```plantuml
@startuml
!theme plain
title System Architecture (Supabase Full-Stack)

package "User-Facing" {
    package "Frontend Application (Next.js)" {
        [UI Components (Shadcn/React)] as UI
        [State Management (Zustand)] as State
    }
}

package "Cloud Backend" {
    package "Supabase" {
        [Supabase Auth] as Auth
        [PostgreSQL Database] as DB
        [Realtime] as Realtime
        [Edge Functions] as Functions
        [Storage] as SupabaseStorage
    }

    package "External Compute" {
        [Data Ingestion Service] as Ingestion
        [ML Model Worker (Retraining & Prediction)] as MLWorker
    }
}

package "Third-Party APIs" {
    [F1 Data API (e.g., Ergast, FastF1)] as F1API
    [Generative AI API (e.g., OpenAI)] as GenAIAPI
}

' Frontend to Supabase
UI -> Auth: User login/signup
UI -> DB: Queries for data (via Supabase client)
UI -> Functions: Invokes serverless functions (e.g., AI chat)
State <--> Realtime: Subscribes to live race data channels

' Supabase Internal & Edge Functions
Functions -> DB: CRUD operations
Functions -> GenAIAPI: Proxies requests to AI assistant
Functions -> DB: Updates tables (e.g., driver/team changes via cron job)
Auth -> DB: Manages users table

' Backend Services to Supabase
Ingestion -> F1API: Fetches live/historical data
Ingestion -> Realtime: Broadcasts live data
Ingestion -> DB: Stores historical data

MLWorker -> DB: Reads data for training
MLWorker -> DB: Writes new predictions
MLWorker -> SupabaseStorage: Stores trained model artifacts

@enduml
```

## 3. Database ER Diagram

The database schema is designed to be relational and normalized, providing a single source of truth for all application data.

```plantuml
@startuml
!theme plain
title Database ER Diagram (Supabase PostgreSQL)

' Entities
entity "users" as users {
  * id: uuid <<PK>> (from auth.users)
  --
  email: varchar
  ...
}

entity "profiles" as profiles {
  * user_id: uuid <<PK, FK>>
  --
  username: varchar
  favorite_driver_id: uuid <<FK>>
  favorite_constructor_id: uuid <<FK>>
}

entity "drivers" as drivers {
  * id: uuid <<PK>>
  --
  name: varchar
  nationality: varchar
  permanent_number: integer
  code: varchar(3)
  dob: date
  headshot_url: varchar
}

entity "constructors" as constructors {
  * id: uuid <<PK>>
  --
  name: varchar
  nationality: varchar
  logo_url: varchar
}

entity "seasons" as seasons {
  * year: integer <<PK>>
  --
  url: varchar
}

entity "circuits" as circuits {
  * id: uuid <<PK>>
  --
  name: varchar
  location: varchar
  country: varchar
}

entity "races" as races {
  * id: uuid <<PK>>
  --
  season_year: integer <<FK>>
  round: integer
  name: varchar
  date: date
  time: time
  circuit_id: uuid <<FK>>
}

entity "race_results" as race_results {
  * race_id: uuid <<PK, FK>>
  * driver_id: uuid <<PK, FK>>
  --
  constructor_id: uuid <<FK>>
  position: integer
  points: float
  status: varchar
  fastest_lap_time: time
  fastest_lap_rank: integer
}

entity "predictions" as predictions {
    * id: uuid <<PK>>
    --
    race_id: uuid <<FK>>
    driver_id: uuid <<FK>>
    prediction_type: varchar (e.g., 'QUALIFYING', 'RACE_WINNER')
    predicted_position: integer
    confidence: float
    model_version: varchar
    created_at: timestamp
}

' Relationships
users ||..o| profiles : "one-to-one"
profiles }o..|| drivers
profiles }o..|| constructors

seasons ||--o{ races
circuits ||--o{ races

races ||--o{ race_results
drivers ||--o{ race_results
constructors ||--o{ race_results

races ||--o{ predictions
drivers ||--o{ predictions

@enduml
```

## 4. Program Call Flow (Live Race Experience)

This sequence diagram illustrates how real-time data flows through the system to the user during a live race.

```plantuml
@startuml
actor User
participant "Frontend (UI)" as UI
participant "Supabase Realtime" as Realtime
participant "Data Ingestion Service" as Ingestion
participant "F1 Data API" as F1API

autonumber

User -> UI: Navigates to Live Race page
UI -> Realtime: Subscribe to channel "race:{raceId}"
Realtime --> UI: Acknowledges subscription

loop Periodically (e.g., every 5s)
    Ingestion -> F1API: GET /live_timing
    F1API --> Ingestion: Returns latest lap data
    Ingestion -> Realtime: Broadcast on "race:{raceId}"
    note right
        Payload: {
            type: "LAP_UPDATE",
            driver: "VER",
            lap_time: "1:32.543",
            position: 1
        }
    end note
end

Realtime -> UI: Pushes new event to client
UI -> UI: Updates leaderboard and track map
User -> UI: Sees updated race state

@enduml
```

## 5. Anything UNCLEAR / Assumptions

-   **Live Data Source**: The quality and latency of the "Live Race Experience" are entirely dependent on the third-party F1 Data API. We assume a reliable API (like FastF1's wrapper) is available, but its real-world performance will need to be validated.
-   **ML Worker Infrastructure**: The design of the ML Worker is high-level. The specific infrastructure (e.g., AWS SageMaker, Google AI Platform, a custom Docker container) and associated costs need to be defined in a separate, more detailed Data Science design document.
-   **Cost Management**: This architecture introduces new costs: Supabase plan, compute for ingestion/ML services, and GenAI API calls. These will need to be monitored and optimized.
-   **Logging and Monitoring**: A proper logging and monitoring solution (e.g., Supabase's built-in logs, or integrating with a service like Sentry or Datadog) must be implemented for all backend services to ensure reliability, especially for the "Agent" and ML worker.