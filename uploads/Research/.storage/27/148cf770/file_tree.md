# F1 Real-Time Analytics Engine - File Structure

```
f1-analytics-engine/
├── frontend/                           # React + TypeScript web application
│   ├── public/
│   │   ├── index.html
│   │   └── assets/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Dashboard/
│   │   │   │   ├── LiveLeaderboard.tsx
│   │   │   │   ├── TrackMap.tsx
│   │   │   │   ├── PredictionPanel.tsx
│   │   │   │   └── TimingTower.tsx
│   │   │   ├── Strategy/
│   │   │   │   ├── AlternativeSimulator.tsx
│   │   │   │   ├── PitStopOptimizer.tsx
│   │   │   │   └── StrategyComparison.tsx
│   │   │   ├── Telemetry/
│   │   │   │   ├── SpeedTrace.tsx
│   │   │   │   ├── TelemetryChart.tsx
│   │   │   │   └── TireHeatmap.tsx
│   │   │   ├── Battle/
│   │   │   │   ├── BattleForecast.tsx
│   │   │   │   └── OvertakingPrediction.tsx
│   │   │   └── Shared/
│   │   │       ├── Button.tsx
│   │   │       ├── Card.tsx
│   │   │       └── Chart.tsx
│   │   ├── pages/
│   │   │   ├── Home.tsx
│   │   │   ├── LiveRace.tsx
│   │   │   ├── StrategyCenter.tsx
│   │   │   ├── DriverProfile.tsx
│   │   │   └── Historical.tsx
│   │   ├── hooks/
│   │   │   ├── useWebSocket.ts
│   │   │   ├── usePredictions.ts
│   │   │   └── useRaceData.ts
│   │   ├── store/
│   │   │   ├── raceStore.ts
│   │   │   ├── userStore.ts
│   │   │   └── predictionStore.ts
│   │   ├── services/
│   │   │   ├── api.ts
│   │   │   ├── websocket.ts
│   │   │   └── auth.ts
│   │   ├── utils/
│   │   │   ├── formatters.ts
│   │   │   └── constants.ts
│   │   ├── App.tsx
│   │   └── main.tsx
│   ├── package.json
│   ├── tsconfig.json
│   ├── vite.config.ts
│   └── tailwind.config.js
│
├── backend/                            # FastAPI Python backend
│   ├── app/
│   │   ├── api/
│   │   │   ├── v1/
│   │   │   │   ├── endpoints/
│   │   │   │   │   ├── races.py
│   │   │   │   │   ├── predictions.py
│   │   │   │   │   ├── strategy.py
│   │   │   │   │   ├── telemetry.py
│   │   │   │   │   ├── battles.py
│   │   │   │   │   └── users.py
│   │   │   │   └── api.py
│   │   │   └── deps.py
│   │   ├── core/
│   │   │   ├── config.py
│   │   │   ├── security.py
│   │   │   └── logging.py
│   │   ├── db/
│   │   │   ├── base.py
│   │   │   ├── session.py
│   │   │   └── init_db.py
│   │   ├── models/
│   │   │   ├── driver.py
│   │   │   ├── team.py
│   │   │   ├── race.py
│   │   │   ├── telemetry.py
│   │   │   ├── prediction.py
│   │   │   └── user.py
│   │   ├── schemas/
│   │   │   ├── driver.py
│   │   │   ├── race.py
│   │   │   ├── prediction.py
│   │   │   └── strategy.py
│   │   ├── services/
│   │   │   ├── race_service.py
│   │   │   ├── telemetry_service.py
│   │   │   ├── prediction_service.py
│   │   │   ├── strategy_service.py
│   │   │   ├── battle_service.py
│   │   │   └── notification_service.py
│   │   ├── ml/
│   │   │   ├── models/
│   │   │   │   ├── lap_time_predictor.py
│   │   │   │   ├── position_predictor.py
│   │   │   │   └── overtaking_predictor.py
│   │   │   ├── explainability/
│   │   │   │   ├── shap_explainer.py
│   │   │   │   └── lime_explainer.py
│   │   │   ├── feature_engineering.py
│   │   │   └── model_loader.py
│   │   ├── workers/
│   │   │   ├── celery_app.py
│   │   │   └── tasks.py
│   │   ├── websocket/
│   │   │   ├── connection_manager.py
│   │   │   └── handlers.py
│   │   └── main.py
│   ├── alembic/
│   │   ├── versions/
│   │   └── env.py
│   ├── tests/
│   │   ├── api/
│   │   ├── services/
│   │   └── ml/
│   ├── requirements.txt
│   ├── pyproject.toml
│   └── Dockerfile
│
├── ml-pipeline/                        # Machine Learning pipeline
│   ├── training/
│   │   ├── lap_time_model.py
│   │   ├── position_model.py
│   │   ├── overtaking_model.py
│   │   └── train_pipeline.py
│   ├── inference/
│   │   ├── model_server.py
│   │   └── batch_inference.py
│   ├── feature_store/
│   │   ├── feast_config.py
│   │   └── feature_definitions.py
│   ├── notebooks/
│   │   ├── exploratory_analysis.ipynb
│   │   └── model_evaluation.ipynb
│   └── requirements.txt
│
├── stream-processing/                  # Kafka Streams / Flink
│   ├── src/
│   │   ├── processors/
│   │   │   ├── telemetry_processor.py
│   │   │   ├── prediction_processor.py
│   │   │   └── alert_processor.py
│   │   └── main.py
│   └── requirements.txt
│
├── data-pipeline/                      # Airflow DAGs
│   ├── dags/
│   │   ├── historical_data_sync.py
│   │   ├── model_training.py
│   │   └── feature_store_update.py
│   ├── plugins/
│   └── requirements.txt
│
├── mobile/                             # React Native mobile app
│   ├── src/
│   │   ├── screens/
│   │   ├── components/
│   │   ├── navigation/
│   │   └── services/
│   ├── package.json
│   └── app.json
│
├── infrastructure/                     # Infrastructure as Code
│   ├── terraform/
│   │   ├── main.tf
│   │   ├── variables.tf
│   │   ├── outputs.tf
│   │   ├── modules/
│   │   │   ├── eks/
│   │   │   ├── rds/
│   │   │   ├── kafka/
│   │   │   └── redis/
│   │   └── environments/
│   │       ├── dev/
│   │       ├── staging/
│   │       └── prod/
│   ├── kubernetes/
│   │   ├── base/
│   │   │   ├── deployment.yaml
│   │   │   ├── service.yaml
│   │   │   └── ingress.yaml
│   │   ├── overlays/
│   │   │   ├── dev/
│   │   │   ├── staging/
│   │   │   └── prod/
│   │   └── helm/
│   │       └── f1-analytics/
│   │           ├── Chart.yaml
│   │           ├── values.yaml
│   │           └── templates/
│   └── docker-compose.yml
│
├── monitoring/                         # Monitoring configurations
│   ├── prometheus/
│   │   ├── prometheus.yml
│   │   └── alerts.yml
│   ├── grafana/
│   │   ├── dashboards/
│   │   └── datasources.yml
│   └── elk/
│       ├── logstash.conf
│       └── elasticsearch.yml
│
├── scripts/                            # Utility scripts
│   ├── setup_dev.sh
│   ├── deploy.sh
│   ├── backup_db.sh
│   └── seed_data.py
│
├── docs/                               # Documentation
│   ├── design/
│   │   ├── f1_system_design.md
│   │   ├── architect.plantuml
│   │   ├── class_diagram.plantuml
│   │   ├── sequence_diagram.plantuml
│   │   └── er_diagram.plantuml
│   ├── api/
│   │   └── openapi.yaml
│   ├── deployment/
│   │   └── deployment_guide.md
│   └── user_guide/
│       └── user_manual.md
│
├── .github/
│   └── workflows/
│       ├── ci.yml
│       ├── cd.yml
│       └── test.yml
│
├── .gitignore
├── README.md
├── LICENSE
└── docker-compose.yml
```

## Key Directory Explanations

### Frontend (`/frontend`)
- React 18+ with TypeScript
- Vite for build tooling
- Tailwind CSS + shadcn/ui for styling
- Zustand for state management
- Socket.IO client for real-time updates

### Backend (`/backend`)
- FastAPI for REST API
- SQLAlchemy for ORM
- Pydantic for data validation
- Celery for background tasks
- WebSocket support for real-time features

### ML Pipeline (`/ml-pipeline`)
- PyTorch/TensorFlow models
- Feast for feature store
- MLflow for experiment tracking
- Jupyter notebooks for analysis

### Stream Processing (`/stream-processing`)
- Kafka consumers/producers
- Apache Flink for stream processing
- Real-time data transformation

### Data Pipeline (`/data-pipeline`)
- Apache Airflow DAGs
- ETL jobs for historical data
- Model training orchestration

### Infrastructure (`/infrastructure`)
- Terraform for AWS resources
- Kubernetes manifests
- Helm charts for deployment
- Docker Compose for local development

### Monitoring (`/monitoring`)
- Prometheus for metrics
- Grafana dashboards
- ELK Stack for logging
- Jaeger for tracing

## Technology Stack Summary

**Frontend:**
- React 18+, TypeScript, Tailwind CSS, Vite, Socket.IO

**Backend:**
- FastAPI, Python 3.11+, SQLAlchemy, Celery, Redis

**Data:**
- PostgreSQL, TimescaleDB, MongoDB, Redis, S3

**ML/AI:**
- PyTorch, scikit-learn, XGBoost, SHAP, LIME, Feast, MLflow

**Streaming:**
- Apache Kafka, Apache Flink

**Infrastructure:**
- AWS (EKS, RDS, MSK, S3), Docker, Kubernetes, Terraform

**Monitoring:**
- Prometheus, Grafana, ELK Stack, Jaeger, DataDog