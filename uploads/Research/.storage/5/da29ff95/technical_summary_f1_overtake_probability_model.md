# Technical Summary: Real-Time F1 Overtake Probability Model

This report outlines the methodologies, data inputs, and conceptual framework required to build a real-time overtake probability model, based on the F1 Insights powered by Amazon Web Services (AWS).

## 1. Algorithms and Statistical Models

The F1 Insights system leverages machine learning on the AWS platform to generate its predictions [ref: 0-0]. While the exact algorithm for the "Overtake Probability" insight is not publicly detailed, information from related F1-AWS projects provides strong indicators of the technologies used.

*   **Gradient Boosting Machines:** For the "Pit Strategy Battle" insight, which shares a similar real-time predictive nature, AWS used its built-in Amazon SageMaker **XGBoost** algorithm, an efficient open-source implementation of the gradient boosted trees algorithm [ref: 0-1].
*   **Deep Learning:** F1 and AWS have trained deep learning models on 65 years of historical race data to generate various performance statistics [ref: 0-3].
*   **Neural Networks:** For certain insights like "Close to the Wall," F1 uses a fusion of deep neural networks and computer vision algorithms [ref: 0-4].

The overall approach involves training models on historical data and then deploying them for real-time inference [ref: 0-1].

## 2. Critical Input Features

The model's accuracy depends on a rich set of live and historical data inputs. F1 and AWS combine over 70 years of historical race data with millions of real-time data points [ref: 0-4]. The system has the unique advantage of knowing the data from both cars involved in a battle, which individual teams cannot access for direct comparison [ref: 0-0].

| Feature Category | Data Point | Description | Source |
| --- | --- | --- | --- |
| **Car-Specific Data** | Telemetry | Each car has 300 sensors generating 1.1 million data points per second. | [ref: 0-1] |
| | Tire Performance | Tire condition, age, compound, history, and degradation are critical factors. | [ref: 0-0] |
| | Tire Wear Energy | Estimated from car speed, longitudinal/lateral accelerations, and gyro data to model the energy transfer of the tire on the road surface. | [ref: 0-4] |
| | Speed Differential | The difference in speed between the attacking and defending car. | [ref: 0-0] |
| | Car Performance | Metrics include cornering performance, straight-line performance, and car balance/handling. | [ref: 0-4] |
| | Vehicle Dynamics | Data related to aerodynamics, power unit, and overall vehicle optimization. | [ref: 0-4] |
| **Driver-Specific Data** | Driver Skill | The model likely establishes a baseline for a "generic" driver, allowing for analysis of which drivers consistently outperform the predicted probability. | [ref: 0-0] |
| | Performance Metrics | Analysis of driver performance in key areas like acceleration, braking, and cornering. | [ref: 0-4] |
| | Overtaking Score | As part of the "Driver Season Performance" insight, drivers are scored on their overtaking skill. | [ref: 0-4] |
| **Contextual Data** | Live Car Gaps | Real-time timing, spatial gaps, and relative velocities of the cars. | [ref: 0-0] |
| | Track Position | The specific location on the track, including corner type and distance to the car ahead. | [ref: 0-0] |
| | DRS Availability | The availability and use of the Drag Reduction System is a key contextual factor. | [ref: 0-0] |
| | Race State | The overall race situation, including factors like safety cars and yellow flags. | [ref: 0-4] |
| | Track History | Historical data about the specific track is used in predictions. | [ref: 0-4] |
| | Projected Pace | The predicted pace of a driver is used for forecasting battles. | [ref: 0-4] |

## 3. Function of the Official AWS "Overtake Probability" Insight

The "Overtake Probability" graphic was introduced for the 2019 F1 season as part of the F1 Insights powered by AWS [ref: 0-0].

*   **Purpose:** It is designed to show the probability of an overtake maneuver when two drivers are fighting for position, providing viewers with more context [ref: 0-0].
*   **Real-Time Updates:** The on-screen probability figures update in real-time, synced with the video feed, allowing viewers to see the probability change as one car closes on another [ref: 0-0].
*   **Underlying Data:** The prediction is made using machine learning that processes both live and historical data from the two cars involved in the battle [ref: 0-0].
*   **Analytical Analogy:** The insight is conceptually similar to "Expected Goals (xG)" in football. It provides a statistical baseline of what a generic driver might achieve in a given situation, which helps in judging how a specific driver's skill may have contributed to the outcome [ref: 0-0].
*   **Related Insight ("Battle Forecast"):** A complementary insight, "Battle Forecast," uses track history and projected driver pace to predict how many laps it will take for a chasing car to get within "striking distance" of the car ahead [ref: 0-4].

## 4. Conceptual Implementation Framework

A case study on the related "Pit Strategy Battle" insight provides a detailed blueprint for a real-time prediction system [ref: 0-1]. This serverless architecture is designed for high-throughput, low-latency predictions, with the entire process from data capture to broadcast completing in under 500 milliseconds [ref: 0-1].

**Key Architectural Components:**
1.  **Data Ingestion:** Signals are captured at the track, passed through F1 infrastructure, and sent via an HTTP call to the AWS Cloud [ref: 0-1]. Amazon Kinesis is a service used for ingesting the high volume of stream data [ref: 0-3].
2.  **API & Application Logic:** Amazon API Gateway acts as the entry point, invoking an AWS Lambda function that contains the core application logic [ref: 0-1].
3.  **State Management:** The Lambda function updates the race state (e.g., driver positions, timing data, ML features) stored in Amazon DynamoDB, which provides single-digit millisecond performance [ref: 0-1].
4.  **Prediction & Inference:** When a trigger condition is met, the Lambda function uses a pre-trained model to make a prediction [ref: 0-1]. To minimize latency, the model is loaded directly into the Lambda function's memory rather than being called from a separate SageMaker endpoint [ref: 0-1].
5.  **Model Training:** Models are trained in Amazon SageMaker using historical data stored in Amazon S3 [ref: 0-1]. Exploratory analysis is done in SageMaker notebooks [ref: 0-1].
6.  **Deployment:** The system uses CI/CD tools like AWS CodePipeline and AWS CodeBuild, with infrastructure provisioned as code using AWS CloudFormation for predictable deployments [ref: 0-1].

## 5. Published Research and Case Studies

The search results did not yield formal academic research papers or public open-source projects on this specific topic. The primary sources of information are case studies and technical blogs published by AWS describing its partnership with Formula 1.

*   **AWS Blog:** The post "Accelerating innovation: How serverless machine learning on AWS powers F1 Insights" is the most detailed technical case study, providing a transferable architecture and methodology, despite its focus on pit strategy [ref: 0-1].
*   **AWS F1 Page:** The "F1 on AWS" and "F1 Insights" pages offer high-level descriptions of the various predictive graphics, including "Battle Forecast" and "Overtaking" as a driver skill metric [ref: 0-4].