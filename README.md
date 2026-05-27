# Google Cloud ML — Vertex AI, TensorFlow, and Keras

![Type](https://img.shields.io/badge/Type-Cloud_ML_%7C_Distributed_Training_%7C_MLOps-4285F4?style=flat-square)
![Stack](https://img.shields.io/badge/Stack-Vertex_AI_%7C_TensorFlow_%7C_Keras_%7C_TPU_%7C_Gemini-4285F4?style=flat-square)
![Platform](https://img.shields.io/badge/Platform-Google_Cloud_Platform-4285F4?style=flat-square)

---

## Overview

A collection of Google Cloud Platform (GCP) labs and notebooks covering the full spectrum of cloud-native ML engineering: from **Keras model building fundamentals** to **distributed TPU training**, **data validation pipelines**, and **Vertex AI model deployment**. Also includes the **Google AI Python SDK** for Gemini API integration.

These labs represent the production engineering layer of the ML stack — moving from local notebook experiments to scalable, reproducible, cloud-deployed models.

---

## Notebooks

| Notebook | Topic | Key Concepts |
|---|---|---|
| `keras.ipynb` | Keras fundamentals | Sequential/Functional API, layers, callbacks |
| `distributed_training.ipynb` | Multi-GPU/TPU strategy | `tf.distribute.MirroredStrategy`, data sharding |
| `distributed_training_with_TF.ipynb` | TF distributed training | `tf.distribute`, gradient aggregation, batch scaling |
| `tfdv_advanced_taxi.ipynb` | TF Data Validation | Schema inference, drift detection, anomaly alerts |
| `tpu_speed_data_pipelines.ipynb` | TPU + tf.data optimization | Pipeline bottleneck elimination, prefetch, cache |
| `train_deploy.ipynb` | Vertex AI deployment | Custom training job, model registry, endpoint serving |
| `lab_exercise.ipynb` | Integrated lab | End-to-end cloud ML workflow |

---

## Architecture Overview

```
Local Development
    └── keras.ipynb ← Model architecture & training fundamentals

Data Quality Gate
    └── tfdv_advanced_taxi.ipynb ← Schema validation, drift detection, anomaly flagging

Distributed Training (Scale-out)
    ├── distributed_training.ipynb          ← MirroredStrategy (multi-GPU)
    ├── distributed_training_with_TF.ipynb  ← TF distributed training
    └── tpu_speed_data_pipelines.ipynb      ← TPU data pipeline optimization

Production Deployment (Vertex AI)
    └── train_deploy.ipynb
          ├── Submit custom training job
          ├── Register model in Vertex AI Model Registry
          └── Deploy to Vertex AI Endpoint
```

---

## Key Concepts

### Distributed Training
```python
# MirroredStrategy: synchronous multi-GPU training
strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    model = create_model()  # Model created within strategy scope
    model.compile(...)

# Data pipeline must be sharded across devices
train_dataset = train_dataset.batch(batch_size * strategy.num_replicas_in_sync)
```

Gradient aggregation across devices uses `AllReduce` — each GPU computes gradients on its shard, then gradients are averaged before weight updates.

### TF Data Validation (TFDV)
```python
# Infer schema from training data
train_stats = tfdv.generate_statistics_from_csv(train_data_path)
schema = tfdv.infer_schema(statistics=train_stats)

# Validate serving data against training schema
serving_stats = tfdv.generate_statistics_from_csv(serving_data_path)
anomalies = tfdv.validate_statistics(serving_stats, schema)
tfdv.display_anomalies(anomalies)
```

**Why this matters:** Schema drift — where incoming inference data diverges from training data distributions — causes silent model degradation. TFDV makes this observable before it damages predictions.

### Vertex AI Deployment
```bash
# Service account setup
gcloud iam service-accounts create vertex-custom-training-sa \
    --description="Vertex AI Custom Training SA"

# IAM binding: Storage, BigQuery, AI Platform
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member=serviceAccount:$SERVICE_ACCOUNT_ID@$PROJECT_ID.iam.gserviceaccount.com \
    --role="roles/aiplatform.user"
```

### Google AI SDK — Gemini API
```python
import google.generativeai as genai
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

model = genai.GenerativeModel('gemini-1.5-flash')
response = model.generate_content("prompt here")
```

---

## Files

```
Google-VertexAI-TF-Keras/
├── keras.ipynb                             # Keras model building fundamentals
├── distributed_training.ipynb              # Multi-GPU MirroredStrategy
├── distributed_training_with_TF.ipynb      # TF native distributed training
├── tfdv_advanced_taxi.ipynb                # TF Data Validation on taxi dataset
├── tpu_speed_data_pipelines.ipynb          # TPU-optimized tf.data pipelines (3.3GB)
├── train_deploy.ipynb                      # Vertex AI train + deploy workflow
├── lab_exercise.ipynb                      # Integrated lab exercise
│
├── Google Cloud Platform.drawio            # GCP architecture diagram
├── Google AI Python SDK for the Gemini API # Gemini SDK quickstart
└── base case dependencies                  # IAM setup + gcloud config commands
```

---

## GCP Architecture Diagram

The `Google Cloud Platform.drawio` file contains a visual architecture diagram — open with [draw.io](https://app.diagrams.net/) or the VS Code draw.io extension.

---

## Connection to Computational Biology

Cloud-scale ML is increasingly essential for genomic data at scale:

| Vertex AI / GCP concept | Genomics / bioinformatics equivalent |
|---|---|
| Distributed training across GPUs/TPUs | Large-scale GWAS or AlphaFold2 training |
| tf.data pipeline optimization | Streaming variant/BAM data for deep learning |
| TFDV schema validation | VCF file format validation, QC pipeline monitoring |
| Vertex AI model registry | Model versioning in production bioinformatics pipelines |
| Model endpoint serving | Real-time variant pathogenicity scoring APIs |
| TPU-accelerated training | Protein structure prediction at scale |

Google Cloud BigQuery and Vertex AI are increasingly used as the compute backbone for:
- **UK Biobank / All of Us** genomic analysis
- **Terra / FireCloud** bioinformatics workflows
- **Google Health / DeepMind** clinical AI deployment
