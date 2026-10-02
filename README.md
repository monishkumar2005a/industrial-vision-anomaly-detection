

Pasted markdown(4).md
File

Yes. Since we're preparing this for GitHub, let's make the README.md industry/project-portfolio quality before we push.

I suggest the README covers:

Project title + one-line description

What the system does

Key features

Architecture / pipeline

Tech stack

MVTec AD dataset

PatchCore + ViT approach

Benchmark results — including your ~0.938 image AUROC and ~0.960 pixel AUROC

Project structure

Installation

Training

Benchmarking

Prediction / inspection reports

Gemini AI reporting

Testing

Git LFS / model files

Example outputs

Future improvements

License

One important point
Your current repository layout is:

industrial-anomaly-detection (3)/
├── industrial-anomaly-detection/
├── models/
├── outputs/
└── .idea/
So the README should be placed at the Git root:

industrial-anomaly-detection (3)/
└── README.md
rather than inside the nested industrial-anomaly-detection folder.

If you want, I can write the complete final README.md now, specifically using the actual features/results of your project rather than a generic README.


give full comtent i can paste it

Absolutely. Below is a complete GitHub-ready README tailored to your actual industrial anomaly detection project, including the MVTec benchmark, PatchCore pipeline, ViT features, inspection reports, Gemini AI reporting, testing, and Git LFS.

Paste this into:

C:\Users\Monis\Downloads\industrial-anomaly-detection (3)\README.md
# Industrial Anomaly Detection

> Vision Transformer + PatchCore based industrial defect detection and localization system with automated inspection reports and optional Gemini AI analysis.

---

## Overview

This project implements an end-to-end **industrial visual anomaly detection system** designed to detect and localize defects in manufactured products.

The system uses **deep visual features**, a **PatchCore-style memory bank**, nearest-neighbor anomaly scoring, and image-level/pixel-level evaluation to identify abnormal regions without requiring manually labeled defect masks during training.

The project is evaluated using the **MVTec Anomaly Detection (MVTec AD)** dataset and provides:

- Training of category-specific anomaly detection models
- Image-level anomaly detection
- Pixel-level anomaly localization
- Anomaly heatmaps
- Defect bounding-box visualization
- MVTec benchmark evaluation
- Automated HTML inspection reports
- CSV/JSON prediction logs
- Optional Gemini AI generated inspection notes
- Automated test suite
- Git LFS support for trained model artifacts

---

## Key Features

### 1. Unsupervised Anomaly Detection

The model is trained primarily using **normal images** and learns the distribution of normal visual features.

Defects can then be detected as regions that differ significantly from the learned normal feature distribution.

### 2. Patch-Level Feature Extraction

Input images are resized and processed using a deep vision feature extractor.

The image is represented using multiple local patch-level feature vectors rather than treating the entire image as a single feature.

This makes the system suitable for detecting localized manufacturing defects.

### 3. PatchCore-Style Memory Bank

Normal patch features are stored in a memory bank.

To reduce memory requirements, coreset sampling is used to retain a representative subset of normal features.

During inference, test patches are compared against the memory bank using nearest-neighbor search.

### 4. Image-Level Anomaly Detection

The system produces an anomaly score for the complete image.

Conceptually:

```text
Normal image
     ↓
Feature extraction
     ↓
Patch features
     ↓
Nearest-neighbor comparison
     ↓
Patch anomaly scores
     ↓
Image anomaly score
     ↓
Normal / Anomalous
5. Pixel-Level Localization
Patch-level anomaly scores are converted into an anomaly map.

The system can identify where an abnormal region is located and generate a visual overlay.

6. Defect Localization
Detected anomaly regions can be converted into bounding boxes.

Example:

Input Image
     ↓
Anomaly Map
     ↓
Thresholding
     ↓
Connected Components
     ↓
Defect Region
     ↓
Bounding Box
7. Automated Inspection Reports
For each prediction, the system can generate:

Original image

Anomaly visualization

Prediction result

Anomaly score

Threshold

Hotspot information

Defect localization

Optional AI inspection note

The inspection results are organized into HTML reports.

8. Gemini AI Reporting
The system optionally integrates with Google's Gemini API to generate human-readable inspection notes for anomalous samples.

Gemini is used as a reporting/interpretation layer and does not replace the anomaly detector.

The core detection pipeline remains independent of the AI reporting component.

The project supports:

gemini-2.5-flash-lite
An API key is supplied through an environment variable and is never intended to be committed to Git.

System Architecture
                    MVTec AD Dataset
                           │
                           ▼
                  ┌──────────────────┐
                  │ Image Discovery  │
                  └────────┬─────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │ Image Preprocess │
                  │   224 × 224      │
                  └────────┬─────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ Deep Feature Extractor│
                │      ViT / CNN       │
                └──────────┬───────────┘
                           │
                           ▼
                 ┌────────────────────┐
                 │ Patch-Level Feature│
                 │     Extraction     │
                 └─────────┬──────────┘
                           │
              ┌────────────┴────────────┐
              │                         │
              ▼                         ▼
      Normal Training Images       Test Images
              │                         │
              ▼                         ▼
      ┌─────────────────┐       ┌─────────────────┐
      │ Memory Bank     │       │ Patch Features  │
      │ Construction    │       │                 │
      └────────┬────────┘       └────────┬────────┘
               │                         │
               ▼                         ▼
      ┌─────────────────┐       ┌─────────────────┐
      │ Coreset Sampling│       │ Nearest Neighbor│
      └────────┬────────┘       └────────┬────────┘
               │                         │
               └────────────┬────────────┘
                            ▼
                    ┌────────────────┐
                    │ Anomaly Scores │
                    └───────┬────────┘
                            │
              ┌─────────────┴─────────────┐
              │                           │
              ▼                           ▼
       Image-Level Score          Pixel-Level Map
              │                           │
              ▼                           ▼
       Normal / Anomaly          Heatmap / Bounding Box
              │                           │
              └─────────────┬─────────────┘
                            ▼
                   Inspection Report
                            │
                            ▼
                     Optional Gemini
                       AI Analysis
Methodology
Step 1 — Image Preprocessing
Images are resized to:

224 × 224
and passed through the feature extraction pipeline.

Step 2 — Deep Feature Extraction
The system extracts high-dimensional visual representations from the input image.

The project is designed around patch-level representations so that both global image abnormality and localized defects can be detected.

Step 3 — Normal Feature Memory Bank
Normal training samples are processed to obtain patch-level feature vectors.

These features form the normality reference:

Normal Training Images
        ↓
Feature Extraction
        ↓
Patch Features
        ↓
Memory Bank
Step 4 — Coreset Sampling
The complete memory bank can become very large.

Coreset sampling is therefore used to retain a representative subset of the normal feature distribution.

This reduces memory and nearest-neighbor search requirements while retaining representative normal patterns.

Step 5 — Nearest-Neighbor Anomaly Scoring
For a test image:

Test Patch
    ↓
Feature Vector
    ↓
Nearest Normal Feature
    ↓
Distance
    ↓
Patch Anomaly Score
Higher distance indicates greater deviation from the learned normal distribution.

Step 6 — Image-Level Decision
Patch anomaly scores are aggregated to obtain an image-level anomaly score.

The score is compared against a threshold to classify the sample as:

NORMAL
or

ANOMALOUS
Step 7 — Anomaly Localization
Patch-level scores are mapped back to the image to create an anomaly heatmap.

The system can then identify concentrated abnormal regions and generate bounding boxes around detected areas.

Dataset
This project uses the MVTec Anomaly Detection (MVTec AD) dataset.

The benchmark contains multiple industrial object and texture categories with normal and anomalous samples.

The project currently supports these 15 categories:

bottle
cable
capsule
carpet
grid
hazelnut
leather
metal_nut
pill
screw
tile
toothbrush
transistor
wood
zipper
Expected dataset location:

D:\mvtec_anomaly_detection
The path can be changed using the command-line options provided by the project.

Benchmark Results
A full benchmark was performed across all 15 supported MVTec categories.

Aggregate Results
Metric	Mean Score
Image AUROC	0.9380
Pixel AUROC	0.9597
Interpretation
Image AUROC — 0.9380

Measures the ability of the system to distinguish anomalous images from normal images.

Pixel AUROC — 0.9597

Measures the ability of the system to distinguish anomalous pixels/regions from normal pixels.

The benchmark results are generated from the project's evaluation pipeline and should be interpreted as dataset-specific experimental results rather than a guarantee of performance on real production data.

Project Structure
industrial-anomaly-detection (3)/
│
├── .github/
│   └── workflows/
│
├── industrial-anomaly-detection/
│   │
│   ├── src/
│   │   └── industrial_anomaly/
│   │       ├── __init__.py
│   │       ├── api.py
│   │       ├── cli.py
│   │       ├── config.py
│   │       ├── data.py
│   │       ├── detector.py
│   │       ├── evaluation.py
│   │       ├── features.py
│   │       ├── inspection_report.py
│   │       ├── memory_bank.py
│   │       ├── report.py
│   │       ├── scoring.py
│   │       └── visualization.py
│   │
│   ├── tests/
│   │
│   ├── scripts/
│   │
│   ├── pyproject.toml
│   ├── README.md
│   └── ...
│
├── models/
│   ├── bottle/
│   ├── cable/
│   ├── capsule/
│   ├── carpet/
│   ├── grid/
│   ├── hazelnut/
│   ├── leather/
│   ├── metal_nut/
│   ├── pill/
│   ├── screw/
│   ├── tile/
│   ├── toothbrush/
│   ├── transistor/
│   ├── wood/
│   └── zipper/
│
├── outputs/
│   └── benchmark/
│
├── results/
│
├── reports/
│   └── inspection/
│
├── .gitignore
├── .gitattributes
├── Dockerfile
├── LICENSE
└── README.md
Installation
Requirements
Recommended environment:

Python 3.11+
Windows / Linux
Git
Git LFS
The project was developed and tested using Python 3.11.

Clone the Repository
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd "industrial-anomaly-detection (3)"
Create a Virtual Environment
Windows
python -m venv .venv
Activate:

.\.venv\Scripts\Activate.ps1
Linux
python3 -m venv .venv
source .venv/bin/activate
Install the Project
pip install --upgrade pip
pip install -e .
Dataset Setup
Download the MVTec AD dataset and extract it.

Example:

D:\mvtec_anomaly_detection
The resulting structure should resemble:

mvtec_anomaly_detection/
├── bottle/
├── cable/
├── capsule/
├── carpet/
├── grid/
├── hazelnut/
├── leather/
├── metal_nut/
├── pill/
├── screw/
├── tile/
├── toothbrush/
├── transistor/
├── wood/
└── zipper/
Command Line Interface
The project provides the anomaly-detect command.

Check the available commands:

anomaly-detect --help
Available workflows include:

train
evaluate
benchmark
report
predict
Training
To train a model for one MVTec category:

anomaly-detect train `
  --data-root "D:\mvtec_anomaly_detection" `
  --component "bottle" `
  --out-dir ".\models\bottle"
The exact available arguments can be viewed using:

anomaly-detect train --help
Evaluation
Evaluate a trained model:

anomaly-detect evaluate `
  --data-root "D:\mvtec_anomaly_detection" `
  --component "bottle" `
  --model-dir ".\models\bottle" `
  --out-dir ".\outputs\bottle"
Evaluation results include metrics and visualization artifacts.

Full MVTec Benchmark
Run the benchmark across all supported categories:

anomaly-detect benchmark `
  --data-root "D:\mvtec_anomaly_detection" `
  --out-dir ".\outputs\benchmark" `
  --save-models ".\models"
This produces benchmark artifacts such as:

outputs/
└── benchmark/
and category-specific model artifacts under:

models/
├── bottle/
├── cable/
├── capsule/
...
└── zipper/
Prediction
Run prediction on an image:

anomaly-detect predict `
  --model-dir ".\models\bottle" `
  --images "D:\mvtec_anomaly_detection\bottle\test\broken_large\000.png" `
  --out-dir ".\reports\inspection\bottle" `
  --component "bottle"
The output contains the prediction result and visualization.

Inspection Reports
The prediction pipeline can generate an HTML inspection report.

Example:

anomaly-detect predict `
  --model-dir ".\models\bottle" `
  --images "D:\mvtec_anomaly_detection\bottle\test" `
  --out-dir ".\reports\inspection\bottle" `
  --component "bottle"
The report can contain:

inspection_report.html
inspection_log.csv
predictions.json
along with generated visualization images.

Gemini AI Inspection Reporting
Gemini integration is optional.

The anomaly detector itself does not require Gemini.

To enable Gemini reporting, set the API key as an environment variable.

Windows PowerShell
$env:GEMINI_API_KEY = Read-Host "Enter Gemini API key"
Verify that the environment variable exists without printing the secret:

if ($env:GEMINI_API_KEY) {
    Write-Host "GEMINI_API_KEY is configured"
}
Run prediction with AI reporting:

anomaly-detect predict `
  --model-dir ".\models\bottle" `
  --images "D:\mvtec_anomaly_detection\bottle\test" `
  --out-dir ".\reports\inspection\bottle" `
  --report `
  --component "bottle"
The Gemini model used by the project can be configured through the CLI.

Current recommended model:

gemini-2.5-flash-lite
Important Security Rule
Never commit an API key to GitHub.

Do not place credentials directly inside Python source code.

Use:

GEMINI_API_KEY
as an environment variable.

Benchmark Reporting
Benchmark results can also be summarized through the reporting pipeline.

Example:

anomaly-detect report `
  --results ".\results\mvtec\results.csv"
Check the available options:

anomaly-detect report --help
Generated Artifacts
The project generates several types of artifacts.

Metrics
Examples include:

metrics.json
results.csv
Visualizations
Examples:

evaluation.png
heatmaps
bounding-box visualizations
Prediction Logs
inspection_log.csv
predictions.json
HTML Inspection Reports
inspection_report.html
These reports are intended to make the anomaly detection output easier to inspect and communicate.

Testing
The project includes an automated test suite.

Run:

pytest -q
The current test suite contains:

20 tests
and the project has been validated with:

20 passed
Tests cover core functionality such as:

Data handling

Feature processing

Scoring

Memory bank behavior

Detection pipeline

Reporting functionality

Project components

Production-Oriented Design
The project is structured to separate major responsibilities.

Data
 │
 ├── data.py
 │
 ▼
Feature Extraction
 │
 ├── features.py
 │
 ▼
Memory Bank
 │
 ├── memory_bank.py
 │
 ▼
Anomaly Detection
 │
 ├── detector.py
 │
 ├── scoring.py
 │
 ▼
Evaluation
 │
 ├── evaluation.py
 │
 ▼
Visualization
 │
 ├── visualization.py
 │
 ▼
Inspection Reporting
 │
 ├── inspection_report.py
 │
 ▼
AI Reporting
 │
 └── report.py
This modular structure makes it easier to:

Replace the feature extractor

Change anomaly scoring

Experiment with memory-bank strategies

Add new datasets

Add APIs

Add production monitoring

Run experiments independently

API Layer
The project also contains an API module:

src/industrial_anomaly/api.py
This provides a foundation for exposing anomaly detection functionality through a service layer.

A future deployment can connect:

Frontend
   ↓
REST API
   ↓
Anomaly Detection Engine
   ↓
Model
   ↓
Prediction
Example Industrial Workflow
A potential production workflow is:

Camera
  ↓
Product Image
  ↓
Image Validation
  ↓
Preprocessing
  ↓
Feature Extraction
  ↓
Anomaly Detection
  ↓
Anomaly Score
  ↓
Defect Localization
  ↓
Pass / Fail Decision
  ↓
Inspection Report
  ↓
Database / Dashboard
The current repository focuses primarily on the computer-vision anomaly detection and inspection-reporting components.

Why Patch-Level Detection?
Traditional image classification can answer:

Is this image defective?
but may not clearly answer:

Where is the defect?
Patch-level anomaly detection provides spatial information.

For example:

Product Image
      ↓
┌─────────────────────────┐
│                         │
│       NORMAL            │
│                         │
│              ████       │
│              ████       │
│              DEFECT      │
│                         │
└─────────────────────────┘
This makes the approach more suitable for industrial inspection scenarios where defect localization is important.

Advantages
The implemented approach provides several useful characteristics:

Does not require defect labels for the core training process

Uses normal samples as the reference distribution

Supports image-level anomaly detection

Supports pixel-level localization

Produces visual explanations

Supports automated inspection reports

Can process multiple MVTec categories

Provides quantitative evaluation metrics

Supports optional AI-generated inspection notes

Includes automated tests

Uses a modular Python architecture

Limitations
This project is an experimental/research-oriented industrial inspection system.

Important limitations include:

Dataset Dependency
Performance is measured using the MVTec AD dataset and may change significantly on different manufacturing environments.

Domain Shift
Changes in:

Camera

Lighting

Product orientation

Background

Manufacturing process

Surface material

Image resolution

can affect anomaly scores.

Threshold Calibration
Production systems require threshold calibration based on the actual acceptable defect rate and manufacturing requirements.

Real-Time Requirements
The current implementation is not presented as a certified real-time production inspection system.

Deployment would require benchmarking:

Inference latency

Throughput

CPU/GPU utilization

Memory usage

Camera acquisition time

End-to-end processing time

Future Improvements
Potential future upgrades include:

1. Production REST API
Expose inference through FastAPI:

POST /predict
POST /batch-predict
GET /health
GET /model-info
2. Web Dashboard
Create a React-based dashboard showing:

Live inspection results

Anomaly score

Defect location

Historical inspections

Category statistics

Failure rate

Confidence trends

3. Real-Time Camera Inspection
Integrate:

Industrial Camera
       ↓
Frame Capture
       ↓
Anomaly Detection
       ↓
Defect Localization
       ↓
Pass / Fail
4. Model Monitoring
Track:

Mean anomaly score

False-positive rate

False-negative rate

Drift

Model version

Data distribution changes

5. Model Versioning
Store model metadata such as:

Model Version
Dataset Version
Feature Extractor
Training Date
Threshold
Evaluation Metrics
Software Version
6. Containerized Deployment
The project includes a Dockerfile foundation for reproducible deployment.

A production architecture could use:

Docker
   ↓
FastAPI
   ↓
Inference Engine
   ↓
Model
7. GPU Acceleration
For higher-throughput industrial environments, GPU inference can be used to reduce processing latency.

8. Additional Feature Extractors
The architecture can be extended to experiment with:

Vision Transformers

CNN backbones

WideResNet

Hybrid feature extractors

Other anomaly detection architectures

Reproducibility
For reproducible experiments, record:

Python version
Package versions
Dataset version
Model configuration
Feature extractor
Memory bank size
Coreset ratio
Anomaly threshold
Hardware
Random seed
This is especially important when comparing anomaly detection experiments.

Model Artifacts and Git LFS
Trained models can be relatively large.

This project uses Git Large File Storage (Git LFS) for large model artifacts.

Install Git LFS:

git lfs install
Track large model files:

git lfs track "*.faiss"
git lfs track "*.joblib"
Then:

git add .gitattributes
Check tracked LFS files:

git lfs ls-files
Security
Never commit:

.env
API keys
passwords
tokens
private credentials
Gemini credentials should be provided through:

GEMINI_API_KEY
Environment variables should be used instead of hard-coded credentials.

If an API key is ever accidentally exposed, revoke/rotate it immediately.

GitHub Development Workflow
Recommended workflow:

Modify code
    ↓
Run tests
    ↓
Run benchmark / sample prediction
    ↓
Review generated artifacts
    ↓
Check Git status
    ↓
Check secrets
    ↓
Commit
    ↓
Push
Example:

pytest -q
Then:

git status
Then:

git diff --cached
before committing.

Performance Summary
Current MVTec benchmark:

Category	Status
bottle	Evaluated
cable	Evaluated
capsule	Evaluated
carpet	Evaluated
grid	Evaluated
hazelnut	Evaluated
leather	Evaluated
metal_nut	Evaluated
pill	Evaluated
screw	Evaluated
tile	Evaluated
toothbrush	Evaluated
transistor	Evaluated
wood	Evaluated
zipper	Evaluated
Aggregate:

Mean Image AUROC : 0.9380
Mean Pixel AUROC : 0.9597
Example Output
A typical inspection workflow produces:

reports/
└── inspection/
    └── bottle/
        ├── inspection_report.html
        ├── inspection_log.csv
        ├── predictions.json
        ├── 000.png
        ├── 001.png
        ├── 002.png
        └── ...
The HTML report provides a convenient way to review multiple inspection results.

Technology Stack
Programming
Python
Machine Learning / Computer Vision
PyTorch
Vision Transformer
PatchCore-style anomaly detection
scikit-learn
OpenCV
NumPy
Nearest-Neighbor Search
FAISS / nearest-neighbor based search
Reporting
HTML
JSON
CSV
AI Integration
Google Gemini API
gemini-2.5-flash-lite
Testing
pytest
Deployment / Development
Git
Git LFS
Docker
Research Direction
This project can be extended into a broader industrial visual inspection framework by combining:

Vision Transformer
        +
CNN Features
        +
Patch-Level Memory Bank
        +
Anomaly Localization
        +
AI-Assisted Inspection Reporting
The long-term objective is to develop a modular inspection platform that can be adapted to different industrial products and manufacturing environments.

Project Status
Core anomaly detection       ✅
MVTec dataset support        ✅
15-category benchmark        ✅
Image-level evaluation       ✅
Pixel-level evaluation       ✅
Anomaly visualization        ✅
Defect localization          ✅
HTML inspection reports      ✅
CSV/JSON logging             ✅
Gemini AI reporting          ✅
Automated tests              ✅
Git LFS support              ✅
Docker foundation            ✅
Production API               🚧
Web dashboard                🚧
Real-time camera integration 🚧
Model monitoring             🚧
License
This project is released under the license included in the repository.

See:

LICENSE
for the complete license terms.

Acknowledgements
This project uses the MVTec Anomaly Detection dataset for industrial anomaly detection experiments.

The implementation is intended for educational, research, and experimental industrial computer-vision applications.

Author
A Monish Kumar

Industrial Anomaly Detection
Computer Vision • Machine Learning • Deep Learning

Quick Start
For a quick test after installation:

anomaly-detect --help
Run the test suite:

pytest -q
Run a benchmark:

anomaly-detect benchmark `
  --data-root "D:\mvtec_anomaly_detection" `
  --out-dir ".\outputs\benchmark" `
  --save-models ".\models"
Run a prediction:

anomaly-detect predict `
  --model-dir ".\models\bottle" `
  --images "D:\mvtec_anomaly_detection\bottle\test" `
  --out-dir ".\reports\inspection\bottle" `
  --component "bottle"
Enable Gemini reporting:

$env:GEMINI_API_KEY = Read-Host "Enter Gemini API key"
Then:

anomaly-detect predict `
  --model-dir ".\models\bottle" `
  --images "D:\mvtec_anomaly_detection\bottle\test" `
  --out-dir ".\reports\inspection\bottle" `
  --report `
  --component "bottle"
Industrial Anomaly Detection — Deep Learning based visual inspection with localization, evaluation, reporting, and AI-assisted analysis.


### One thing I recommend before pasting

Because your **Git root is `industrial-anomaly-detection (3)`**, paste this into:

```text
C:\Users\Monis\Downloads\industrial-anomaly-detection (3)\README.md
not the nested folder.

Also, I intentionally didn't claim unsupported production capabilities as completed. The README separates what you have already implemented from future upgrades, which will look much better on GitHub than overstating the project.


give me as a file

Done. Here is the complete GitHub-ready README file:


Place it at:

C:\Users\Monis\Downloads\industrial-anomaly-detection (3)\README.md
Then we'll move on to .gitignore + .gitattributes + Git LFS setup before committing. 



Library
/
README.md



Industrial Anomaly Detection
Vision Transformer + PatchCore based industrial defect detection and localization system with automated inspection reports and optional Gemini AI analysis.

Overview
This project implements an end-to-end industrial visual anomaly detection system designed to detect and localize defects in manufactured products.

The system uses deep visual features, a PatchCore-style memory bank, nearest-neighbor anomaly scoring, and image-level/pixel-level evaluation to identify abnormal regions without requiring manually labeled defect masks during training.

The project is evaluated using the MVTec Anomaly Detection (MVTec AD) dataset and provides:

Training of category-specific anomaly detection models

Image-level anomaly detection

Pixel-level anomaly localization

Anomaly heatmaps

Defect bounding-box visualization

MVTec benchmark evaluation

Automated HTML inspection reports

CSV/JSON prediction logs

Optional Gemini AI generated inspection notes

Automated test suite

Git LFS support for trained model artifacts

Key Features
1. Unsupervised Anomaly Detection
The model is trained primarily using normal images and learns the distribution of normal visual features.

Defects can then be detected as regions that differ significantly from the learned normal feature distribution.

2. Patch-Level Feature Extraction
Input images are resized and processed using a deep vision feature extractor.

The image is represented using multiple local patch-level feature vectors rather than treating the entire image as a single feature.

This makes the system suitable for detecting localized manufacturing defects.

3. PatchCore-Style Memory Bank
Normal patch features are stored in a memory bank.

To reduce memory requirements, coreset sampling is used to retain a representative subset of normal features.

During inference, test patches are compared against the memory bank using nearest-neighbor search.

4. Image-Level Anomaly Detection
The system produces an anomaly score for the complete image.

Conceptually:

Normal image
     ↓
Feature extraction
     ↓
Patch features
     ↓
Nearest-neighbor comparison
     ↓
Patch anomaly scores
     ↓
Image anomaly score
     ↓
Normal / Anomalous
5. Pixel-Level Localization
Patch-level anomaly scores are converted into an anomaly map.

The system can identify where an abnormal region is located and generate a visual overlay.

6. Defect Localization
Detected anomaly regions can be converted into bounding boxes.

Example:

Input Image
     ↓
Anomaly Map
     ↓
Thresholding
     ↓
Connected Components
     ↓
Defect Region
     ↓
Bounding Box
7. Automated Inspection Reports
For each prediction, the system can generate:

Original image

Anomaly visualization

Prediction result

Anomaly score

Threshold

Hotspot information

Defect localization

Optional AI inspection note

The inspection results are organized into HTML reports.

8. Gemini AI Reporting
The system optionally integrates with Google's Gemini API to generate human-readable inspection notes for anomalous samples.

Gemini is used as a reporting/interpretation layer and does not replace the anomaly detector.

The core detection pipeline remains independent of the AI reporting component.

The project supports:

gemini-2.5-flash-lite
An API key is supplied through an environment variable and is never intended to be committed to Git.

System Architecture
                    MVTec AD Dataset
                           │
                           ▼
                  ┌──────────────────┐
                  │ Image Discovery  │
                  └────────┬─────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │ Image Preprocess │
                  │   224 × 224      │
                  └────────┬─────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ Deep Feature Extractor│
                │      ViT / CNN       │
                └──────────┬───────────┘
                           │
                           ▼
                 ┌────────────────────┐
                 │ Patch-Level Feature│
                 │     Extraction     │
                 └─────────┬──────────┘
                           │
              ┌────────────┴────────────┐
              │                         │
              ▼                         ▼
      Normal Training Images       Test Images
              │                         │
              ▼                         ▼
      ┌─────────────────┐       ┌─────────────────┐
      │ Memory Bank     │       │ Patch Features  │
      │ Construction    │       │                 │
      └────────┬────────┘       └────────┬────────┘
               │                         │
               ▼                         ▼
      ┌─────────────────┐       ┌─────────────────┐
      │ Coreset Sampling│       │ Nearest Neighbor│
      └────────┬────────┘       └────────┬────────┘
               │                         │
               └────────────┬────────────┘
                            ▼
                    ┌────────────────┐
                    │ Anomaly Scores │
                    └───────┬────────┘
                            │
              ┌─────────────┴─────────────┐
              │                           │
              ▼                           ▼
       Image-Level Score          Pixel-Level Map
              │                           │
              ▼                           ▼
       Normal / Anomaly          Heatmap / Bounding Box
              │                           │
              └─────────────┬─────────────┘
                            ▼
                   Inspection Report
                            │
                            ▼
                     Optional Gemini
                       AI Analysis
Methodology
Step 1 — Image Preprocessing
Images are resized to:

224 × 224
and passed through the feature extraction pipeline.

Step 2 — Deep Feature Extraction
The system extracts high-dimensional visual representations from the input image.

The project is designed around patch-level representations so that both global image abnormality and localized defects can be detected.

Step 3 — Normal Feature Memory Bank
Normal training samples are processed to obtain patch-level feature vectors.

These features form the normality reference:

Normal Training Images
        ↓
Feature Extraction
        ↓
Patch Features
        ↓
Memory Bank
Step 4 — Coreset Sampling
The complete memory bank can become very large.

Coreset sampling is therefore used to retain a representative subset of the normal feature distribution.

This reduces memory and nearest-neighbor search requirements while retaining representative normal patterns.

Step 5 — Nearest-Neighbor Anomaly Scoring
For a test image:

Test Patch
    ↓
Feature Vector
    ↓
Nearest Normal Feature
    ↓
Distance
    ↓
Patch Anomaly Score
Higher distance indicates greater deviation from the learned normal distribution.

Step 6 — Image-Level Decision
Patch anomaly scores are aggregated to obtain an image-level anomaly score.

The score is compared against a threshold to classify the sample as:

NORMAL
or

ANOMALOUS
Step 7 — Anomaly Localization
Patch-level scores are mapped back to the image to create an anomaly heatmap.

The system can then identify concentrated abnormal regions and generate bounding boxes around detected areas.

Dataset
This project uses the MVTec Anomaly Detection (MVTec AD) dataset.

The benchmark contains multiple industrial object and texture categories with normal and anomalous samples.

The project currently supports these 15 categories:

bottle
cable
capsule
carpet
grid
hazelnut
leather
metal_nut
pill
screw
tile
toothbrush
transistor
wood
zipper
Expected dataset location:

D:\mvtec_anomaly_detection
The path can be changed using the command-line options provided by the project.

Benchmark Results
A full benchmark was performed across all 15 supported MVTec categories.

Aggregate Results
Metric	Mean Score
Image AUROC	0.9380
Pixel AUROC	0.9597
Interpretation
Image AUROC — 0.9380

Measures the ability of the system to distinguish anomalous images from normal images.

Pixel AUROC — 0.9597

Measures the ability of the system to distinguish anomalous pixels/regions from normal pixels.

The benchmark results are generated from the project's evaluation pipeline and should be interpreted as dataset-specific experimental results rather than a guarantee of performance on real production data.

Project Structure
industrial-anomaly-detection (3)/
│
├── .github/
│   └── workflows/
│
├── industrial-anomaly-detection/
│   │
│   ├── src/
│   │   └── industrial_anomaly/
│   │       ├── __init__.py
│   │       ├── api.py
│   │       ├── cli.py
│   │       ├── config.py
│   │       ├── data.py
│   │       ├── detector.py
│   │       ├── evaluation.py
│   │       ├── features.py
│   │       ├── inspection_report.py
│   │       ├── memory_bank.py
│   │       ├── report.py
│   │       ├── scoring.py
│   │       └── visualization.py
│   │
│   ├── tests/
│   ├── scripts/
│   ├── pyproject.toml
│   └── ...
│
├── models/
│   ├── bottle/
│   ├── cable/
│   ├── capsule/
│   ├── carpet/
│   ├── grid/
│   ├── hazelnut/
│   ├── leather/
│   ├── metal_nut/
│   ├── pill/
│   ├── screw/
│   ├── tile/
│   ├── toothbrush/
│   ├── transistor/
│   ├── wood/
│   └── zipper/
│
├── outputs/
│   └── benchmark/
│
├── results/
├── reports/
│   └── inspection/
│
├── .gitignore
├── .gitattributes
├── Dockerfile
├── LICENSE
└── README.md
Installation
Requirements
Recommended environment:

Python 3.11+
Windows / Linux
Git
Git LFS
The project was developed and tested using Python 3.11.

Clone the Repository
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd "industrial-anomaly-detection (3)"
Create a Virtual Environment
Windows
python -m venv .venv
.\.venv\Scripts\Activate.ps1
Linux
python3 -m venv .venv
source .venv/bin/activate
Install the Project
pip install --upgrade pip
pip install -e .
Dataset Setup
Download the MVTec AD dataset and extract it.

Example:

D:\mvtec_anomaly_detection
The resulting structure should resemble:

mvtec_anomaly_detection/
├── bottle/
├── cable/
├── capsule/
├── carpet/
├── grid/
├── hazelnut/
├── leather/
├── metal_nut/
├── pill/
├── screw/
├── tile/
├── toothbrush/
├── transistor/
├── wood/
└── zipper/
Command Line Interface
The project provides the anomaly-detect command.

Check available commands:

anomaly-detect --help
Available workflows include:

train
evaluate
benchmark
report
predict
Training
To train a model for one MVTec category:

anomaly-detect train `
  --data-root "D:\mvtec_anomaly_detection" `
  --component "bottle" `
  --out-dir ".\models\bottle"
View all available arguments:

anomaly-detect train --help
Evaluation
Evaluate a trained model:

anomaly-detect evaluate `
  --data-root "D:\mvtec_anomaly_detection" `
  --component "bottle" `
  --model-dir ".\models\bottle" `
  --out-dir ".\outputs\bottle"
Evaluation results include metrics and visualization artifacts.

Full MVTec Benchmark
Run the benchmark across all supported categories:

anomaly-detect benchmark `
  --data-root "D:\mvtec_anomaly_detection" `
  --out-dir ".\outputs\benchmark" `
  --save-models ".\models"
This produces benchmark artifacts such as:

outputs/
└── benchmark/
and category-specific model artifacts under:

models/
├── bottle/
├── cable/
├── capsule/
...
└── zipper/
Prediction
Run prediction on an image:

anomaly-detect predict `
  --model-dir ".\models\bottle" `
  --images "D:\mvtec_anomaly_detection\bottle\test\broken_large\000.png" `
  --out-dir ".\reports\inspection\bottle" `
  --component "bottle"
Inspection Reports
The prediction pipeline can generate an HTML inspection report.

Example:

anomaly-detect predict `
  --model-dir ".\models\bottle" `
  --images "D:\mvtec_anomaly_detection\bottle\test" `
  --out-dir ".\reports\inspection\bottle" `
  --component "bottle"
The report can contain:

inspection_report.html
inspection_log.csv
predictions.json
along with generated visualization images.

Gemini AI Inspection Reporting
Gemini integration is optional.

The anomaly detector itself does not require Gemini.

To enable Gemini reporting, set the API key as an environment variable.

Windows PowerShell
$env:GEMINI_API_KEY = Read-Host "Enter Gemini API key"
Verify that the environment variable exists without printing the secret:

if ($env:GEMINI_API_KEY) {
    Write-Host "GEMINI_API_KEY is configured"
}
Run prediction with AI reporting:

anomaly-detect predict `
  --model-dir ".\models\bottle" `
  --images "D:\mvtec_anomaly_detection\bottle\test" `
  --out-dir ".\reports\inspection\bottle" `
  --report `
  --component "bottle"
Current recommended model:

gemini-2.5-flash-lite
Important Security Rule
Never commit an API key to GitHub.

Do not place credentials directly inside Python source code.

Use:

GEMINI_API_KEY
as an environment variable.

Benchmark Reporting
Benchmark results can also be summarized through the reporting pipeline.

Example:

anomaly-detect report `
  --results ".\results\mvtec\results.csv"
Check available options:

anomaly-detect report --help
Generated Artifacts
The project generates several types of artifacts.

Metrics
metrics.json
results.csv
Visualizations
evaluation.png
heatmaps
bounding-box visualizations
Prediction Logs
inspection_log.csv
predictions.json
HTML Inspection Reports
inspection_report.html
Testing
The project includes an automated test suite.

Run:

pytest -q
The current test suite contains:

20 tests
Current validation:

20 passed
Tests cover core functionality such as:

Data handling

Feature processing

Scoring

Memory bank behavior

Detection pipeline

Reporting functionality

Project components

Production-Oriented Design
The project is structured to separate major responsibilities.

Data
 │
 ├── data.py
 │
 ▼
Feature Extraction
 │
 ├── features.py
 │
 ▼
Memory Bank
 │
 ├── memory_bank.py
 │
 ▼
Anomaly Detection
 │
 ├── detector.py
 │
 ├── scoring.py
 │
 ▼
Evaluation
 │
 ├── evaluation.py
 │
 ▼
Visualization
 │
 ├── visualization.py
 │
 ▼
Inspection Reporting
 │
 ├── inspection_report.py
 │
 ▼
AI Reporting
 │
 └── report.py
This modular structure makes it easier to:

Replace the feature extractor

Change anomaly scoring

Experiment with memory-bank strategies

Add new datasets

Add APIs

Add production monitoring

Run experiments independently

API Layer
The project also contains an API module:

src/industrial_anomaly/api.py
This provides a foundation for exposing anomaly detection functionality through a service layer.

A future deployment can connect:

Frontend
   ↓
REST API
   ↓
Anomaly Detection Engine
   ↓
Model
   ↓
Prediction
Example Industrial Workflow
A potential production workflow is:

Camera
  ↓
Product Image
  ↓
Image Validation
  ↓
Preprocessing
  ↓
Feature Extraction
  ↓
Anomaly Detection
  ↓
Anomaly Score
  ↓
Defect Localization
  ↓
Pass / Fail Decision
  ↓
Inspection Report
  ↓
Database / Dashboard
The current repository focuses primarily on the computer-vision anomaly detection and inspection-reporting components.

Advantages
Does not require defect labels for the core training process

Uses normal samples as the reference distribution

Supports image-level anomaly detection

Supports pixel-level localization

Produces visual explanations

Supports automated inspection reports

Can process multiple MVTec categories

Provides quantitative evaluation metrics

Supports optional AI-generated inspection notes

Includes automated tests

Uses a modular Python architecture

Limitations
This project is an experimental/research-oriented industrial inspection system.

Dataset Dependency
Performance is measured using the MVTec AD dataset and may change significantly on different manufacturing environments.

Domain Shift
Changes in camera, lighting, product orientation, background, manufacturing process, surface material, and image resolution can affect anomaly scores.

Threshold Calibration
Production systems require threshold calibration based on the actual acceptable defect rate and manufacturing requirements.

Real-Time Requirements
The current implementation is not presented as a certified real-time production inspection system.

Deployment would require benchmarking:

Inference latency

Throughput

CPU/GPU utilization

Memory usage

Camera acquisition time

End-to-end processing time

Future Improvements
Potential future upgrades include:

1. Production REST API
Expose inference through FastAPI:

POST /predict
POST /batch-predict
GET /health
GET /model-info
2. Web Dashboard
Create a React-based dashboard showing:

Live inspection results

Anomaly score

Defect location

Historical inspections

Category statistics

Failure rate

Confidence trends

3. Real-Time Camera Inspection
Industrial Camera
       ↓
Frame Capture
       ↓
Anomaly Detection
       ↓
Defect Localization
       ↓
Pass / Fail
4. Model Monitoring
Track:

Mean anomaly score

False-positive rate

False-negative rate

Drift

Model version

Data distribution changes

5. Model Versioning
Store model metadata such as:

Model Version
Dataset Version
Feature Extractor
Training Date
Threshold
Evaluation Metrics
Software Version
6. Containerized Deployment
The project includes a Dockerfile foundation for reproducible deployment.

A production architecture could use:

Docker
   ↓
FastAPI
   ↓
Inference Engine
   ↓
Model
7. GPU Acceleration
For higher-throughput industrial environments, GPU inference can be used to reduce processing latency.

8. Additional Feature Extractors
The architecture can be extended to experiment with:

Vision Transformers

CNN backbones

WideResNet

Hybrid feature extractors

Other anomaly detection architectures

Reproducibility
For reproducible experiments, record:

Python version
Package versions
Dataset version
Model configuration
Feature extractor
Memory bank size
Coreset ratio
Anomaly threshold
Hardware
Random seed
Model Artifacts and Git LFS
Trained models can be relatively large.

This project uses Git Large File Storage (Git LFS) for large model artifacts.

Install Git LFS:

git lfs install
Track large model files:

git lfs track "*.faiss"
git lfs track "*.joblib"
Then:

git add .gitattributes
Check tracked LFS files:

git lfs ls-files
Security
Never commit:

.env
API keys
passwords
tokens
private credentials
Gemini credentials should be provided through:

GEMINI_API_KEY
If an API key is ever accidentally exposed, revoke/rotate it immediately.

GitHub Development Workflow
Recommended workflow:

Modify code
    ↓
Run tests
    ↓
Run benchmark / sample prediction
    ↓
Review generated artifacts
    ↓
Check Git status
    ↓
Check secrets
    ↓
Commit
    ↓
Push
Example:

pytest -q
git status
git diff --cached
before committing.

Performance Summary
Current MVTec benchmark:

Category	Status
bottle	Evaluated
cable	Evaluated
capsule	Evaluated
carpet	Evaluated
grid	Evaluated
hazelnut	Evaluated
leather	Evaluated
metal_nut	Evaluated
pill	Evaluated
screw	Evaluated
tile	Evaluated
toothbrush	Evaluated
transistor	Evaluated
wood	Evaluated
zipper	Evaluated
Aggregate:

Mean Image AUROC : 0.9380
Mean Pixel AUROC : 0.9597
Example Output
A typical inspection workflow produces:

reports/
└── inspection/
    └── bottle/
        ├── inspection_report.html
        ├── inspection_log.csv
        ├── predictions.json
        ├── 000.png
        ├── 001.png
        ├── 002.png
        └── ...
The HTML report provides a convenient way to review multiple inspection results.

Technology Stack
Programming
Python
Machine Learning / Computer Vision
PyTorch
Vision Transformer
PatchCore-style anomaly detection
scikit-learn
OpenCV
NumPy
Nearest-Neighbor Search
FAISS / nearest-neighbor based search
Reporting
HTML
JSON
CSV
AI Integration
Google Gemini API
gemini-2.5-flash-lite
Testing
pytest
Deployment / Development
Git
Git LFS
Docker
Research Direction
This project can be extended into a broader industrial visual inspection framework by combining:

Vision Transformer
        +
CNN Features
        +
Patch-Level Memory Bank
        +
Anomaly Localization
        +
AI-Assisted Inspection Reporting
The long-term objective is to develop a modular inspection platform that can be adapted to different industrial products and manufacturing environments.

Project Status
Core anomaly detection       ✅
MVTec dataset support        ✅
15-category benchmark        ✅
Image-level evaluation       ✅
Pixel-level evaluation       ✅
Anomaly visualization        ✅
Defect localization          ✅
HTML inspection reports      ✅
CSV/JSON logging             ✅
Gemini AI reporting          ✅
Automated tests              ✅
Git LFS support              ✅
Docker foundation            ✅
Production API               🚧
Web dashboard                🚧
Real-time camera integration 🚧
Model monitoring             🚧
License
This project is released under the license included in the repository.

See:

LICENSE
for the complete license terms.

Acknowledgements
This project uses the MVTec Anomaly Detection dataset for industrial anomaly detection experiments.

The implementation is intended for educational, research, and experimental industrial computer-vision applications.

Author
A Monish Kumar

Industrial Anomaly Detection
Computer Vision • Machine Learning • Deep Learning

Quick Start
After installation:

anomaly-detect --help
Run the test suite:

pytest -q
Run a benchmark:

anomaly-detect benchmark `
  --data-root "D:\mvtec_anomaly_detection" `
  --out-dir ".\outputs\benchmark" `
  --save-models ".\models"
Run a prediction:

anomaly-detect predict `
  --model-dir ".\models\bottle" `
  --images "D:\mvtec_anomaly_detection\bottle\test" `
  --out-dir ".\reports\inspection\bottle" `
  --component "bottle"
Enable Gemini reporting:

$env:GEMINI_API_KEY = Read-Host "Enter Gemini API key"
Then:

anomaly-detect predict `
  --model-dir ".\models\bottle" `
  --images "D:\mvtec_anomaly_detection\bottle\test" `
  --out-dir ".\reports\inspection\bottle" `
  --report `
  --component "bottle"
Industrial Anomaly Detection — Deep Learning based visual inspection with localization, evaluation, reporting, and AI-assisted analysis.

