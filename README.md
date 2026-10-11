# 🧍 Human Pose Data Pipeline: Scrape → Auto-Label → Train → Audit

An end-to-end AI/ML project built for the **Augle AI Internship Assignment**.

This project automates the process of collecting human images from the internet, identifying people in those images, training a custom deep learning model, and checking whether the dataset and training results are reliable.

Instead of doing everything manually, I connected different tools to create an automated AI pipeline.

## 🚀 Project Overview

The project has three main phases:

| Phase | What it does | Technology used |
|---|---|---|
| 📥 Phase 1: Data Collection | Collects human images from the internet and creates bounding-box annotations | Playwright, Python, Grounding DINO |
| 🧠 Phase 2: Model Training | Prepares images and trains a CNN model to predict a person's location | OpenCV, PyTorch |
| 🤖 Phase 3: Data Auditing | Checks dataset quality and answers questions about training results | Python, LLM API, RAG, Tool Calling |

### 🔄 How the Project Works

**Collect Images → Detect People → Create Annotations → Validate Data → Train Model → Evaluate Results → Audit with AI**

Each phase uses the output of the previous phase.

## 📁 Project Structure

```text
augle-ai-assessment/
│
├── collect_and_annotate.py    # Collects images from the web
├── setup_clean_images.py      # Alternative image downloading script
├── generate_annotations.py    # Creates COCO annotations
├── dataset.py                 # Loads and prepares images
├── train.py                   # Builds and trains the CNN
├── inference_eval.py          # Tests the trained model
├── final_verify.py            # Checks dataset correctness
├── llm_auditor.py             # AI-based dataset auditor
├── requirements.txt           # Required Python libraries
│
└── data/
    ├── raw_images/            # Downloaded images
    ├── rejected_images/       # Images that failed filtering
    ├── annotations/           # Annotation JSON files
    ├── outputs/               # Sample prediction images
    ├── metrics.csv             # Training results
    └── model_checkpoint.pth   # Saved model weights
```

## ⚙️ Technologies Used

- **Python:** Main programming language used to build the pipeline.
- **Playwright:** Automates a web browser to collect image URLs from web pages.
- **Grounding DINO:** Detects objects such as people in images without requiring a custom-trained detector.
- **OpenCV:** Reads, resizes, and processes images.
- **PyTorch:** Builds and trains the custom CNN model.
- **COCO JSON:** Stores image information and bounding-box annotations in a standard format.
- **LLM API:** Allows users to ask questions about dataset quality and training results in natural language.
- **RAG:** Retrieves relevant project data before generating an answer.
- **Tool Calling:** Allows the AI to request calculations from Python functions instead of guessing numerical results.

## 📥 Phase 1: Image Collection and Auto-Labeling

### What does this phase do?

The first step is to collect human images from the internet and identify where people appear in those images.

### Step 1: Collect images using Playwright

Playwright controls a browser automatically. My script opens Wikimedia Commons category pages, scrolls through the pages, collects image URLs, and downloads the images.

The script also removes duplicate URLs and filters out unsupported image formats.

**Why Playwright?**

Manually downloading hundreds of images takes time. Browser automation makes the collection process faster and repeatable.

### Step 2: Detect people using Grounding DINO

Grounding DINO is a vision model that can detect objects based on text descriptions.

For example, I can provide the prompt:

`a person`

The model examines an image and predicts where it finds a person.

### Step 3: Create bounding boxes

A **bounding box** is a rectangle drawn around an object.

For example:

- `x` = horizontal starting position
- `y` = vertical starting position
- `w` = width of the rectangle
- `h` = height of the rectangle

These values are stored in the format:

`[x, y, width, height]`

### Step 4: Save annotations in COCO JSON

COCO JSON is a standard format for storing image details and object annotations.

My annotation file stores:

- Image filename, width, and height
- Object category, such as `person`
- Bounding-box coordinates
- Annotation and image IDs

The script also checks whether boxes fit within image boundaries and whether the image files can be read.

**Output:** A collection of images and their corresponding annotations, ready for model training.

## 🧠 Phase 2: Dataset Preparation and Model Training

### Step 1: Load images using PyTorch

I created a custom PyTorch Dataset class in `dataset.py`.

A Dataset class tells PyTorch how to load each image and find its corresponding annotation.

OpenCV is used to read and resize images to `640 × 640` pixels. The image is converted from BGR to RGB and its pixel values are scaled to the range `0–1`.

The bounding boxes are resized using the same horizontal and vertical scaling factors as the image.

### Step 2: Build a CNN from scratch

I created a custom CNN called `MultiTaskCustomPoseCNN`.

**CNN (Convolutional Neural Network)** is a deep learning model that learns visual patterns from images.

My model contains:

- **Convolution layers:** Learn visual features such as edges, shapes, and body patterns.
- **Batch Normalization:** Helps stabilize the learning process.
- **ReLU:** Adds non-linearity so the network can learn more complex patterns.
- **Max Pooling:** Reduces the size of feature maps.
- **Adaptive Average Pooling:** Converts feature maps into a fixed-size representation.
- **Fully Connected Layers:** Use the learned features to make predictions.
- **Dropout:** Helps reduce overfitting by randomly disabling some neurons during training.

The model has two output heads:

1. **Classification head:** Predicts the object class.
2. **Bounding-box head:** Predicts the object's location.

I used a custom CNN instead of a pretrained backbone to meet the assignment's requirement to build and train a model from scratch.

### Step 3: Train the model

The training process follows this sequence:

`Forward Pass → Calculate Loss → Backpropagation → Update Weights`

- **Forward pass:** The image goes through the network to produce predictions.
- **Loss:** Measures how different the predictions are from the target values.
- **Backpropagation:** Calculates how the model's weights should change to reduce the loss.
- **Optimizer:** Updates the model's weights.
- **Learning rate scheduler:** Reduces the learning rate during training according to a schedule.

I use Adam as the optimizer, Cross-Entropy Loss for classification, and Smooth L1 Loss for bounding-box prediction.

The training script records the losses in `metrics.csv` and saves the best model checkpoint based on validation loss.

### Step 4: Evaluate the model

After training, `inference_eval.py` tests the model on validation images.

It reports classification accuracy and normalized bounding-box coordinate MAE.

**MAE (Mean Absolute Error)** measures the average absolute difference between predicted and actual coordinates. Lower MAE generally means more accurate coordinates, but it does not measure overall object-detection quality by itself.

The script also saves a sample prediction image so that the predicted box can be inspected visually.

**Output:** A trained model, training logs, evaluation results, and a sample prediction image.

## 🤖 Phase 3: LLM-Based Dataset Auditor

### What does this phase do?

The final phase helps inspect the dataset and training results using natural-language questions.

For example, a user can ask:

- Are there any invalid bounding boxes?
- How many annotations are available?
- Are any boxes outside the image boundaries?
- How did validation loss change during training?
- Are there suspiciously small bounding boxes?

### How does it work?

**1. Data ingestion**

The auditor reads the COCO annotation JSON and the training metrics CSV.

**2. Python audit tools**

I created Python functions that check bounding-box validity, count categories, calculate box areas, and retrieve training metrics.

**3. RAG (Retrieval-Augmented Generation)**

RAG means retrieving relevant information from a data source before asking an LLM to generate an answer.

In this project, the dataset and training logs provide the information needed to answer questions about the project.

**4. LLM and tool calling**

The LLM interprets the user's question and can request the appropriate Python audit function. Python performs the actual calculation, and the result can then be explained in natural language.

This is useful because the LLM does not need to guess counts or calculate dataset statistics from memory.

**Output:** An interactive AI auditor that helps users understand dataset quality and training results.

## 📸 Project Outputs


### 1. Dataset Audit Results

The dataset auditor validates image annotations and bounding boxes. The results show 400 images and annotations in the first audit, with no malformed annotations or invalid bounding boxes reported.

![Dataset Audit Results](data\show_result\Screenshot01.png)

### 2. Final Pipeline Verification

The final verification script validates the dataset and checks bounding-box dimensions, image references, and category annotations. The displayed results show 401 images and annotations, with no malformed annotations, out-of-bounds boxes, invalid box dimensions, or missing image references.

![Final Pipeline Verification](data\show_result\Screenshot02.png)



*Note: The local Python audit functions work without an LLM API call. Natural-language AI responses depend on the configured API and its availability.*

## ▶️ How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/augle-ai-assessment.git
cd augle-ai-assessment
```

Replace `<your-username>` with the actual GitHub username or repository owner.

### 2. Create a virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate
```

A virtual environment keeps the project's Python dependencies separate from other projects.

### 3. Install the required libraries

```bash
pip install -r requirements.txt
playwright install chromium
```

### 4. Run the pipeline

Run each command from the project root in the following order:

```bash
python collect_and_annotate.py
python generate_annotations.py
python final_verify.py
python train.py
python inference_eval.py
python llm_auditor.py
```

Set the LLM API key using the method required by your chosen provider before running the AI auditor. Never upload your API key to GitHub.

**Important:** The exact commands depend on the implementation of each script. If image annotations are not generated automatically by the collection script, make sure the Grounding DINO annotation step has been run and `dino_cache.json` is available before generating the COCO file.

## 📊 Results and Validation

The project includes checks to help identify data and training problems.

| Check | Purpose |
|---|---|
| Image validation | Finds unreadable or invalid images |
| Bounding-box validation | Checks box dimensions and image boundaries |
| COCO validation | Checks image references, category IDs, and annotation structure |
| Training loss | Tracks how much prediction error changes over time |
| Classification accuracy | Measures correct class predictions |
| Bounding-box MAE | Measures coordinate prediction error |
| Sample prediction | Helps visually inspect the model's output |

The local audit report can also count annotations, check missing image references, and identify invalid coordinates.

**Add your actual results here after running the complete pipeline.** Include the number of valid images, training and validation loss, classification accuracy, bounding-box MAE, and a sample prediction image.

## ⚠️ Limitations and Future Improvements

Every machine learning project has limitations. The current implementation can be improved in the following ways:

- **Better annotations:** Review Grounding DINO predictions and remove incorrect labels.
- **Data augmentation:** Add image rotation, scaling, and translation while correctly transforming the bounding boxes.
- **More training:** Experiment with more epochs and suitable hyperparameters.
- **Better evaluation:** Add Intersection over Union (IoU) and suitable detection metrics.
- **Early stopping:** Stop training when validation performance no longer improves.
- **More object classes:** Extend the model to distinguish standing, sitting, and kneeling poses.
- **Experiment tracking:** Add TensorBoard or Weights & Biases to visualize training progress.
- **Improved detection:** Support multiple people in one image instead of predicting only one box.
- **Reproducibility:** Pin exact dependency versions and document the complete setup.

## 📚 What I Learned

Through this project, I gained practical experience with:

- Automating image collection using Playwright.
- Understanding how zero-shot object detection works.
- Creating and validating COCO annotations.
- Working with image preprocessing and bounding-box geometry.
- Building a custom CNN using PyTorch.
- Implementing a training loop with backpropagation and an optimizer.
- Tracking model performance using validation metrics.
- Connecting Python audit tools with an LLM using tool calling.
- Understanding how RAG can help an AI answer questions using project data.
- Designing a multi-stage, reproducible AI/ML pipeline.

## 👨‍💻 Author

**Radheya Gaikwad**

B.E. Artificial Intelligence and Data Science

- GitHub: [RadheyaGaikwad](https://github.com/RadheyaGaikwad)
- LinkedIn: [Radheya Gaikwad](https://linkedin.com/in/radhegaikwad/)

---

⭐ *Built as part of the Augle AI Internship Assignment — AI/ML Engineering Track.*
#   A I _ A s s e s s m e n t  
 