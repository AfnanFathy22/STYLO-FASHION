# STYLO – An AI-Powered Fashion Recommendation System

**Your Intelligent Fashion Assistant**

STYLO is an AI-powered fashion recommendation system designed to provide a personalized and interactive fashion shopping experience. It combines Generative AI, Retrieval-Augmented Generation (RAG), Multimodal Search, Computer Vision, and Natural Language Processing (NLP) to help users discover fashion products, build complete outfits, analyze customer reviews, and visualize clothing through virtual try-on technology.

The system focuses on modest fashion and integrates intelligent product discovery, personalized fashion experiences, outfit building, sentiment analysis, and AI-powered virtual try-on into one platform.

---

# ✨ Key Features

## 1. Fashion RAG & Multimodal Smart Search

An intelligent fashion retrieval system that allows users to discover products using natural language queries, uploaded images, or a combination of both.

### Features
- Arabic and English natural language search.
- Image-based product retrieval.
- Multimodal search combining text and image inputs.
- Intelligent query understanding and intent extraction.
- Semantic product retrieval using FashionCLIP embeddings.
- Similar product recommendations.
- Filtering by category, gender, fabric, and maximum price.
- New-arrivals-only and fast-shipping filters.
- Context-aware responses in the same language used by the customer.
- Retrieval from a locally stored Qdrant vector database.

### Search Workflow

1. The user enters a text query, uploads an image, or provides both.
2. The vision model analyzes the uploaded image.
3. The LLM understands the user's query, detects the language, and extracts the search intent.
4. FashionCLIP generates image and text embeddings.
5. The system combines the available search information.
6. Qdrant retrieves the most relevant products using vector similarity search.
7. The LLM generates a contextual response in Arabic or English.

### Technology Stack
- FashionCLIP
- Groq API
- Llama Vision / LLM
- Qdrant Vector Database
- FastAPI
- Gradio

### API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/search` | Text-based product search |
| POST | `/search-image` | Image and text-based product search |
| GET | `/health` | Service health check |

---

## 2. AI Virtual Try-On (IDM-VTON)

An AI-powered virtual try-on feature based on IDM-VTON, allowing users to visualize how garments look on a person using advanced diffusion models.

### Features
- Upload a person image.
- Select or upload a garment image.
- Generate realistic virtual try-on results.
- Preserve garment details and improve clothing alignment.
- Visualize fashion items on a person before purchasing.

### Technology Stack
- IDM-VTON
- Stable Diffusion XL
- Diffusion Models
- IP-Adapter
- Human Parsing
- DensePose
- OpenPose
- PyTorch
- Gradio

### Research References
- [IDM-VTON Project Page](https://idm-vton.github.io/)
- [Research Paper – arXiv](https://arxiv.org/abs/2403.05139)
- [Official GitHub Repository](https://github.com/yisol/IDM-VTON)
- [Hugging Face Demo](https://huggingface.co/spaces/yisol/IDM-VTON)
- [Pretrained Model](https://huggingface.co/yisol/IDM-VTON)

---

## 3. AI Outfit Builder

An intelligent outfit-building feature that allows users to create and visualize a complete look by combining different clothing items.

The Outfit Builder uses a sequential virtual try-on workflow to apply multiple garments to the same person image, producing a final visualization of the combined outfit.

### Features
- Select an upper-body garment.
- Select a lower-body garment.
- Combine different clothing pieces into a complete outfit.
- Generate virtual try-on results for individual clothing items.
- Sequentially process multiple garments.
- Preserve the previously generated result when applying the next garment.
- Display the final combined outfit on the same person.

### Workflow

1. Upload or select a person image.
2. Choose an upper-body garment.
3. Run IDM-VTON to apply the selected upper-body garment.
4. Receive the generated image.
5. Use the generated image as the person input for the next try-on operation.
6. Choose a lower-body garment.
7. Run IDM-VTON again to apply the lower-body garment.
8. Display the final image showing the combined outfit.

### Technology Stack
- IDM-VTON
- Stable Diffusion XL
- PyTorch
- Python
- Image Processing
- Computer Vision

---

## 4. Customer Review Sentiment Analysis

An NLP-powered feature designed to analyze customer reviews and understand opinions about fashion products.

The system processes customer feedback, identifies sentiment polarity, and classifies reviews into Positive, Negative, and Neutral categories. The results provide useful insights into customer satisfaction and product performance.

### Features
- Analyze customer reviews and feedback.
- Classify reviews into Positive, Negative, and Neutral sentiments.
- Process and clean textual reviews.
- Identify positive and negative customer experiences.
- Calculate sentiment distribution and percentages.
- Generate product-level sentiment summaries.
- Identify frequently mentioned positive and negative aspects.
- Support data-driven product evaluation.

### Sentiment Analysis Workflow

**Step 1 – Data Collection**

Collect customer reviews from the Modanisa reviews dataset, including the review text and associated product information when available.

**Step 2 – Data Preprocessing**

Prepare the reviews for NLP processing through:
- Removing duplicate reviews.
- Handling missing and invalid values.
- Removing unnecessary characters and noise.
- Normalizing text.
- Tokenization.
- Removing irrelevant words when appropriate.
- Preparing the cleaned text for feature extraction.

**Step 3 – Text Representation**

Convert textual reviews into numerical representations that can be processed by a machine learning model.

Possible approaches include:
- TF-IDF vectorization.
- Word embeddings.
- Transformer-based text embeddings.

**Step 4 – Sentiment Classification**

Apply a sentiment classification model to predict the sentiment of each review.

The classification categories are:

- **Positive:** The customer expresses satisfaction or a favorable opinion.
- **Negative:** The customer expresses dissatisfaction or an unfavorable opinion.
- **Neutral:** The review contains objective information or no clear positive or negative sentiment.

A supervised classification approach can use algorithms such as Logistic Regression or other NLP classification models, depending on the selected training setup.

**Step 5 – Sentiment Analysis Results**

Aggregate the predictions to calculate:
- Number of positive reviews.
- Number of negative reviews.
- Number of neutral reviews.
- Sentiment percentages.
- Overall customer sentiment distribution.

**Step 6 – Insights and Visualization**

Present the results through charts and summaries to help understand customer opinions and identify common product-related issues and strengths.

### Dataset

Modanisa customer reviews dataset containing approximately 1,200 reviews.

### Sentiment Distribution

| Sentiment | Percentage |
|---|---:|
| Positive | 61.8% |
| Negative | 25.6% |
| Neutral | 12.6% |

### Technology Stack
- Python
- Pandas
- NumPy
- Natural Language Processing (NLP)
- Scikit-learn
- TF-IDF
- Machine Learning
- Text Preprocessing
- Sentiment Classification
- Matplotlib / Seaborn for visualization

---

# 🏗️ System Architecture

STYLO integrates multiple AI technologies and services into one fashion recommendation ecosystem.

| Component | Responsibility |
|---|---|
| Fashion RAG | Intelligent product discovery and retrieval |
| FashionCLIP | Multimodal image and text embeddings |
| Qdrant | Vector similarity search |
| Groq / LLM | Query understanding and response generation |
| IDM-VTON | AI-powered virtual try-on |
| Outfit Builder | Sequential combination of clothing items |
| Sentiment Analysis | Customer review classification and insights |
| FastAPI | Backend API services |
| Gradio | Interactive user interfaces |

---

# 🛠️ Technology Stack

### Programming Languages
- Python
- SQL

### Artificial Intelligence
- Machine Learning
- Deep Learning
- Generative AI
- Large Language Models (LLMs)
- Retrieval-Augmented Generation (RAG)
- Computer Vision
- Multimodal AI
- Natural Language Processing (NLP)
- Sentiment Analysis

### Frameworks & Libraries
- PyTorch
- Hugging Face Transformers
- FashionCLIP
- FastAPI
- Gradio
- Diffusers
- OpenCV
- Scikit-learn
- Pandas
- NumPy

### Vector Database & Infrastructure
- Qdrant
- REST APIs
- Git & GitHub

---

# 📁 Project Structure

```text
Fashion_Project/
│
├── IDM-VTON/
│   ├── ckpt/
│   │   ├── densepose/
│   │   ├── humanparsing/
│   │   ├── openpose/
│   │   ├── ip_adapter/
│   │   └── image_encoder/
│   │
│   ├── gradio_demo/
│   ├── inference.py
│   ├── inference_dc.py
│   ├── train_xl.py
│   ├── train_xl.sh
│   ├── inference.sh
│   └── environment.yaml
│
├── rag_local/
│   ├── app.py
│   ├── requirements.txt
│   ├── .env
│   ├── .env.example
│   └── data/
│       ├── modanisa_valid.csv
│       └── qdrant_storage/
│
├── outfit_builder/
│   └── ...
│
├── sentiment_analysis/
│   └── ...
│
└── README.md
```

*Note: This is an illustrative project structure. Adjust filenames and directories to match the actual implementation.*

---

# 🚀 Fashion RAG – Local Installation & Setup

The Fashion RAG service runs locally as a standard Python service without Google Colab, Google Drive mounting, Colab secrets, or ngrok.

## 1. Folder Layout

Place the entire `rag_local` folder inside your `Fashion_Project` directory, next to `IDM-VTON`.

```text
Fashion_Project/
│
├── IDM-VTON/
│
└── rag_local/
    ├── app.py
    ├── requirements.txt
    ├── .env
    └── data/
        ├── modanisa_valid.csv
        └── qdrant_storage/
```

Copy your CSV file and the `qdrant_storage` folder downloaded from Google Drive into `rag_local/data/` exactly as shown above.

If you prefer to keep them somewhere else, update the following paths at the top of `app.py`:

```python
CONFIG["valid_csv_path"]
CONFIG["qdrant_local"]
```

## 2. Create a Virtual Environment and Install Dependencies

You can reuse the virtual environment from your IDM-VTON project:

`Fashion_Project/IDM-VTON/venv`

Alternatively, create a separate environment for the RAG service.

Using PowerShell:

```powershell
cd Fashion_Project\rag_local

python -m venv venv

venv\Scripts\activate

pip install -r requirements.txt
```

If you have an NVIDIA GPU and want CUDA-accelerated CLIP embedding, install the CUDA-compatible PyTorch build first.

Refer to the official PyTorch installation guide:

https://pytorch.org/get-started/locally/

Then install the remaining dependencies:

```powershell
pip install -r requirements.txt
```

## 3. Configure the Groq API Key

Create your local environment file:

```powershell
copy .env.example .env
```

Open `.env` and replace the placeholder with your actual Groq API key:

```env
GROQ_API_KEY=your_groq_api_key_here
```

## 4. Run the RAG Service

Start the service:

```powershell
python app.py
```

Expected startup output:

```text
Using local Qdrant storage at ...
Loaded N products
FashionCLIP (compat) loaded
Collections: ['modanisa_image', 'modanisa_multimodal']

Starting RAG API at http://localhost:8000
(docs: /docs, health: /health)

Gradio UI available at http://localhost:8000/gradio
```

---

# 🖥️ Using the Fashion RAG Gradio Interface

Open the following URL in your browser:

http://localhost:8000/gradio

The interactive interface provides a point-and-click experience for the complete fashion search workflow.

### Available Operations
- Enter a query in Arabic, English, or a mixture of both.
- Leave the query empty and search using an image.
- Upload a fashion product image.
- Filter products by category.
- Filter products by gender.
- Filter products by fabric.
- Set a maximum price.
- Enable new-arrivals-only filtering.
- Enable fast-shipping filtering.
- Adjust the number of returned results.

Click **Search** to display:
- The parsed query.
- The applied filters.
- The assistant's response.
- Matched products as cards.

The assistant responds in the same language used by the customer.

## Health Check

Open:

http://localhost:8000/health

Or use PowerShell:

```powershell
curl http://localhost:8000/health
```

## Test Text-Based Search

```powershell
curl -X POST http://localhost:8000/search -H "Content-Type: application/json" -d "{\"query\": \"black abaya\", \"k\": 5}"
```

---

# 🖼️ Image + Arabic/English Search

The `/search-image` endpoint implements the complete multimodal search workflow:

**Image + Text → Understanding → Retrieval → Recommendations**

Users can submit an image and optionally provide a text query in Arabic, English, or a mixture of both.

The `chat()` pipeline:

1. Processes the uploaded image using the vision model.
2. Processes the text query using the LLM.
3. Detects the user's language.
4. Extracts search intent and relevant attributes.
5. Combines image and text information into a unified CLIP search.
6. Retrieves relevant products from Qdrant.
7. Generates a response in the same language used by the customer.

## Test Through Swagger UI

Open:

http://localhost:8000/docs

1. Expand `POST /search-image`.
2. Click **Try it out**.
3. Upload an image file.
4. Optionally enter a query, such as `فستان أسود للمناسبات` or `black dress`.
5. Set the number of results.
6. Click **Execute**.

## Test Using PowerShell

```powershell
curl -X POST http://localhost:8000/search-image `
  -F "image=@C:\path\to\photo.jpg" `
  -F "query=فستان أسود للمناسبات" `
  -F "k=5"
```

### API Response

The response includes:

- `message`: The assistant's reply in Arabic or English.
- `products`: The matched product results.
- `meta`: Additional search information.

---

# 👗 IDM-VTON – Installation & Setup

IDM-VTON is the official implementation of the research paper:

**Improving Diffusion Models for Authentic Virtual Try-on in the Wild**

### Official Resources

- [Project Page](https://idm-vton.github.io/)
- [Research Paper](https://arxiv.org/abs/2403.05139)
- [Hugging Face Demo](https://huggingface.co/spaces/yisol/IDM-VTON)
- [Hugging Face Model](https://huggingface.co/yisol/IDM-VTON)
- [GitHub Repository](https://github.com/yisol/IDM-VTON)

## Requirements

Clone the official repository:

```bash
git clone https://github.com/yisol/IDM-VTON.git

cd IDM-VTON
```

Create and activate the Conda environment:

```bash
conda env create -f environment.yaml

conda activate idm
```

---

# 📂 IDM-VTON Data Preparation

IDM-VTON supports VITON-HD and DressCode datasets.

## 1. VITON-HD

Download the VITON-HD dataset:

https://github.com/shadow2496/VITON-HD

After downloading the dataset:

- Move `vitonhd_test_tagged.json` into the test folder.
- Move `vitonhd_train_tagged.json` into the train folder.

### Expected Dataset Structure

```text
train/
├── image/
├── image-densepose/
├── agnostic-mask/
├── cloth/
└── vitonhd_train_tagged.json

test/
├── image/
├── image-densepose/
├── agnostic-mask/
├── cloth/
└── vitonhd_test_tagged.json
```

## 2. DressCode

Download the DressCode dataset:

https://github.com/aimagelab/dress-code

Pre-computed DensePose images and captions are available here:

https://kaistackr-my.sharepoint.com/:u:/g/personal/cpis7_kaist_ac_kr/EaIPRG-aiRRIopz9i002FOwBDa-0-BHUKVZ7Ia5yAVVG3A?e=YxkAip

The original implementation uses Detectron2 for obtaining DensePose images.

Additional reference:

https://github.com/sangyun884/HR-VITON/issues/45

### Expected Dataset Structure

```text
DressCode/
├── dresses/
│   ├── images/
│   ├── image-densepose/
│   ├── dc_caption.txt
│   └── ...
│
├── lower_body/
│   ├── images/
│   ├── image-densepose/
│   ├── dc_caption.txt
│   └── ...
│
└── upper_body/
    ├── images/
    ├── image-densepose/
    ├── dc_caption.txt
    └── ...
```

---

# 🧠 IDM-VTON Training

## Preparation

Download the pretrained IP-Adapter for SDXL:

`IP-Adapter/sdxl_models/ip-adapter-plus_sdxl_vit-h.bin`

Also download the image encoder:

`IP-Adapter/models/image_encoder`

Official repository:

https://github.com/tencent-ailab/IP-Adapter

Clone the repository:

```bash
git clone https://huggingface.co/h94/IP-Adapter
```

Move the IP-Adapter checkpoint into `ckpt/ip_adapter` and the image encoder into `ckpt/image_encoder`.

### Expected Checkpoint Structure

```text
ckpt/
├── ip_adapter/
│   └── ip-adapter-plus_sdxl_vit-h.bin
│
└── image_encoder/
    └── ...
```

## Start Training

Using the Python training script:

```bash
accelerate launch train_xl.py \
    --gradient_checkpointing --use_8bit_adam \
    --output_dir=result --train_batch_size=6 \
    --data_dir=DATA_DIR
```

Alternatively, run the training script:

```bash
sh train_xl.sh
```

---

# 🎨 IDM-VTON Inference

## 1. VITON-HD Inference

Run inference using:

```bash
accelerate launch inference.py \
    --width 768 --height 1024 --num_inference_steps 30 \
    --output_dir "result" \
    --unpaired \
    --data_dir "DATA_DIR" \
    --seed 42 \
    --test_batch_size 2 \
    --guidance_scale 2.0
```

Alternatively:

```bash
sh inference.sh
```

## 2. DressCode Inference

For DressCode, specify the desired category using the `--category` argument.

```bash
accelerate launch inference_dc.py \
    --width 768 --height 1024 --num_inference_steps 30 \
    --output_dir "result" \
    --unpaired \
    --data_dir "DATA_DIR" \
    --seed 42 \
    --test_batch_size 2 \
    --guidance_scale 2.0 \
    --category "upper_body"
```

Alternatively:

```bash
sh inference.sh
```

---

# 🖥️ IDM-VTON Local Gradio Demo

Download the required pretrained checkpoints for human parsing and pose estimation:

https://huggingface.co/spaces/yisol/IDM-VTON/tree/main/ckpt

Place the checkpoints inside the `ckpt` directory.

### Required Checkpoint Structure

```text
ckpt/
├── densepose/
│   └── model_final_162be9.pkl
│
├── humanparsing/
│   ├── parsing_atr.onnx
│   └── parsing_lip.onnx
│
└── openpose/
    └── ckpts/
        └── body_pose_model.pth
```

### Run the Gradio Demo

```python
python gradio_demo/app.py
```

The demo will launch a local Gradio interface for virtual try-on.

---

# 🔗 Service Integration

Both the Fashion RAG service and IDM-VTON Gradio application can run simultaneously on the same machine using different ports.

| Service | Port |
|---|---:|
| Fashion RAG API | 8000 |
| IDM-VTON Gradio | 7860 (default example) |

A frontend or the IDM-VTON interface can communicate with the Fashion RAG service using:

```text
POST http://localhost:8000/search
```

### Integration Workflow

1. The user searches for a fashion item.
2. Fashion RAG retrieves matching products.
3. The user selects a product.
4. The selected product's garment image is passed to IDM-VTON.
5. IDM-VTON generates a virtual try-on result.
6. The Outfit Builder can use sequential try-on operations to combine multiple garments.
7. Sentiment Analysis provides customer review insights to support product evaluation.

---

# ⚙️ Important Notes

### Fashion RAG
- The first run downloads the `patrickjohncyh/fashion-clip` model weights from Hugging Face.
- The model weights are several hundred MB and are cached locally under `~/.cache/huggingface`.
- Internet access is required for the initial download.
- The local Qdrant folder must be available before starting the service.
- If Qdrant reports a lock-file error from a previous session, `app.py` already clears `data/qdrant_storage/.lock` automatically on startup.

### IDM-VTON
- Pretrained checkpoints are required before running inference.
- GPU acceleration is recommended.
- Training and inference require the appropriate datasets and dependencies.
- The virtual try-on implementation uses pretrained diffusion models and supporting computer vision components.

### Sentiment Analysis
- The dataset contains approximately 1,200 customer reviews.
- The reported sentiment distribution is 61.8% Positive, 25.6% Negative, and 12.6% Neutral.
- Classification performance should be evaluated using appropriate metrics such as Accuracy, Precision, Recall, and F1-score.
- The classification algorithm and preprocessing steps should match the actual trained implementation.

### Security
- Store API keys in local environment files.
- Never upload `.env` files containing real credentials.
- Avoid publishing private datasets, model weights, or restricted project assets.

---

# 🙏 Acknowledgements

Special thanks to the researchers and open-source communities behind:

- [IDM-VTON](https://github.com/yisol/IDM-VTON) – Virtual try-on implementation.
- [IP-Adapter](https://github.com/tencent-ailab/IP-Adapter) – Image prompt adapter.
- [OOTDiffusion](https://github.com/levihsu/OOTDiffusion) – Virtual try-on research.
- [DCI-VTON](https://github.com/bcmi/DCI-VTON-Virtual-Try-On) – Virtual try-on research.
- [SCHP](https://github.com/GoGoDuck912/Self-Correction-Human-Parsing) – Human parsing.
- [DensePose](https://github.com/facebookresearch/DensePose) – Human pose estimation.
- [ZeroGPU](https://huggingface.co/zero-gpu-explorers) – GPU resources.
- [FashionCLIP](https://huggingface.co/patrickjohncyh/fashion-clip) – Fashion image and text embeddings.
- [Qdrant](https://qdrant.tech/) – Vector database.
- [Gradio](https://www.gradio.app/) – Interactive interfaces.
- [Groq](https://groq.com/) – LLM inference services.

---

# 📚 Citation

If you use IDM-VTON in your research, please cite the original paper:

```bibtex
@article{choi2024improving,
  title={Improving Diffusion Models for Authentic Virtual Try-on in the Wild},
  author={Choi, Yisol and Kwak, Sangkyung and Lee, Kyungmin and Choi, Hyungwon and Shin, Jinwoo},
  journal={arXiv preprint arXiv:2403.05139},
  year={2024}
}
```

---

# 📄 License

The original IDM-VTON code and checkpoints are released under the **CC BY-NC-SA 4.0** license.

https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode

Other third-party components and models may have separate licenses. Refer to their original repositories for licensing and usage requirements.
