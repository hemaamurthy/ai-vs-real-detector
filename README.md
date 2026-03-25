# 🚀 AI vs Real Content Detector

A multi-modal AI system that detects whether content is **AI-generated or human-created** using Machine Learning and NLP techniques.

---

## 🔍 Features

- 📝 **Text Detection**
  - Uses Transformer-based model (RoBERTa)
  - Combines AI probability + perplexity scoring
  - Provides human-readable interpretation

- 🖼️ **Image Detection**
  - Trained CNN model (ResNet18)
  - Detects AI-generated vs real images
  - Outputs prediction with confidence %

- 🌐 **Web Application**
  - Built using Flask
  - Clean and interactive UI
  - Handles invalid inputs gracefully

---

## 🧠 How It Works

### Text Detection
- Uses pretrained transformer model
- Calculates:
  - AI probability
  - Perplexity (language complexity)
- Combines both to classify:
  - AI Generated / Human Written / Uncertain

### Image Detection
- Trained on AI vs Real image dataset
- Uses CNN (ResNet18 architecture)
- Predicts with confidence score

---

## 🛠️ Tech Stack

- Python
- Flask
- PyTorch
- Transformers (HuggingFace)
- OpenCV
- HTML, CSS

---

## 📂 Project Structure
ai_detector/
│
├── backend/
│ ├── text_model.py
│ ├── image_model.py
│ └── utils.py
│
├── templates/
│ └── index.html
│
├── app.py
├── image_model.pth
└── README.md

---

## ▶️ Run Locally

```bash
# Clone repo
git clone https://github.com/hemaamurthy/ai-vs-real-detector.git

# Go to folder
cd ai-vs-real-detector

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run app
python app.py

📊 Output Examples
Text:
Human Written (Low AI probability, High perplexity)
AI Generated (High AI probability, Low perplexity)
Image:
AI Generated (87.45%)
Real Image (92.10%)
⚠️ Limitations
Text detection is not 100% accurate (industry-wide challenge)
Image model trained on limited dataset (can be improved further)
🚀 Future Improvements
Improve model accuracy with larger datasets
Add video detection module
Deploy application online (Render / AWS)
Add user history tracking

🙋‍♀️ Author

Hemalatha Pitla
AI/ML Enthusiast