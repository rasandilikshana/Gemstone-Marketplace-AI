# 🔮 Sri Lankan Gemstone Marketplace AI

**An AI-powered marketplace for Sri Lankan gemstone identification, listing, and trading with 2D-to-3D conversion capabilities.**

![Python](https://img.shields.io/badge/python-v3.12+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19.0-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🌟 Features

- **🤖 AI Gemstone Identification**: Identify 87 types of Sri Lankan gemstones with confidence scoring
- **📱 Web Interface**: User-friendly Streamlit-based marketplace interface
- **🖼️ Image Processing**: Advanced image preprocessing with OpenCV
- **💎 Marketplace Functions**: Create listings, price estimates, and marketplace actions
- **🎯 High Accuracy**: Based on InceptionV3 architecture trained on comprehensive gemstone dataset
- **🚀 Real-time Processing**: Instant gemstone identification from uploaded images

## 🔧 Technology Stack

- **Backend**: Python 3.12, TensorFlow 2.19.0, OpenCV
- **Frontend**: Streamlit web framework
- **AI Models**: InceptionV3-based CNN, Hugging Face Transformers
- **Image Processing**: PIL, OpenCV, NumPy
- **Deployment**: Local development with scalability options

## 📋 Prerequisites

- Python 3.8+ (3.12 recommended)
- 8GB+ RAM
- 50GB+ free disk space
- Optional: NVIDIA GPU for faster processing

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/rasandilikshana/Gemstone-Marketplace-AI.git
cd gemstone-marketplace-ai
```

### 2. Set Up Virtual Environment

```bash
# Create virtual environment
python3 -m venv gemstone_env

# Activate virtual environment
# Linux/Mac:
source gemstone_env/bin/activate
# Windows:
gemstone_env\Scripts\activate
```

### 3. Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install core libraries
pip install tensorflow
pip install opencv-python
pip install numpy pandas matplotlib pillow requests
pip install fastapi uvicorn streamlit
pip install transformers torch torchvision
pip install jupyter notebook
```

### 4. Download Pre-trained Model

```bash
# Clone the Sri Lankan gemstone AI model
git clone https://github.com/lasitha-theWolf/Gemstone-Identification-System-Using-Deep-Learning.git sri-lankan-gemstone-ai

# Navigate to project directory
cd sri-lankan-gemstone-ai
```

### 5. Test the Model

```bash
# Test model loading
python -c "
import tensorflow as tf
model = tf.keras.models.load_model('best_model.keras')
print('Model loaded successfully!')
print(f'Model can identify gemstones with input shape: {model.input_shape}')
"
```

### 6. Launch the Marketplace

```bash
# Start the web application
streamlit run marketplace_app.py
```

Open your browser and go to `http://localhost:8501`

## 📁 Project Structure

```
gemstone-marketplace-ai/
├── gemstone_env/              # Virtual environment
├── sri-lankan-gemstone-ai/    # Main AI project
│   ├── best_model.keras       # Pre-trained model (94MB)
│   ├── marketplace_app.py     # Main web application
│   ├── test_model.py         # Model testing script
│   ├── predict_sample.py     # Sample prediction script
│   ├── gemstones-images/     # Training dataset
│   │   ├── train/            # Training images (87 categories)
│   │   └── test/             # Test images
│   ├── amazonite_3.jpg       # Sample test image
│   └── aventurine green_3.jpg # Sample test image
├── requirements.txt          # Python dependencies
├── README.md                # This file
└── docs/                    # Documentation
```

## 🔍 Supported Gemstones (87 Types)

The AI can identify the following gemstone categories:

**Precious Stones:**
- Sapphire (Blue, Pink, Purple, Yellow)
- Ruby
- Emerald
- Diamond

**Semi-Precious Stones:**
- Amethyst, Citrine, Aquamarine
- Moonstone, Sunstone, Labradorite
- Tourmaline, Spinel, Tanzanite
- Garnet varieties, Quartz varieties
- And 70+ more categories

## 💻 Usage Examples

### Basic Gemstone Identification

```python
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model('best_model.keras')

# Predict gemstone
def identify_gemstone(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((256, 256))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) / 255.0
    
    predictions = model.predict(img_array)
    class_names = sorted(os.listdir('gemstones-images/train'))
    
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * np.max(predictions[0]), 2)
    
    return predicted_class, confidence

# Example usage
gemstone, confidence = identify_gemstone('your_gemstone.jpg')
print(f"Identified: {gemstone} (Confidence: {confidence}%)")
```

### Web Interface Features

1. **Upload Image**: Drag and drop or select gemstone images
2. **AI Analysis**: Get instant identification with confidence scores
3. **Top 5 Predictions**: See alternative possibilities
4. **Marketplace Actions**: Create listings and get price estimates
5. **Responsive Design**: Works on desktop and mobile browsers

## 🔧 Development Setup

### For Developers

```bash
# Install development dependencies
pip install jupyter notebook ipython

# Launch Jupyter for model exploration
jupyter notebook

# Run tests
python test_model.py
python predict_sample.py
```

### Environment Variables

Create a `.env` file for configuration:

```bash
# Model settings
MODEL_PATH=best_model.keras
IMAGE_SIZE=256
CONFIDENCE_THRESHOLD=0.1

# Web app settings
STREAMLIT_PORT=8501
DEBUG_MODE=True

# Future API settings
API_HOST=localhost
API_PORT=8000
```

## 📊 Model Performance

- **Training Dataset**: 87 gemstone categories
- **Model Architecture**: InceptionV3-based CNN
- **Input Size**: 256x256x3 RGB images
- **Model Size**: 94MB
- **Inference Time**: ~2 seconds per image (CPU)
- **Accuracy**: Varies by gemstone type and image quality

## 🚀 Deployment Options

### Local Development
```bash
streamlit run marketplace_app.py
```

### Docker Deployment
```dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "marketplace_app.py", "--server.address", "0.0.0.0"]
```

### Cloud Deployment

**Recommended platforms:**
- **Railway**: Free tier with 500 hours/month
- **Render**: Free static sites + 750 hours compute
- **Vercel**: Free for frontend + serverless functions
- **AWS/GCP/Azure**: For production scale

## 🔮 Future Enhancements

### Phase 1: Core Marketplace
- [ ] User authentication and profiles
- [ ] Gemstone listing creation and management
- [ ] Basic search and filtering
- [ ] Image gallery and 360° views

### Phase 2: Advanced AI Features
- [ ] 2D-to-3D conversion integration
- [ ] Gemtelligence professional analysis
- [ ] Price prediction algorithm
- [ ] Quality assessment automation

### Phase 3: E-commerce Integration
- [ ] Payment processing (Stripe/PayPal)
- [ ] Escrow services for high-value transactions
- [ ] Shipping and logistics integration
- [ ] Mobile app development

### Phase 4: Global Marketplace
- [ ] Multi-language support
- [ ] International compliance
- [ ] Advanced analytics dashboard
- [ ] API for third-party integrations

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 API Documentation

### Gemstone Identification API

```python
POST /api/identify
Content-Type: multipart/form-data

Parameters:
- image: Image file (JPG, PNG)
- confidence_threshold: Float (optional, default: 0.1)

Response:
{
  "gemstone": "Sapphire Blue",
  "confidence": 85.32,
  "alternatives": [
    {"gemstone": "Sapphire Purple", "confidence": 12.45},
    {"gemstone": "Iolite", "confidence": 2.23}
  ],
  "processing_time": 1.87
}
```

## 🔧 Troubleshooting

### Common Issues

**1. Model Loading Error**
```bash
# Solution: Check TensorFlow version
pip install tensorflow>=2.16.0
```

**2. CUDA Warnings**
```bash
# Normal on systems without NVIDIA GPU
# Model runs on CPU - performance may be slower
```

**3. Streamlit Port Already in Use**
```bash
# Use different port
streamlit run marketplace_app.py --server.port 8502
```

**4. Memory Issues**
```bash
# Reduce image batch size or use smaller model
# Monitor memory usage with top/htop
```

### Performance Optimization

**For CPU-only systems:**
```bash
# Set TensorFlow to CPU-only mode
export CUDA_VISIBLE_DEVICES=""
export TF_CPP_MIN_LOG_LEVEL=2
```

**For better performance:**
- Use SSD storage for faster model loading
- Increase RAM to 16GB+ for better caching
- Use GPU acceleration if available

## 📚 Resources and References

### Documentation
- [TensorFlow Documentation](https://www.tensorflow.org/guide)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [OpenCV Python Tutorials](https://opencv-python-tutroals.readthedocs.io/)

### Research Papers
- [Gemtelligence: Accelerating Gemstone Classification with Deep Learning](https://github.com/TommasoBendinelli/Gemtelligence)
- [Automatic Gemstone Classification Using Computer Vision](https://www.mdpi.com/2075-163X/12/1/60)

### Datasets
- [Kaggle Gemstone Datasets](https://www.kaggle.com/datasets/lsind18/gemstones-images)
- [Sri Lankan Gemstone Database](https://github.com/lasitha-theWolf/Gemstone-Identification-System-Using-Deep-Learning)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Acknowledgments

- **Lasitha Heenkenda** - Original Sri Lankan Gemstone AI model
- **Tommaso Bendinelli et al.** - Gemtelligence research
- **Alibaba Cloud** - Qwen language models
- **TensorFlow/Google** - AI framework and pre-trained models
- **Streamlit** - Web application framework

## 📞 Contact

- **Developer**: Rasan Dilikshana
- **Email**: rasandilikshana@gmail.com
- **LinkedIn**: [LinkedIn](https://www.linkedin.com/in/rasan-dilikshana/)
- **Project Repository**: [GitHub](https://github.com/rasandilikshana/Gemstone-Marketplace-AI.git)

## 💎 About Sri Lankan Gemstones

Sri Lanka, known as the "Gem Paradise," has been a source of precious gemstones for over 2,000 years. The island produces a remarkable variety of gems including:

- **Blue Sapphires** - World's finest quality
- **Padparadscha Sapphires** - Rare pink-orange variety
- **Star Sapphires and Rubies** - With natural asterism
- **Moonstones** - Blue and colorless varieties
- **Cat's Eye Chrysoberyl** - Highly prized variety

This marketplace aims to preserve and promote Sri Lanka's rich gemological heritage while leveraging modern AI technology for accurate identification and fair trading.

---

**⭐ Star this repository if you found it helpful!**

*Built with ❤️ for the Sri Lankan gemstone community*