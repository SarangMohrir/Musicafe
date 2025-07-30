# 🎵 Music Transcription Pipeline

A complete deep learning system that converts audio recordings into musical notation (MIDI files). This project uses TensorFlow/Keras to train neural networks that can transcribe monophonic and polyphonic music from audio to symbolic representation.

## 🎯 What This Pipeline Does

1. **Audio Analysis** → Converts audio waveforms to mel-spectrograms
2. **Deep Learning** → Trains CNN+LSTM models to predict musical notes
3. **MIDI Generation** → Outputs transcribed music as MIDI files
4. **Deployment** → Exports models for production use

## 🏗️ Pipeline Architecture

```
Audio File (.wav) 
    ↓
[Audio Preprocessing]
    ↓ 
Mel-Spectrogram Features
    ↓
[Deep Learning Model]
    ↓
Note Predictions (Piano Roll)
    ↓
[Post-Processing]
    ↓
MIDI File (.mid)
```

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Pipeline Components](#pipeline-components)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Data Pipeline](#data-pipeline)
- [Training Process](#training-process)
- [Evaluation](#evaluation)
- [Deployment](#deployment)
- [Advanced Features](#advanced-features)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## 🚀 Quick Start

### Option 1: Jupyter Notebook (Recommended)
```bash
# 1. Install dependencies
pip install tensorflow librosa pretty_midi soundfile matplotlib scikit-learn

# 2. Run the complete notebook
jupyter notebook music_transcription.ipynb

# 3. Execute all cells in order
# The notebook will automatically:
# - Generate sample data
# - Train a model
# - Create transcriptions
# - Export deployment files
```

### Option 2: Python Script
```bash
# 1. Run quick test
python music_transcription.py

# 2. Full pipeline
python music_transcription.py full

# 3. Transcribe your audio
python music_transcription.py transcribe your_audio.wav
```

## 🔧 Pipeline Components

### 1. Audio Preprocessing (`AudioProcessor`)
- **Input**: WAV audio files (any sample rate)
- **Processing**: 
  - Resampling to 22.05 kHz
  - Mel-spectrogram extraction (128 mel bins)
  - Log-scale conversion
- **Output**: Time-frequency representations

```python
processor = AudioProcessor(sr=22050, n_mels=128)
audio = processor.load_audio('song.wav')
mel_spec = processor.extract_melspectrogram(audio)
```

### 2. MIDI Processing (`MIDIProcessor`)
- **Input**: MIDI files with note information
- **Processing**:
  - Piano roll conversion (88 keys, A0-C8)
  - Time alignment with audio features
  - Binary note activation matrices
- **Output**: Ground truth labels for training

```python
midi_processor = MIDIProcessor(fs=100, min_note=21, max_note=108)
midi = midi_processor.load_midi('song.mid')
piano_roll = midi_processor.midi_to_piano_roll(midi)
```

### 3. Dataset Management (`MusicDataset`)
- **Function**: Pairs audio files with corresponding MIDI files
- **Features**:
  - Automatic file matching (same basename)
  - Batch loading and preprocessing
  - Train/validation splitting
  - Sequence padding for batch processing

```python
dataset = MusicDataset('audio/', 'midi/')
X, y = dataset.get_all_data()  # Audio features, Piano rolls
```

### 4. Neural Network Models
- **Simple Model**: Conv1D + LSTM architecture
- **Advanced Model**: Bidirectional LSTM + Attention
- **Custom Models**: Easily extensible framework

```python
model = create_simple_model(input_dim=128, output_dim=88)
model.compile(optimizer='adam', loss='binary_crossentropy')
```

## 📦 Installation

### Requirements
- Python 3.8+
- TensorFlow 2.13+
- Audio processing libraries

### Full Installation
```bash
# Clone or download the project
git clone <repository-url>
cd music-transcription-pipeline

# Install with conda (recommended)
conda create -n music-transcription python=3.9
conda activate music-transcription
pip install -r requirements.txt

# Or install with pip
pip install tensorflow>=2.13.0 librosa>=0.10.1 pretty_midi>=0.2.10 soundfile>=0.12.1 matplotlib>=3.7.2 scikit-learn>=1.3.0 jupyter
```

### Development Installation
```bash
# For development work
pip install -e .
pip install pytest black flake8  # Additional dev tools
```

## 💻 Usage

### Basic Usage

#### 1. Generate Sample Data
```python
# Creates synthetic audio-MIDI pairs for testing
generate_sample_data(num_samples=5)
```

#### 2. Train Model
```python
# Train with default parameters
model, history = train_model_notebook(model, dataset, epochs=15)

# Custom training
model = create_simple_model(input_dim=128, output_dim=88)
model, history = train_model_notebook(model, dataset, epochs=50)
```

#### 3. Transcribe Audio
```python
# Single file transcription
transcribe_audio_notebook('audio.wav', model, 'output.mid')

# Batch transcription
transcribe_custom_audio()  # Transcribes all files in data/audio/
```

### Advanced Usage

#### Custom Model Architecture
```python
def create_custom_model(input_dim, output_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(None, input_dim)),
        # Your custom layers here
        tf.keras.layers.Dense(output_dim, activation='sigmoid')
    ])
    return model
```

#### Real-time Transcription Simulation
```python
simulate_real_time_transcription(
    model, 
    'audio.wav', 
    window_size=2.0,  # 2-second windows
    overlap=0.5       # 50% overlap
)
```

#### Performance Analysis
```python
# Detailed model evaluation
analyze_performance(model, dataset)

# Test different thresholds
test_different_thresholds(model, dataset, [0.3, 0.5, 0.7])
```

## 🧠 Model Architecture

### Simple Model (Default)
```
Input (None, 128)
    ↓
Conv1D(64, kernel=3) + ReLU + Dropout(0.2)
    ↓
Conv1D(128, kernel=3) + ReLU + Dropout(0.2)
    ↓
LSTM(256, return_sequences=True) + Dropout(0.3)
    ↓
LSTM(128, return_sequences=True) + Dropout(0.3)
    ↓
Dense(88, sigmoid) → Output (None, 88)
```

### Advanced Model (Optional)
```
Input (None, 128)
    ↓
Multi-layer Conv1D with BatchNorm
    ↓
Bidirectional LSTM layers
    ↓
Dense layers with dropout
    ↓
Multi-task outputs (notes + onsets)
```

### Model Specifications
- **Input**: Mel-spectrogram features (128 dimensions)
- **Output**: Piano roll predictions (88 piano keys)
- **Loss**: Binary crossentropy (multi-label classification)
- **Metrics**: Accuracy, Precision, Recall, F1-score

## 📊 Data Pipeline

### Data Flow
```
Raw Audio (.wav) → Feature Extraction → Neural Network → Post-processing → MIDI (.mid)
```

### Feature Engineering
1. **Audio Preprocessing**:
   - Sample rate: 22.05 kHz
   - Frame size: 2048 samples
   - Hop length: 512 samples
   - Mel filters: 128

2. **MIDI Processing**:
   - Piano range: A0 (21) to C8 (108) 
   - Time resolution: 100 Hz
   - Binary activation: Note on/off

3. **Data Alignment**:
   - Automatic time synchronization
   - Sequence padding for batching
   - Train/validation splitting

### Supported Formats
- **Audio Input**: WAV, MP3, FLAC, M4A
- **MIDI Input**: Standard MIDI files (.mid)
- **Output**: MIDI files, visualizations, model exports

## 🏋️ Training Process

### Training Pipeline
1. **Data Preparation**:
   ```python
   dataset = MusicDataset('audio/', 'midi/')
   X, y = dataset.get_all_data()
   X_padded = pad_sequences(X)  # Ensure uniform length
   ```

2. **Model Creation**:
   ```python
   model = create_simple_model(input_dim=128, output_dim=88)
   ```

3. **Training Configuration**:
   ```python
   callbacks = [
       tf.keras.callbacks.EarlyStopping(patience=10),
       tf.keras.callbacks.ReduceLROnPlateau(factor=0.5)
   ]
   ```

4. **Training Execution**:
   ```python
   history = model.fit(
       X_train, y_train,
       validation_data=(X_val, y_val),
       epochs=epochs,
       callbacks=callbacks
   )
   ```

### Training Monitoring
- **Real-time Metrics**: Loss, accuracy, precision, recall
- **Visualization**: Training curves, validation performance
- **Early Stopping**: Prevents overfitting
- **Learning Rate Scheduling**: Adaptive optimization

### Hyperparameter Tuning
```python
# Example hyperparameter configurations
configs = [
    {'learning_rate': 0.001, 'dropout': 0.3, 'lstm_units': 256},
    {'learning_rate': 0.0005, 'dropout': 0.4, 'lstm_units': 512},
    # Add more configurations
]
```

## 📈 Evaluation

### Evaluation Metrics
1. **Frame-level Accuracy**: Percentage of correctly predicted time frames
2. **Note-level Precision**: True positives / (True positives + False positives)
3. **Note-level Recall**: True positives / (True positives + False negatives)
4. **F1-Score**: Harmonic mean of precision and recall

### Evaluation Tools
```python
# Comprehensive evaluation
accuracy, pred_probs, pred_binary = evaluate_and_visualize(model, dataset)

# Performance analysis
analyze_performance(model, dataset)

# Threshold optimization
test_different_thresholds(model, dataset, [0.3, 0.4, 0.5, 0.6, 0.7])
```

### Visualization Features
- **Training Progress**: Loss and metric curves
- **Prediction Comparison**: Ground truth vs predictions
- **Piano Roll Visualization**: Note activations over time
- **Spectrogram Analysis**: Input feature visualization

## 🚀 Deployment

### Export Formats
The pipeline supports multiple deployment formats:

```python
export_model_for_deployment()
```

Creates:
- **`music_transcriber.keras`**: Native Keras format (recommended)
- **`music_transcriber.h5`**: Legacy HDF5 format
- **`music_transcriber_savedmodel/`**: TensorFlow SavedModel
- **`music_transcriber.tflite`**: TensorFlow Lite (mobile)
- **`deployment_script.py`**: Standalone inference script

### Production Deployment

#### 1. Standalone Script
```bash
cd exports/
python deployment_script.py audio_file.wav output.mid
```

#### 2. Web API Integration
```python
from tensorflow import keras
model = keras.models.load_model('music_transcriber.keras')

def transcribe_api(audio_data):
    # Your API logic here
    prediction = model.predict(audio_data)
    return convert_to_midi(prediction)
```

#### 3. Mobile Deployment
```python
# Use the .tflite model for mobile apps
interpreter = tf.lite.Interpreter(model_path='music_transcriber.tflite')
```

### Performance Optimization
- **Model Quantization**: Reduce model size for deployment
- **Batch Processing**: Handle multiple files efficiently
- **GPU Acceleration**: Utilize CUDA for faster inference
- **Model Pruning**: Remove unnecessary parameters

## 🔬 Advanced Features

### 1. Real-time Transcription
```python
simulate_real_time_transcription(
    model=model,
    audio_path='live_audio.wav',
    window_size=2.0,    # Process 2-second chunks
    overlap=0.5         # 50% overlap between windows
)
```

### 2. Multi-instrument Support
```python
# Extend for different instruments
guitar_model = create_model_for_instrument(
    instrument='guitar',
    note_range=(40, 88),  # Guitar range
    model_type='advanced'
)
```

### 3. Polyphonic Transcription
- Handles multiple simultaneous notes
- Advanced post-processing for chord detection
- Separates overlapping frequencies

### 4. Custom Data Integration
```python
# Use your own dataset
dataset = MusicDataset('my_audio/', 'my_midi/')
model = train_on_custom_data(dataset, epochs=100)
```

## 🛠️ Troubleshooting

### Common Issues

#### 1. Import Errors
```bash
# Solution: Install missing packages
pip install librosa pretty_midi soundfile
```

#### 2. Memory Issues
```python
# Solution: Reduce batch size or sequence length
model.fit(X, y, batch_size=1)  # Smaller batches
```

#### 3. Poor Transcription Quality
```python
# Solutions:
# 1. More training data
generate_sample_data(num_samples=50)

# 2. Longer training
model, history = train_model_notebook(model, dataset, epochs=100)

# 3. Different threshold
pred_binary = (pred_piano_roll > 0.3).astype(np.float32)  # Lower threshold
```

#### 4. Model Export Errors
```python
# Solution: Use different export format
try:
    model.save('model.keras')  # New format
except:
    model.save('model.h5')     # Legacy format
```

### Performance Issues

#### Slow Training
- **Use GPU**: Enable CUDA acceleration
- **Reduce Model Size**: Fewer LSTM units
- **Smaller Dataset**: Start with fewer samples

#### Poor Accuracy
- **More Data**: Increase dataset size
- **Data Quality**: Ensure audio-MIDI alignment
- **Model Complexity**: Try advanced architectures
- **Hyperparameter Tuning**: Adjust learning rate, dropout

#### Memory Errors
- **Batch Size**: Reduce from 4 to 1
- **Sequence Length**: Limit audio duration
- **Model Size**: Use fewer parameters

### Debug Mode
```python
# Enable verbose logging
tf.debugging.set_log_device_placement(True)

# Monitor memory usage
import psutil
print(f"Memory usage: {psutil.virtual_memory().percent}%")
```

## 📚 Understanding the Science

### Audio Signal Processing
- **Mel-scale**: Perceptually motivated frequency scale
- **Spectrograms**: Time-frequency analysis of audio
- **Feature Engineering**: Extracting relevant musical information

### Deep Learning Components
- **Convolutional Layers**: Capture local frequency patterns
- **LSTM Networks**: Model temporal dependencies in music
- **Multi-label Classification**: Predict multiple simultaneous notes

### Music Information Retrieval
- **Onset Detection**: Identifying when notes start
- **Pitch Estimation**: Determining fundamental frequencies
- **Harmonic Analysis**: Understanding chord structures

## 🤝 Contributing

### Development Setup
```bash
git clone <repository>
cd music-transcription-pipeline
pip install -e .
pip install -r requirements-dev.txt
```

### Contributing Guidelines
1. **Code Style**: Follow PEP 8, use Black formatter
2. **Testing**: Add unit tests for new features
3. **Documentation**: Update README and docstrings
4. **Performance**: Profile code for optimization opportunities

### Areas for Contribution
- **New Model Architectures**: Transformer-based models
- **Data Augmentation**: Improve training robustness
- **Multi-instrument Support**: Extend beyond piano
- **Real-time Processing**: Optimize for live transcription
- **Web Interface**: Create user-friendly GUI

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **TensorFlow/Keras**: Deep learning framework
- **Librosa**: Audio analysis library
- **Pretty MIDI**: MIDI file processing
- **Research Papers**: Various music transcription studies
- **Open Source Community**: Tools and libraries used

## 📞 Support

- **Issues**: Report bugs via GitHub Issues
- **Discussions**: Join community discussions
- **Documentation**: Check inline code documentation
- **Examples**: See `examples/` directory for more use cases

---

## 🎼 Example Results

After running the complete pipeline:

```
🎵 MUSIC TRANSCRIPTION PIPELINE COMPLETE! 🎵
===============================================

📊 Project Summary:
  ✅ Generated 5 sample audio files
  ✅ Generated 5 sample MIDI files  
  ✅ Trained music transcription model
  ✅ Model saved to: models/music_transcriber.keras
  📁 Model file size: 45.2 MB
  🎵 Generated transcriptions: transcribed_sample_00.mid

📊 Final Training Metrics:
  Validation Loss: 0.234
  Validation Accuracy: 0.874
  Validation Precision: 0.782
  Validation Recall: 0.698

🚀 Next Steps:
  1. Upload your own .wav files to data/audio/ directory
  2. Run transcribe_custom_audio() to transcribe them
  3. Experiment with different model architectures
  4. Use the exported models for deployment

🎊 Congratulations! You've built a complete music transcription system!
```

Ready to transcribe some music? 🎵
