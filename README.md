# NLP Text Summarization

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)](https://www.python.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-orange?style=flat-square&logo=huggingface)](https://huggingface.co/transformers/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.x-red?style=flat-square&logo=pytorch)](https://pytorch.org/)

This repository provides an end-to-end solution for text summarization, covering both abstractive and extractive approaches. It leverages state-of-the-art NLP models from the HuggingFace Transformers library to generate concise and coherent summaries from longer texts.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Models](#models)
- [Contributing](#contributing)
- [License](#license)

## Introduction
Text summarization is a critical task in natural language processing, aiming to create a short, accurate, and fluent summary of a longer document. This project explores various techniques and models to achieve high-quality summarization for different use cases.

## Features
- **Abstractive Summarization:** Generates new sentences to form a summary.
- **Extractive Summarization:** Selects key sentences from the original text.
- **Pre-trained Models:** Utilizes models like BART, T5, and Pegasus.
- **Evaluation Metrics:** ROUGE score calculation for summary quality.

## Project Structure
```
.gitignore
README.md
requirements.txt
src/
├── __init__.py
├── abstractive_summarizer.py
└── extractive_summarizer.py
data/
├── raw/
└── processed/
models/
```

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Wremn1987/NLP-Text-Summarization.git
   cd NLP-Text-Summarization
   ```
2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scriptsctivate`
   ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
- To run the abstractive summarizer:
  ```bash
  python src/abstractive_summarizer.py --text "Your long text here..."
  ```
- To run the extractive summarizer:
  ```bash
  python src/extractive_summarizer.py --text "Your long text here..."
  ```

## Models
This project supports various pre-trained models. Refer to the `src/` directory for specific model implementations and configurations.

## Contributing
Feel free to contribute by opening issues or submitting pull requests.

## License
This project is licensed under the MIT License.
