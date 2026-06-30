# RAG Learning Lab

A comprehensive, hands-on Python project implementing **all 14 RAG types** and **9 advanced techniques** as standalone, runnable modules for learning.

## Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **API Key Setup**
   Copy `.env.example` to `.env` and add your Google Gemini API key:
   ```bash
   cp .env.example .env
   # Edit .env and insert your key
   ```

3. **Generate Sample Data**
   Run the setup script to generate the synthetic AI/ML documents used by all demos:
   ```bash
   python setup_data.py
   ```

## Running the Demos

The easiest way to explore the project is using the interactive runner:

```bash
python run_demo.py
```

This will open a menu where you can select any of the 14 RAG architectures or 9 optimization techniques to run and observe. Each script is heavily commented to explain the *why* and *how* behind the code.
