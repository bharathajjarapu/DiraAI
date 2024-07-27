# Dira AI : AI Legal Assistant for India

Welcome to **Dira AI** – your AI-powered legal advisor specializing in Indian laws! ⚖️ 
Lawy currently leverages Google's advanced AI models to provide precise and reliable legal advice. 
Lawy can answer all your questions about legal procedures, rights, or regulations in India.

**AI can still hallucinate so, use it as Google not a Lawyer**

## Packages Used 💻

- **Python** 🐍
- **PyPDF2** 📄
- **LangChain** 🔗
- **Google Generative AI** 🌐
- **FAISS** 🔍
- **Streamlit** 🌟

## Features ✨

- 💬 **Conversational AI:** Get answers to your legal questions through a natural, conversational interface.
- 🌐 **Streamlit Interface:** Interactive web app for seamless user experience.

## How It Works 🔍

1. **Ingest Data:**
   - Extract text from PDF files located in the `dataset` folder.
   - Split the text into chunks for better processing.
   - Create a FAISS vector store with embeddings for efficient searches.

2. **Ask Questions:**
   - Type your legal questions in the chat interface.
   - Lawy uses the vector store to find relevant documents.
   - Get accurate and context-aware legal advice based on Indian laws.

## Setup 🛠️

1. **Clone the repository:**
   ```bash
   git clone https://github.com/bharathajjarapu/diraai.git
   cd lawy
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   - Create a `.env` file in the project root.
   - Add your Google API key:
     ```
     GOOGLE_API_KEY=your_api_key
     ```

4. **Run the application:**
   ```bash
   python run ingest.py
   streamlit run lawy.py
   ```

## Streamlit Deploy 💡

1. **Start the app:**
   ```bash
   streamlit run app.py
   ```

2. **Interact with Lawy:**
   - Type your legal questions in the chat input.
   - Lawy will provide answers based on the context and Indian laws.

Made with 💡& ❤️ by [Bharath](https://github.com/bharathajjarapu)
