Chat with PDF using Gemini

This Streamlit application allows you to ask questions and have a conversation based on the information extracted from uploaded PDF documents. It leverages the power of Google Cloud AI's Gemini-Pro model for natural language processing and retrieval.

How it Works:

Upload PDF Files: Select and upload one or more PDF documents you want to process.
Ask a Question: Once processing is complete, type your question in the text box.
Get Answers: The application searches the extracted text from the PDFs and uses Gemini-Pro to answer your question in a comprehensive way.

Key Features:

Contextual Information Retrieval: Extracts relevant pieces of text from uploaded PDFs based on your question.
Natural Language Processing: Uses Gemini-Pro to understand your questions and generate informative answers.
User-Friendly Interface: Streamlit provides a straightforward interface for easy interaction.
Requirements:

Python libraries: streamlit, PyPDF2, langchain, google.generativeai, dotenv
Google Cloud Project with API keys for:
Google GenerativeAI (instructions on https://cloud.google.com/)

Setup Instructions:

Clone the Repository:
git clone https://github.com/shikaasor/pdf-llm-chattool


Install Dependencies:

pip install -r requirements.txt

Configure Environment Variables:
Create a .env file in the project root.
Add the following lines, replacing placeholders with your actual API keys:
GOOGLE_API_KEY=<your_google_api_key>

Run the App:

streamlit run app.py
