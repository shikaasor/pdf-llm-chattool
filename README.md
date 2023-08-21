# Chat with Your PDFs

This is a Streamlit application that allows you to chat with your PDF documents using conversational AI models. You can upload your PDF files and ask questions about the content of the documents, and the system will provide relevant responses based on the trained models.

## Getting Started

To run the application, follow these steps:

1. Clone the repository to your local machine.
2. Install the required dependencies by running `pip install -r requirements.txt`.
3. Make sure you have the required PDF files ready for uploading.

## Usage

1. Run the application by executing `streamlit run app.py` in your terminal.
2. The application will open in your browser and display an interface for uploading PDF files.
3. Click on the "Upload your PDF files here" button and select one or more PDF files from your local machine.
4. Once the files are uploaded, click on the "Process" button to start processing the PDF content.
5. You can now enter your questions in the text input field and click "Enter" to receive responses from the system.
6. The system will display a conversation history, alternating between your questions and the system's responses.

## Dependencies

The application relies on the following libraries:

- streamlit
- dotenv
- PyPDF2
- langchain
- htmlTemplates

You can find the complete list of dependencies in the `requirements.txt` file.

## Acknowledgments

This application utilizes several open-source libraries and technologies. Here are the main components:

- Streamlit: A Python library for building interactive web applications.
- PyPDF2: A library for reading and manipulating PDF files.
- langchain: A library for text processing, embeddings, and chat models.
- htmlTemplates: HTML templates for displaying chat messages in the Streamlit app.

## License

This project is licensed under the [MIT License](LICENSE).
