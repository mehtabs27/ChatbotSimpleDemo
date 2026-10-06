# PDF Chatbot

## Run locally

1. Install dependencies with `pip install -r requirements.txt`.
2. Set `GEMINI_API_KEY` in a local `.env` file.
3. Start the app with `streamlit run app.py`.
4. Upload a PDF in the app and ask questions about it.

## Deploy on Streamlit Community Cloud

1. Push this project to a GitHub repository. Keep `.env` out of the repository.
2. Create a Streamlit Community Cloud app for that repository and set its entrypoint to `app.py`.
3. In the app's **Settings > Secrets**, add:

   ```toml
   GEMINI_API_KEY = "your-gemini-api-key"
   ```

4. Deploy the app. Upload the PDF through the running app.
