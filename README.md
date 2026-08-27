# Aequitas XAI: Credit Decision & Algorithmic Fairness Explainer

An interactive Explainable AI (XAI) dashboard built with Streamlit and Plotly to demonstrate how high-stakes credit decisions are made by machine learning models, how systemic human bias is absorbed, and how SHAP (Shapley Additive exPlanations) values can peel open the "black box" to explain predictions and verify fairness.

This project was built to accompany the essay **"When the Machine Gets It Wrong: Bias and the Social Implications of Artificial Intelligence"** for the student course project. It serves as both a practical demonstration (working model) and a comprehensive report.

---

## 🚀 How to Run the Project Locally

To run this application on your local machine:

1.  **Open a Terminal** and navigate to this project folder.
2.  **Create a Virtual Environment**:
    ```bash
    python3 -m venv .venv
    ```
3.  **Activate the Virtual Environment**:
    *   On macOS/Linux:
        ```bash
        source .venv/bin/activate
        ```
    *   On Windows (Command Prompt):
        ```cmd
        .venv\Scripts\activate.bat
        ```
    *   On Windows (PowerShell):
        ```powershell
        .venv\Scripts\Activate.ps1
        ```
4.  **Install the Required Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
5.  **Run the Streamlit Dashboard**:
    ```bash
    streamlit run app.py
    ```
6.  The application will automatically open in your default web browser at `http://localhost:8501`.

---

## 📁 Project Directory Structure
```
xai-visualizer/
│
├── app.py              # Main interactive Streamlit application & SHAP math engine
├── requirements.txt    # Required Python libraries (pandas, numpy, scikit-learn, plotly, streamlit)
├── project_report.md   # Markdown copy of the academic course project report
└── README.md           # This instructions and hosting guide file
```

---

## 🌐 How to Host the Project Online (For Submission Links)

The project instructions require you to **"Attach the link of your working project."** Below are the two simplest ways to host this Streamlit project online for free:

### Method A: Hosting via Streamlit Community Cloud (Recommended)
Streamlit offers free hosting for apps directly connected to GitHub:
1.  Push this folder (`xai-visualizer`) to a new public repository on [GitHub](https://github.com).
2.  Go to [Streamlit Share](https://share.streamlit.io/) and log in with your GitHub account.
3.  Click **New app**, select your repository, branch (`main`), and set the main file path to `app.py`.
4.  Click **Deploy!** Your app will be live at a custom URL (e.g., `https://your-app-name.streamlit.app`) in under 2 minutes.
5.  Submit this URL as your project link!

### Method B: Hosting via Hugging Face Spaces
Hugging Face offers free hosting for Streamlit spaces:
1.  Log in to [Hugging Face](https://huggingface.co/) (create a free account if you don't have one).
2.  Go to **Spaces** -> **Create new Space**.
3.  Set the name of your Space, select **Streamlit** as the SDK, and choose the public/free hardware tier.
4.  Upload the project files (`app.py`, `requirements.txt`, `project_report.md`, `README.md`) directly using the web interface or via Git.
5.  The Space will automatically build and run your Streamlit app, generating a shareable link.
