# Policy Comparison using RAG Approach

This project implements a Retrieval-Augmented Generation (RAG) approach to compare two insurance policies. It uses ChromaDB as the vector database and GROQ to call a Large Language Model (LLM) for summarization, question generation, and policy comparison.

---

## Features

- **Policy Comparison:** Compare two insurance policies based on user queries.
- **Automatic Question Generation:** Generate questions based on the uploaded policy documents.
- **PDF Content Handling:** Extract, embed, and summarize content from PDF documents.

---

## Technology Stack

- **ChromaDB:** Vector database for storing and querying embedded documents.
- **LangChain:** Framework for integrating LLMs with external tools.
- **GROQ API:** Call an LLM to perform tasks such as summarization and comparison.
- **Streamlit:** Build an interactive web-based user interface.
- **HuggingFace Embeddings:** Generate vector embeddings for text similarity tasks.

---

## Prerequisites

Ensure you have the following installed:

1. **Python**: Version 3.8 or higher.
2. **pip**: Python's package installer.
3. **GROQ API Key**: Sign up and obtain an API key from [GROQ](https://www.groq.com).

---

## Installation

### Step 1: Clone the Repository

```bash
git clone <repository_url>
cd <repository_name>Policy Comparison using RAG Approach
This project implements a Retrieval-Augmented Generation (RAG) approach to compare two insurance policies. It uses ChromaDB as the vector database and GROQ to call a Large Language Model (LLM) for summarization, question generation, and policy comparison.


```

### Step 2: Set Up a Virtual Environment

```bash
python -m venv env
source env/bin/activate   # On Windows: env\Scripts\activate
```

### Step 3: Install Dependencies

Install required Python libraries from `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Step 4: Set Up Environment Variables

Create a `.env` file in the root directory and add your GROQ API key:

```env
GROQ_API_KEY=your_api_key_here
```

---

## Usage

### Step 1: Start the Streamlit App

Run the following command to launch the web application:

```bash
streamlit run app.py
```

### Step 2: Upload Policy Documents

- Use the left column to upload two insurance policy PDFs.

### Step 3: Enter a Query

- In the central column, input a query (e.g., "What is the suicide exclusion policy?").

### Step 4: Compare Policies

- Click the "Compare Policies" button to generate a comparison based on your query.

### Step 5: Auto-Generated Questions

- View auto-generated questions in the right column.

---

## Directory Structure

```plaintext
.
├── app.py                     # Main Streamlit application
├── requirements.txt           # Python dependencies
├── .env                       # Environment variables
├── vector_db_1/               # Persisted vector database for policy 1
├── vector_db_2/               # Persisted vector database for policy 2
└── README.md                  # Project documentation
```

---

### UI
![Screenshot from 2024-11-29 16-34-50](https://github.com/user-attachments/assets/a577d6ba-63c3-4b0c-a8b2-376ecd108dda)



---

## Key Functions

### Summarize Content

Summarize extracted content using an LLM:

```python
def summarize_content(content, llm, max_length=2500):
    summarization_prompt = f"""
    Summarize the following content into a concise summary, limited to {max_length} characters:
    {content}
    """
    response = llm.invoke(summarization_prompt)
    return response.content.strip()
```

### Compare Policies

Compare two policy documents based on user queries:

```python
def compare_policies(file1, file2, query, embedding_model, llm, persist_dir1, persist_dir2):
    # Vector database and content summarization logic...
    comparison_prompt = f"""
    Based on the content provided, compare the two policies on the basis of the user's question "{query}".
    """
    response = llm.invoke(comparison_prompt)
    return response.content
```

---

## Notes

- Ensure the `HuggingFaceEmbeddings` model and GROQ API are properly configured for your system.
- The application can run on CPU or GPU. Modify `model_kwargs` accordingly.

---

## Future Enhancements

- Add support for additional document formats (e.g., Word, HTML).
- Enable storage and retrieval of previous comparisons.
- Improve UI/UX for a better user experience.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

For more information or troubleshooting, feel free to reach out!


