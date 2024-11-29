import os
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain_groq import ChatGroq  # Replace with your LLM configuration

import re
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')
# Summarization function

def summarize_content(content, llm2, max_length=2500):
    summarization_prompt = f"""
    Summarize the following content into a concise summary, limited to {max_length} characters:
    {content}
    """
    response = llm2.invoke(summarization_prompt)
    return response.content.strip()


# Create vector database
def create_vector_database(pdf_path, embedding_model, persist_directory):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    os.makedirs(persist_directory, exist_ok=True)
    vector_db = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        collection_name=os.path.basename(pdf_path).split(".")[0],
        persist_directory=persist_directory
    )
    return vector_db


# Compare policies
def compare_policies(file1, file2, query, embedding_model, llm,llm2, persist_dir1, persist_dir2, max_summary_length=2500):
    vector_db_1 = create_vector_database(file1, embedding_model, persist_dir1)
    vector_db_2 = create_vector_database(file2, embedding_model, persist_dir2)

    query_embedding = embedding_model.embed_query(query)
    result_query_1 = vector_db_1.similarity_search_by_vector(query_embedding, k=5)
    result_query_2 = vector_db_2.similarity_search_by_vector(query_embedding, k=5)

    all_page_contents_1 = " ".join([i.page_content for i in result_query_1])
    all_page_contents_2 = " ".join([i.page_content for i in result_query_2])

    summarized_content_1 = summarize_content(all_page_contents_1, llm2, max_length=max_summary_length)
    summarized_content_2 = summarize_content(all_page_contents_2, llm2, max_length=max_summary_length)

    comparison_prompt = f"""
    You are an AI assistant for question-answering tasks. Your role is to provide accurate, relevant, and concise answers based on the given context and chat history. Follow these guidelines:

    1. Use the provided context (delimited by triple backticks) to answer the question.
    2. Consider the chat history for additional context, but prioritize the given context for your answer.
    3. Determine if the answer can be generated based on the provided context.
    4. If you determine answer isn't in the context, say "I don't have enough information to answer that question."
    5. Avoid making up information or using external knowledge not provided in the context.
    6. If the question is unclear, ask for clarification instead of making assumptions.
    Based on the content provided, compare the two policies on the basis of the user's question "{query}".
    Context1 is derived from {os.path.basename(file1)}, and Context2 from {os.path.basename(file2)}.
    Which policy is better and why? Remember, accuracy and relevance are key. Base your response solely on the provided information.

    Context1: {summarized_content_1}
    Context2: {summarized_content_2}
    """

    response = llm.invoke(comparison_prompt)
    return response.content


def generate_questions(pdf_path, embedding_model, persist_dir, llm, chunk_size=1000, overlap_size=100):
    """
    Generate 10 questions from the PDF content using embeddings and LLM, while handling large context size.
    """
    # Create a vector database for the PDF
    vector_db = create_vector_database(pdf_path, embedding_model, persist_dir)

    # Retrieve content from the vector database
    docs = vector_db.similarity_search_by_vector([0] * 384, k=10)  # Dummy vector for initial content
    text_content = " ".join([doc.page_content for doc in docs])

    # Split content into smaller chunks if it's too large
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap_size
    )
    chunks = text_splitter.split_text(text_content)

    # Summarize the chunks to reduce context size further
    summarized_chunks = []
    for chunk in chunks[:10]:
        summarized_chunk = summarize_content(chunk, llm, max_length=2500)  # Summarize each chunk
        summarized_chunks.append(summarized_chunk)

    # Combine the summarized content into a final context
    final_context = " ".join(summarized_chunks)

    # Use LLM to generate questions
    question_generation_prompt = f"""
                    You are tasked with generating **exactly 8 clear, factual, and relevant questions** based on the provided context. 
                    The questions should:
                    1. Be directly answerable using only the information in the given context.
                    2. Focus on key details, facts, policies, terms, or concepts mentioned in the content.
                    3. Avoid hypothetical, open-ended, or ambiguous questions.
                    4. Be concise, specific, and relevant to the subject matter.
                    5. Cover a variety of points within the context to ensure comprehensive coverage.

                    **Context:**
                    {final_context}

                    Please return a numbered list of 8 distinct questions following the above criteria.
                    """

    response = llm.invoke(question_generation_prompt)
    questions = re.findall(r'\d+\.\s*(.*)', response.content)

    return questions


# Streamlit Web App
def main():
    # Page layout
    st.set_page_config(layout="wide")

    # Create columns for layout
    left_col, center_col, right_col = st.columns([1, 2, 1.2])

    # Left column: File uploads
    with left_col:
        st.header("Policy Uploads")
        uploaded_file1 = st.file_uploader("Upload the first PDF file", type="pdf", key="file1")
        uploaded_file2 = st.file_uploader("Upload the second PDF file (optional)", type="pdf", key="file2")

    # Center column: Chat-like interface
    with center_col:
        st.title("Policy Comparison with RAG")
        st.write("Enter your query below to compare the policies based on the uploaded files.")

        # Query input
        user_query = st.text_input("Enter your query (e.g., 'What is the suicide exclusion policy?')")

        # Process when files and query are provided
        if st.button("Compare Policies") and uploaded_file1 and user_query:
            with st.spinner("Processing... Please wait."):
                # Save uploaded files locally
                pdf_path1 = f"uploaded_file1.pdf"
                pdf_path2 = f"uploaded_file2.pdf" if uploaded_file2 else None

                with open(pdf_path1, "wb") as f:
                    f.write(uploaded_file1.read())
                if uploaded_file2:
                    with open(pdf_path2, "wb") as f:
                        f.write(uploaded_file2.read())

                # Placeholder example for response


            # Setup embedding model
            embedding_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cuda'}
            )

            # Setup LLM (replace with your LLM configuration)
            llm = ChatGroq(
                model="llama3-70b-8192",
                #model='llama-3.1-8b-instant',
                #model='llama3-groq-70b-8192-tool-use-preview',
                temperature=0.1,
                groq_api_key=groq_api_key
            )
            llm2 = ChatGroq(
                #model="llama3-70b-8192",
                model='llama-3.1-8b-instant',
                # model='llama3-groq-70b-8192-tool-use-preview',
                temperature=0.1,
                groq_api_key=groq_api_key
            )

            # Persist directories
            persist_dir1 = "vector_db_1"
            persist_dir2 = "vector_db_2" if pdf_path2 else None

            try:
                # Compare policies
                if pdf_path2:
                    comparison_response = compare_policies(
                        pdf_path1, pdf_path2, user_query, embedding_model, llm,llm2, persist_dir1, persist_dir2
                    )
                else:
                    # Single file mode: Retrieve and summarize only
                    vector_db = create_vector_database(pdf_path1, embedding_model, persist_dir1)
                    query_embedding = embedding_model.embed_query(user_query)
                    result_query = vector_db.similarity_search_by_vector(query_embedding, k=6)
                    all_page_contents = " ".join([i.page_content for i in result_query])
                    summarized_content = summarize_content(all_page_contents, llm, max_length=2500)
                    comparison_response = f"Summary of the policy:\n{summarized_content}"

                # Display the result
                st.success("Comparison Complete!")
                st.write("### Response:")
                st.write(comparison_response)
            except Exception as e:
                st.error(f"An error occurred: {e}")

    with right_col:
        st.header("Auto-Generated Questions")


        # Initialize session state for questions
        if "auto_questions" not in st.session_state:
            st.session_state.auto_questions = []

        # Generate questions when files are uploaded
        if uploaded_file1 and not st.session_state.auto_questions:
            with st.spinner("Generating questions from the uploaded files..."):
                pdf_path1 = f"uploaded_file1.pdf"
                persist_dir1 = "vector_db_1"
                embedding_model = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    model_kwargs={'device': 'cpu'}
                )

                with open(pdf_path1, "wb") as f:
                    f.write(uploaded_file1.read())

                llm = ChatGroq(
                    #model="llama3-70b-8192",
                    model='llama-3.1-8b-instant',
                    # model='llama3-groq-70b-8192-tool-use-preview',
                    temperature=0.1,
                    groq_api_key=groq_api_key
                )
                st.session_state.auto_questions = generate_questions(pdf_path1, embedding_model, persist_dir1,llm)

        # Display static questions
        if st.session_state.auto_questions:
            for idx, question in enumerate(st.session_state.auto_questions, start=1):
                st.write(f"{idx}. {question}")
        else:
            st.info("Upload a PDF to see auto-generated questions.")


if __name__ == "__main__":
    main()
