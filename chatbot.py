import shutil
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain.vectorstores.chroma import Chroma
import time
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_community.llms import OpenAI
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import re
import tabula
import camelot
import pdfplumber

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
working_dir = os.getcwd()
chroma_path = os.path.join(working_dir, 'chroma')
uploaded_pdfs_file = os.path.join(working_dir, 'uploaded_pdfs.txt')
conversation_history_dir = os.path.join(working_dir, 'conversation_history')
openai_key = os.getenv("ChatGPT")

pdf_content = ""  # Global variable to store PDF content
feedback_data = []  # Global variable to store user feedback data


def extract_pdf_text(pdf):
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def extract_structured_data(pdf):
    structured_data = ""
    # Extract tables using tabula
    try:
        tables = tabula.read_pdf(pdf, pages="all")
        for table in tables:
            structured_data += table.to_string(index=False) + "\n"
    except Exception as e:
        st.warning(f"Tabula error: {e}")

    # Extract tables using camelot
    try:
        tables = camelot.read_pdf(pdf, pages="all")
        for table in tables:
            structured_data += table.df.to_string(index=False) + "\n"
    except Exception as e:
        st.warning(f"Camelot error: {e}")

    # Extract tables using pdfplumber
    try:
        with pdfplumber.open(pdf) as pdf:
            for page in pdf.pages:
                for table in page.extract_tables():
                    for row in table:
                        structured_data += "\t".join([str(cell) for cell in row]) + "\n"
    except Exception as e:
        st.warning(f"pdfplumber error: {e}")

    return structured_data


def extract_multicolumn_text(pdf):
    multicolumn_text = ""
    with pdfplumber.open(pdf) as pdf:
        for page in pdf.pages:
            columns = page.extract_words(x_tolerance=3)
            columns_text = " ".join([word['text'] for word in columns])
            multicolumn_text += columns_text + "\n"
    return multicolumn_text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                                   chunk_overlap=200,
                                                   length_function=len,
                                                   add_start_index=True)
    chunks = text_splitter.split_text(text)
    return chunks


def save_to_chroma(text_chunks):
    if os.path.exists(chroma_path):
        try:
            shutil.rmtree(chroma_path)
        except OSError as e:
            st.warning(f"Error deleting 'chroma' directory: {e}")

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = Chroma.from_texts(text_chunks, embedding=embeddings, persist_directory=chroma_path)
    db.persist()


def get_conversational_chain():
    prompt_template = """
    Answer the question as thoroughly as possible using only the information provided in the context. Ensure that all relevant details are included in your response. 
    If the answer is not available within the provided context, state, "Answer is not available in the context." Do not provide an incorrect or incomplete answer.\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    llm = OpenAI(temperature=0, openai_api_key=openai_key)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)

    return chain


def load_uploaded_pdfs():
    uploaded_pdfs = []
    if os.path.exists(uploaded_pdfs_file):
        with open(uploaded_pdfs_file, 'r') as file:
            uploaded_pdfs = file.read().splitlines()
    return uploaded_pdfs


def save_uploaded_pdfs(uploaded_pdfs):
    with open(uploaded_pdfs_file, 'w') as file:
        for pdf in uploaded_pdfs:
            file.write(pdf + '\n')


def sanitize_session_name(session_name):
    sanitized_name = re.sub(r'[^\w-]', '', session_name)
    return sanitized_name


def save_conversation_session(session_name, conversation_session):
    session_name = sanitize_session_name(session_name)
    session_dir = os.path.join(conversation_history_dir, session_name)
    os.makedirs(session_dir, exist_ok=True)
    session_file = os.path.join(session_dir, f"{session_name}.txt")
    with open(session_file, 'w') as file:
        for turn in conversation_session:
            file.write(turn + '\n')


def delete_conversation_session(session_name):
    session_name = sanitize_session_name(session_name)
    session_dir = os.path.join(conversation_history_dir, session_name)
    if os.path.exists(session_dir):
        shutil.rmtree(session_dir)
        st.sidebar.success(f"Session '{session_name}' deleted successfully.")
    else:
        st.sidebar.warning(f"Session '{session_name}' does not exist.")


def rename_conversation_session(old_name, new_name):
    old_name = sanitize_session_name(old_name)
    new_name = sanitize_session_name(new_name)
    old_dir = os.path.join(conversation_history_dir, old_name)
    new_dir = os.path.join(conversation_history_dir, new_name)
    if os.path.exists(old_dir):
        os.rename(old_dir, new_dir)
        st.sidebar.success(f"Session '{old_name}' renamed to '{new_name}' successfully.")
    else:
        st.sidebar.warning(f"Session '{old_name}' does not exist.")


def user_input(user_question, conversation_history):
    global pdf_content
    global feedback_data
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        new_db = Chroma(persist_directory=chroma_path, embedding_function=embeddings)
        docs = new_db.similarity_search(user_question)
        chain = get_conversational_chain()
        response = chain(
            {"input_documents": docs, "question": user_question}
            , return_only_outputs=True)
        user_turn = f"{datetime.now()} - User: {user_question}"
        bot_turn = f"{datetime.now()} - Bot: {response['output_text']}"
        conversation_history.append(user_turn)
        conversation_history.append(bot_turn)

        if feedback_data:
            st.write("Please provide feedback on the response:")
            feedback_text = st.text_area("Feedback:")
            if st.button("Submit Feedback"):
                feedback_data.append((user_question, response["output_text"], feedback_text))
                st.write("Thank you for your feedback!")
                st.experimental_rerun()

        return conversation_history, response["output_text"], [user_turn, bot_turn]
    except Exception as e:
        st.error(f"Error processing your request: {e}")
        return conversation_history, "An error occurred. Please try again.", []


def process_pdf(pdf):
    global pdf_content
    try:
        text = extract_pdf_text(pdf)
        structured_data = extract_structured_data(pdf)
        multicolumn_text = extract_multicolumn_text(pdf)
        full_content = text + "\n" + structured_data + "\n" + multicolumn_text
        text_chunks = get_text_chunks(full_content)
        save_to_chroma(text_chunks)
        pdf_content = full_content
        return pdf.name
    except Exception as e:
        st.error(f"Error processing PDF {pdf.name}: {e}")
        return None


def show_warning(message, duration=5):
    warning_placeholder = st.empty()
    warning_placeholder.warning(message)
    time.sleep(duration)
    warning_placeholder.empty()


def show_success(message, duration=5):
    success_placeholder = st.empty()
    success_placeholder.success(message)
    time.sleep(duration)
    success_placeholder.empty()


def continuous_learning_and_improvement(feedback_data):
    if feedback_data:
        # Placeholder for analyzing feedback and updating model
        for entry in feedback_data:
            # Process feedback data and update model
            pass
        print("Model updated based on user feedback.")
    else:
        print("No user feedback available.")


def main():
    st.set_page_config("Chat with multiple PDF")
    st.title("Chat with PDF")

    conversation_history = []
    session_name = None
    session_history = []

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        if not session_name:
            session_name = user_question[:50]
        conversation_history, bot_response, chat_session = user_input(user_question, conversation_history)
        session_history.extend(chat_session)
        response_height = min(300, max(100, len(bot_response) // 2))
        st.text_area("Bot:", value=bot_response, height=response_height)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.sidebar.file_uploader("Upload your PDF Files", accept_multiple_files=True)
        if pdf_docs:
            if st.sidebar.button("Process"):
                with st.spinner("Processing..."):
                    uploaded_pdfs = load_uploaded_pdfs()
                    with ThreadPoolExecutor(max_workers=4) as executor:
                        processed_files = list(executor.map(process_pdf, pdf_docs))
                    for pdf_name in processed_files:
                        if pdf_name in uploaded_pdfs:
                            show_warning(f"{pdf_name} already uploaded!")
                            continue
                        uploaded_pdfs.append(pdf_name)
                    save_uploaded_pdfs(uploaded_pdfs)
                    show_success("Done")

    if session_name and st.sidebar.button("Save Conversation"):
        save_conversation_session(session_name, session_history)
        show_success(f"Conversation saved as session: {session_name}")

    continuous_learning_and_improvement(feedback_data)


if __name__ == "__main__":
    main()
