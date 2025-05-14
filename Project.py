import streamlit as st
import requests
import os
import base64
from io import BytesIO
from PyPDF2 import PdfReader
from pdf2image import convert_from_bytes
import pytesseract
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# Streamlit UI
st.set_page_config(page_title="AI-Powered Financial Document Analysis", layout="wide", page_icon="✨")


# Function to encode image into base64
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Provide the correct image path
image_path = "Logoooooooooooo.jpg"  # Make sure this matches your actual file name

# Encode image to Base64
base64_image = get_base64_image(image_path)

st.markdown(f"""
    <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{base64_image}");
            background-size: cover
            background-position: center;
            background-attachment: fixed;
        }}

        /* Make input, textarea, and select boxes transparent with a thick black border */
        div[data-baseweb="input"] > div,
        div[data-baseweb="textarea"] > div,
        div[data-baseweb="select"] > div {{
            background: rgba(255, 255, 255, 0) !important; /* Fully Transparent */
            border: 3px solid black !important;  /* Single Thick Black Border */
            border-radius: 10px;
            padding: 5px;
        }}

        /* Increase font size inside input fields */
        input, textarea {{
            font-size: 20px !important;  /* Increase input text size */
            background: transparent !important;
            color: black !important;
            font-weight: bold !important;
        }}

        /* Increase font size of labels */
        .stTextInput label, .stTextArea label {{
            font-size: 24px !important;  /* Increase label size */
            font-weight: bold !important; /* Make it bold */
            color: black !important; /* Ensure it's clearly visible */
        }}

        /* Improve button visibility */
        .stButton>button {{
            background-color: rgba(255, 255, 255, 0); /* Slight transparency */
            border-radius: 10px;
            border: 3px solid black !important; /* Single thick black border */
            color: black;
            font-size: 18px !important; /* Increase button text size */
        }}

        .stButton>button:hover {{
            background-color: rgba(255, 255, 255, 0.3);
        }}

    </style>
""", unsafe_allow_html=True)


# st.title("AI-Powered Financial Document Analysis")
st.header("📊 Unlock Insights from Annual Financial Reports with AI 🚀")
st.subheader("📈Transform complex financial reports into meaningful insights using advanced LLMs effortlessly!")
# Input fields with placeholders
api_key = st.text_input("", placeholder="Enter Your Google Gemini API Key", type="password")
document_url = st.text_input("", placeholder="Enter Document URL")
user_query = st.text_area("", placeholder="Enter Your Query")


# Model selection dropdown
model_choice = st.selectbox(
    "Select Model",
    ("gemini-1.5-flash", "gemini-pro"),
    index=0  # Default selection is gemini-1.5-flash
)


# Function to fetch document
def fetch_document(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return BytesIO(response.content)
    except requests.exceptions.RequestException as e:
        return f"Error fetching document: {e}"


# Function to extract text from PDF
def extract_text_from_pdf(pdf_bytes):
    reader = PdfReader(pdf_bytes)
    return "".join([page.extract_text() or "" for page in reader.pages])


# Function to extract text from images using OCR
def extract_text_from_images(images):
    return "\n".join([pytesseract.image_to_string(image) for image in images])


# Function to process document
def process_document(pdf_bytes):
    pdf_bytes.seek(0)
    pdf_text = extract_text_from_pdf(pdf_bytes)

    try:
        images = convert_from_bytes(pdf_bytes.getvalue())
        ocr_text = extract_text_from_images(images)
    except Exception:
        ocr_text = ""

    return pdf_text + "\n" + ocr_text


# Function to split text into chunks
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return splitter.split_text(text)


# Function to store vectors
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store


# Function to load LLM model based on user selection
def get_conversational_chain(model_name):
    prompt_template = PromptTemplate(
        template="""
        You are a financial expert with extensive knowledge in analyzing financial statements, market trends, and economic principles. ONLY respond to finance-related questions. If the question is not related to finance domain,
        politely refuse to answer.
        Context:\n{context}\n
        Question:\n{question}\n
        Response:
        """,
        input_variables=["context", "question"]
    )
    model = ChatGoogleGenerativeAI(model=model_name, temperature=0.3)
    return load_qa_chain(model, chain_type="stuff", prompt=prompt_template)


# Button action
if st.button("Get Response"):
    if not api_key or not document_url or not user_query:
        st.error("Please provide all required inputs.")
    else:
        os.environ["GOOGLE_API_KEY"] = api_key  # Set API key
        pdf_bytes = fetch_document(document_url)

        if isinstance(pdf_bytes, str):
            st.error(pdf_bytes)
        else:
            document_text = process_document(pdf_bytes)
            text_chunks = get_text_chunks(document_text)
            vector_store = get_vector_store(text_chunks)
            retriever = vector_store.as_retriever()
            docs = retriever.get_relevant_documents(user_query)
            qa_chain = get_conversational_chain(model_choice)
            response = qa_chain.run(input_documents=docs, question=user_query)
            st.subheader("Response:")
            st.write(response)