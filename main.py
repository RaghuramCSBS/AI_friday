import streamlit as st
from pdfminer.high_level import extract_text
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import tempfile
import os
import httpx
import tiktoken

# Token cache setup
tiktoken_cache_dir = "./token"
os.environ["TIKTOKEN_CACHE_DIR"] = tiktoken_cache_dir

# Disable SSL verification warnings
client = httpx.Client(verify=False)

# LLM and Embedding setustp
llm = ChatOpenAI(
    base_url="https://genailab.tcs.in",
    model="azure_ai/genailab-maas-DeepSeek-V3-0324",
    api_key="sk-IKiDxFyU5AJ8jQ8d7--yrA",
    http_client=client
)

embedding_model = OpenAIEmbeddings(
    base_url="https://genailab.tcs.in",
    model="azure/genailab-maas-text-embedding-3-large",
    api_key="sk-IKiDxFyU5AJ8jQ8d7--yrA",
    http_client=client
)

# Streamlit UI
st.set_page_config(page_title="RAG PDF Summarizer")
st.title("RAG-powered PDF Summarizer")

upload_file = st.file_uploader("Upload a PDF", type="pdf")

if upload_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(upload_file.read())
        temp_file_path = temp_file.name

    # Step 1: Extract text
    raw_text = extract_text(temp_file_path)

    # Step 2: Chunking
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(raw_text)

    # Step 3: Embed and store in Chroma
    with st.spinner("Indexing document..."):
        vectordb = Chroma.from_texts(
            chunks,
            embedding_model,
            persist_directory="./chroma_index"
        )
        vectordb.persist()

    # Step 4: Build RAG chain using LCEL
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    prompt = ChatPromptTemplate.from_template(
        "Use the following context to answer the question:\n\n{context}\n\nQuestion: {question}"
    )

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Step 5: Ask summarization prompt
    summary_prompt = "Please summarize this document based on the key topics:"

    with st.spinner("Running RAG summarization..."):
        result = rag_chain.invoke(summary_prompt)

    # Display summary
    st.subheader("Summary")
    st.write(result)
