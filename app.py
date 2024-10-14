import os
import random
import streamlit as st
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI


from dotenv import load_dotenv
load_dotenv()

# Function to load PDFs and return raw text
def load_pdfs_from_backend(pdf_files):
    raw_text = ''

    # Randomly select a few PDFs from the list
    selected_pdfs = random.sample(pdf_files, min(len(pdf_files), 3))  # Select 3 random PDFs

    for file in selected_pdfs:
        try:
            pdfreader = PdfReader(file)
            for page in pdfreader.pages:
                content = page.extract_text()
                if content:
                    raw_text += content
        except Exception as e:
            st.error(f"Error reading {file}: {e}")

    return raw_text

# Function to split text into chunks
def split_text(raw_text, chunk_overlap=200):
    total_length = len(raw_text)
    chunk_size = max(min(total_length // 100, 2000), 500)

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    
    return text_splitter.split_text(raw_text)

# Function to create document embeddings using OpenAI embeddings
def create_document_embeddings(texts):
    embeddings = OpenAIEmbeddings()
    return FAISS.from_texts(texts, embeddings)

# Function to load a QA chain and run a query
def run_query(document_search, query):
    llm = OpenAI()
    chain = load_qa_chain(llm, chain_type="stuff")
    docs = document_search.similarity_search(query)
    return chain.run(input_documents=docs, question=query)

# Streamlit UI
def main():
    st.title("PDF get respnoses")

    # Predefined backend PDF file paths (assumed these are large PDF files)
    pdf_files = [
        'pdf/Adam Park Condominium - Service Agreement.pdf',
        'pdf/Copy of Track Record (2).pdf',
        'pdf/Landscape_tender_requirements.pdf',
        'pdf/List of Proj Ref.pdf',
        'pdf/New Quotation SH20157 - Adam Park Condo.pdf',
        'pdf/Bid Comparison for Tender#f66b42e3-cbb4-4672-a91c-432de6b15bc7.pdf',
        'pdf/Cleaning Method Statement.pdf',
        'pdf/Cleaning Schedule.pdf',
        'pdf/JASA Security Technology Equipment Proposal.pdf',
        'pdf/MCST 4253 - The Tampines Trilliant (Quotation).pdf',
        'pdf/NW-A2220-290621-TT (cleaning svs).pdf',
        'pdf/Proposal Quotation - The Tampines Trilliant - EP077.pdf',
        'pdf/RAS BizSafe Level 3 until 26 Nov 2023.pdf',
        'pdf/RAS Client Feedback 2021.pdf',
        'pdf/RAS Current Projects as at 16 Sep 2021.pdf',
        'pdf/Sands Company Profile_19May21_HiRes.pdf',
        'pdf/Sands Global License & Insurance.pdf',
        'pdf/SGQ21-041 (Adam Park Condominium).pdf',
        # 'pdf/',
        # 'pdf/.pdf',
        # 'pdf/.pdf',
        # 'Some_other_document.pdf'
        # Add more paths as needed
    ]

    # Load random PDFs and extract raw text
    raw_text = load_pdfs_from_backend(pdf_files)

    if raw_text:
        # st.success("PDF files processed successfully")

        # Split text into smaller chunks
        texts = split_text(raw_text)

        # Generate document embeddings
        document_search = create_document_embeddings(texts)

        # Get user query
        query = st.text_input("Enter your query")

        if query and st.button("Run Query"):
            result = run_query(document_search, query)
            st.write("Query Result:")
            st.write(result)
    else:
        st.error("Failed to load text from backend PDFs.")

# Run the Streamlit app
if __name__ == "__main__":
    main()
