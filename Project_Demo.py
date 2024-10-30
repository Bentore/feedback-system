import streamlit as st
from langchain_experimental.open_clip import OpenCLIPEmbeddings
from langchain_community.document_transformers import (LongContextReorder, EmbeddingsRedundantFilter)
from langchain_chroma import Chroma
from langchain.retrievers import (ContextualCompressionRetriever, MergerRetriever)
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
import google.generativeai as genai
import openai
from langchain_core.documents import Document
import os
from dotenv import load_dotenv, find_dotenv
import hashlib
import time

# Initialize Gemini
def init_gemini():
    genai.configure(api_key=google_api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    return model

# Initialize CLIP embeddings separately and cache it.
@st.cache_resource(show_spinner=False)
def get_embeddings_model():
    return OpenCLIPEmbeddings(weights_only=True)

# Function to generate an overall summary of image summaries
def generate_file_summary(file_path, file_name, model):
    """Generate summary for a single file using Gemini"""
    sample_file = genai.upload_file(path=file_path, display_name=file_name)
    try:
        response = model.generate_content(
            contents=[summary_prompt,sample_file],
            generation_config={"temperature": 0.0}
        )
        time.sleep(10)
        response.resolve()
        return response.text
    except Exception as e:
        st.error(f"Error processing file {file_path}: {str(e)}")
        return None

# Cache document embeddings
@st.cache_resource(show_spinner=False)
def process_and_embed_assignment(pdf):
    global embeddings
    # Initialize Gemini model
    model = init_gemini()
    # Store processed file hashes to avoid duplicates
    processed_hashes = set()
    pdf_summaries=[]

    temp_pdf_path = f"./{pdf.name}"
    with open(temp_pdf_path, mode='wb') as w:
        w.write(pdf.getvalue())

    try:
        # Save uploaded file temporarily
        with open(temp_pdf_path, mode='wb') as w:
            w.write(pdf.getvalue())

        #Extract images from the PDF
        pdf_name = os.path.splitext(os.path.basename(temp_pdf_path))[0]
        pdf_hash = hashlib.md5(pdf.getvalue()).hexdigest()

        # Check if the document already exists based on its hash
        if pdf_hash in processed_hashes:
            st.warning(f"Document {pdf_name} has already been processed. Skipping summarization.")
            return None

        # Add hash to the set of processed hashes
        processed_hashes.add(pdf_hash)

        # Generate summary for the file
        summary = generate_file_summary(temp_pdf_path, pdf_name, model)

        if summary:
            document = Document(
                page_content=summary,
                metadata={
                    "name": pdf.name,
                    "hash": pdf_hash,
                    "type": "file_summary"
                }
            )
            pdf_summaries.append(document)

    except Exception as e:
        st.error(f"Error processing file {pdf.name}: {str(e)}")
        return None

    finally:
        # Clean up temporary file
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)

    # Check if there are valid file summaries before creating the retriever
    if pdf_summaries:
        try:
            bm25_retriever = BM25Retriever.from_documents(pdf_summaries, int=5)
        except ValueError as e:
            st.error(f"BM25 Retriever Error: {e}")
            return None, None

        # Extract the texts and metadata for adding to the vector database.
        texts = [doc.page_content for doc in pdf_summaries]
        metadatas = [doc.metadata for doc in pdf_summaries]

        # Set up the persistence directory and ChromaDB client
        summary_collection = "mpproject_summary"

        vectordb_a = Chroma.from_texts(
            texts=texts,
            embedding=embeddings,
            collection_name=summary_collection,
            collection_metadata={"hnsw:space": "cosine"},
            metadatas=metadatas,  # Include the metadata when creating the collection.
            create_collection_if_not_exists=True
        )

        return vectordb_a, bm25_retriever
    else:
        st.error("No valid summaries were generated for the assignments.")
        return None, None

# Sidebar contents
with st.sidebar:
    st.sidebar.success("For Tutors Uploads ↑")
    st.title('LLM Feedback-based App')
    st.markdown('''
                This app is an LLM-powered chatbot built using
                - [Steamlit](https://streamlit.io/)
                - [LangChain](https://python.langchain.com/)
                - [ChatGPT/OpenAI Model](https://platform.openai.com/docs/models)
                - [Google Generative AI (Gemini) Model](https://ai.google/discover/generativeai/)
                - [ChromaDB](https://docs.trychroma.com/)
                ''')
    st.divider()
    st.title('How do I use this as a Student?')
    st.markdown('''Here are the steps to do so
    \n1. Upload a pdf, docx, or txt file📄 
    \n2. Ask a question about the document💬
    \n3. Obtain feedback, suggestions and the intended grade. 
    ''')
    st.divider()
    st.title('Things to take note:')
    st.markdown('''
    \n1. Indexing and Processing of your file will take a while.
    \n2. Replies may not be completely accurate or sufficient.
    ''')
    st.divider()
    st.write('Made by Bentley Teng (2201144I)')

def main(): 
    # Welcoming the user
    st.header("Obtain Feedback about your assignment Today!")

    # Upload a PDF file
    pdf = st.file_uploader("Start by Uploading your Assignment", type='pdf')

    if pdf:
        # Show a spinner while processing the PDF
        with st.spinner("Processing your PDF... Please wait. This process will take roughly 5 minutes"):
            # Process the PDF and cache the document embeddings
            vectordb, bm25_retriever = process_and_embed_assignment(pdf)
            # Create ASSIGNMENT retrieval (Merger + Redundant Filter + Reordering)
            basic_retriever_sim = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 10})
            basic_retriever_word = bm25_retriever

            # The Lord of the Retrievers will hold the output of both retrievers and can be used as any other
            # retriever on different types of chains.
            lotr = MergerRetriever(retrievers=[basic_retriever_sim , basic_retriever_word])
            # Redundant filter will remove embeddings that are not needed/irrelevant
            redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)
            reordering = LongContextReorder()
            # Intialize the Reordering algorithm/function (For universal use)
            universal_pipeline = DocumentCompressorPipeline(transformers=[redundant_filter, reordering])
            assignment_retriever = ContextualCompressionRetriever(
                base_compressor=universal_pipeline, base_retriever=lotr
            )

        with st.spinner("Pulling Rubrics from the other page. Please Hold!"):
            if 'rubric_document' in st.session_state:
                try: 
                    # Use the cached rubric document
                    vectordb_r = st.session_state['rubric_document']
                    st.error("Uh oh. Your Tutor has yet to upload the module rubric. Please Wait!")
                except:
                    st.stop()

        # Create rubric retriever
        rubric_retriever = ContextualCompressionRetriever(base_compressor=universal_pipeline, base_retriever=vectordb_r.as_retriever(search_type="similarity", search_kwargs={"k": 10}))


        st.write(vectordb._collection.count())
        st.write(vectordb_r._collection.count())
        submit_button = st.button(label="Begin Feedback")
        
        # Proceed only when the user clicks the submit button
        if submit_button:
            with st.spinner("Generating Feedback... Please wait. This process will around 5 minutes or less"):

                # invoke all retrievers 
                assignment = assignment_retriever.invoke(QA_query)
                rubric = rubric_retriever.invoke(QA_query)
                assignments = [Document(page_content=assg.page_content, metadata=assg.metadata) for assg in assignment]
                st.write(assignments)
                st.write(rubric)

                # context = "\n".join([rub.page_content for rub in rubrics])
                student_submission = "\n".join([assg.page_content for assg in assignments])
                context = "\n".join([rub.page_content for rub in rubric])

                # Create the final prompt
                final_prompt = f"""
                {system_prompt}
                
                Benchmarks and Rubrics:
                {context}
                
                Student Submission:
                {student_submission}
                
                Please provide feedback based on the above benchmarks and rubrics.
                """

                response = openai.chat.completions.create(
                    model="gpt-4o-mini",
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": final_prompt},
                    ],
                    n=1,
                    temperature=0
                )

                # Extract and print the assistant's reply
                assistant_reply = response.choices[0].message.content
                st.write("Feedback for Student:")
                st.write(assistant_reply)
                st.write(response.usage)

if __name__ == '__main__':
    embeddings = get_embeddings_model()
    _ = load_dotenv(find_dotenv())  # Load API key
    openai.api_key = os.getenv("OPENAI_API_KEY")
    google_api_key= os.getenv("GOOGLE_API_KEY")
    summary_prompt = os.getenv("SUMMARY_PROMPT")
    system_prompt = os.getenv("SYSTEM_PROMPT")
    QA_query = os.getenv("QNA_QUERY")
    main()
