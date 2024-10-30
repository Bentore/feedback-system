import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.open_clip import OpenCLIPEmbeddings
from langchain_community.document_transformers import (LongContextReorder, EmbeddingsRedundantFilter)
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma.vectorstores import Chroma
from langchain.retrievers import (ContextualCompressionRetriever, MergerRetriever)
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain_core.documents import Document
from dotenv import (load_dotenv, find_dotenv)
import google.generativeai as genai
import openai
import hashlib
import time
import os

def clear_chroma_collections():
    """Clear all collections in the Chroma database"""
    embeddings = get_embeddings_model
    try:
        # Get the Chroma client and delete collections
        collection_names = [
            "mpproject_summary_assignment",
            "mpproject_summary_benchmarks",
            "mpproject_summary_rubrics"
        ]
        
        for collection_name in collection_names:
            try:
                # Create a temporary Chroma instance to delete the collection
                temp_db = Chroma(
                    collection_name=collection_name,
                    embedding_function=embeddings
                )
                # Delete the collection
                temp_db.delete_collection()
                # Reset the client
                temp_db = None
                st.success(f"{collection_name} collection cleared successfully")
            except Exception as e:
                st.warning(f"Could not clear {collection_name}: {str(e)}")
                continue
                
    except Exception as e:
        st.error(f"Error clearing Chroma collections: {str(e)}")

def clear_session_state():
    # Clear the session state for the current page
    if 'rubric_document' in st.session_state:
        del st.session_state['rubric_document']
    # Clear the Chroma collections
    clear_chroma_collections()
    st.success("Session state cleared successfully")

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

@st.cache_resource(show_spinner=False)
def process_and_embed_assignments(assignment_batches):
    # pass global embeddings variable
    global embeddings
    # Initialize an empty list to collect summaries across all assignments
    file_summaries = []
    # Initialize Gemini model
    model = init_gemini()

    # Store processed file hashes to avoid duplicates
    processed_hashes = set()

    for file in assignment_batches:
        file_summaries = []
        temp_file_path = f"./{file.name}"

        try:
            # Save uploaded file temporarily
            with open(temp_file_path, mode='wb') as w:
                w.write(file.getvalue())

            #Extract images from the PDF
            file_name = os.path.splitext(os.path.basename(temp_file_path))[0]
            file_hash = hashlib.md5(file.getvalue()).hexdigest()

            # Check if the document already exists based on its hash
            if file_hash in processed_hashes:
                st.warning(f"Document {file_name} has already been processed. Skipping summarization.")
                continue

            # Add hash to the set of processed hashes
            processed_hashes.add(file_hash)

            # Generate summary for the file
            summary = generate_file_summary(temp_file_path, file_name, model)

            if summary:
                document = Document(
                    page_content=summary,
                    metadata={
                        "name": file.name,
                        "hash": file_hash,
                        "type": "file_summary"
                    }
                )
                file_summaries.append(document)

        except Exception as e:
            st.error(f"Error processing file {file.name}: {str(e)}")
            continue

        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    # Check if there are valid file summaries before creating the retriever
    if file_summaries:
        try:
            bm25_retriever = BM25Retriever.from_documents(file_summaries, int=5)
        except ValueError as e:
            st.error(f"BM25 Retriever Error: {e}")
            return None, None

        # Extract the texts and metadata for adding to the vector database.
        texts = [doc.page_content for doc in file_summaries]
        metadatas = [doc.metadata for doc in file_summaries]
        # Create collection name
        assignment_collection = "mpproject_summary_assignment"

        vectordb_a = Chroma.from_texts(
            texts=texts,
            embedding=embeddings,
            collection_name=assignment_collection,
            collection_metadata={"hnsw:space": "cosine"},
            metadatas=metadatas,  # Include the metadata when creating the collection.
            create_collection_if_not_exists=True
        )

        return vectordb_a, bm25_retriever
    else:
        st.error("No valid summaries were generated for the assignments.")
        return None, None

@st.cache_resource(show_spinner=False)
def process_and_embed_benchmark(benchmarks):
    #Pass global embeddings variable
    global embeddings
    # Initialize an empty list to collect summaries across all benchmarks
    benchmark_summaries = []
    # Initialize Gemini model
    model = init_gemini()
    # Store processed file hashes to avoid duplicates
    #processed_hashes = set()

    for benchmark in benchmarks:
        temp_bench_path = f"./{benchmark.name}"

        try:
            # Save uploaded file temporarily
            with open(temp_bench_path, mode='wb') as w:
                w.write(benchmark.getvalue())

            #Extract images from the PDF
            benchmark_name = os.path.splitext(os.path.basename(temp_bench_path))[0]
            benchmark_hash = hashlib.md5(benchmark.getvalue()).hexdigest()

            # Generate summary for the file
            summary = generate_file_summary(temp_bench_path, benchmark_name, model)

            if summary:
                document = Document(
                    page_content=summary,
                    metadata={
                        "name": benchmark_name,
                        "hash": benchmark_hash,
                        "type": "file_summary"
                    }
                )
                benchmark_summaries.append(document)

        except Exception as e:
            st.error(f"Error processing file {benchmark_name}: {str(e)}")
            continue

        finally:
            # Clean up temporary file
            if os.path.exists(temp_bench_path):
                os.remove(temp_bench_path)

    if benchmark_summaries:
        # Extract the texts and metadata for adding to the vector database.
        texts = [bench.page_content for bench in benchmark_summaries]
        metadatas = [bench.metadata for bench in benchmark_summaries]
        # Create collection name
        benchmark_collection = "mpproject_summary_benchmarks"

        vectordb_b = Chroma.from_texts(
            texts=texts,
            embedding=embeddings,
            collection_name=benchmark_collection,
            collection_metadata={"hnsw:space": "cosine"},
            metadatas=metadatas,  # Include the metadata when creating the collection.
            create_collection_if_not_exists=True
        )

        return vectordb_b
    else:
        st.error("No valid summaries were generated for the benchmarks.")
        return None

@st.cache_resource()
def process_and_embed_rubric(rubrics):
    global embeddings
    # Check if the rubric file is already in session state
    if 'rubric_document' in st.session_state:
        return st.session_state['rubric_document']
    # Initialize a set to keep track of processed rubric hashes
    processed_hashes = set()
    # Calculate the hash of the uploaded rubric
    temp_rubrics_path = f"./{rubrics.name}"
    rubric_hash = hashlib.md5(rubrics.getvalue()).hexdigest()

    # Check if this rubric has already been processed
    if rubric_hash in processed_hashes:
        st.warning(f"Rubric {rubrics.name} has already been processed. Skipping summarization.")
        return None
    
    processed_hashes.add(rubric_hash)

    # define splitter first
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000, 
        chunk_overlap=1000, 
        is_separator_regex=False,
    )

    with open(temp_rubrics_path, mode='wb') as w:
        w.write(rubrics.getvalue())

    loader = PyPDFLoader(file_path=temp_rubrics_path, extract_images=False)
    documents = loader.load_and_split(text_splitter=splitter)

    # Clean up temporary file
    if os.path.exists(temp_rubrics_path):
        os.remove(temp_rubrics_path)

    # Set up the persistence directory and ChromaDB client
    rubric_collection = "mpproject_summary_rubrics"

    vectordb_r = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=rubric_collection,
        collection_metadata={"hnsw:space": "cosine"},
        create_collection_if_not_exists=True
    )

    # Store the vectordb_r in session state
    st.session_state['rubric_document'] = vectordb_r

    return vectordb_r

# side bar contents
with st.sidebar: 
    st.title('About this Section')
    st.markdown('''
    This app allows you to upload multiple documents
    - Student Assignment/s (multiple files permitted)
    - Module Rubrics       
    - Benchmark Samples (multiple files permitted)
    ''')
    st.write('---')
    st.title('How do I interact with this Feedback System?')
    st.markdown('''Here are the steps to do so
    \n1. Upload all documents as PDF Files.
    \n2. Click "begin feedback" to start grading the assignment/s.
    \n3. Obtain feedback, suggestions and the intended grades for them!
    ''')
    st.write('---')
    st.title('Things to take note:')
    st.markdown('''
    \n1. Indexing and Processing of your file will take a while.
    \n2. Replies Or Grade might be inaccurate on some occasions (Refer to grading range for assistance)
    ''')
    st.write('---')
    st.write('''Grading Ranges for reference:
    \n- A: 80% & Above
    \n- B+: 75-79%
    \n- B: 70-74%
    \n- C+: 65-69%
    \n- C: 60-64%
    \n- D+: 55-59%
    \n- D: 50-54%
    \n- F: 49 nd below
    ''')
    st.write('---')
    clear_button = st.button("Clear Session State")
    if clear_button:
        clear_session_state()
        st.success("Session state cleared.")
    st.write('Made by Bentley Teng (2201144I)')

def main2():

    #Welcoming the users
    st.header("Grade assignments by uploading them as PDFs!")

    #Upload the student assignments in bulk
    assignment_batches = st.file_uploader("Upload Assignments in Bulk here: ", type='pdf', accept_multiple_files=True)
    #Upload the benchmark samples for cross-comparison
    benchmarks = st.file_uploader("Upload in Benchmarks here: ", type='pdf', accept_multiple_files=True)
    #Upload marking rubrics
    rubrics = st.file_uploader("Upload module rubrics here: ", type='pdf')

    if assignment_batches and benchmarks and rubrics:

        with st.spinner("Processing each assignment... Please wait. This process will take roughly a few minutes"):
            # Process assignments
            vectordb_a, bm25_retriever = process_and_embed_assignments(assignment_batches)
            # Create ASSIGNMENT retrieval (Merger + Redundant Filter + Reordering)
            basic_retriever_sim = vectordb_a.as_retriever(search_type="similarity", search_kwargs={"k": 10})
            basic_retriever_word = bm25_retriever
            # The Lord of the Retrievers will hold the output of both retrievers and can be used as any other
            # retriever on different types of chains.
            lotr = MergerRetriever(retrievers=[basic_retriever_sim , basic_retriever_word])
            # Redundant filter will remove embeddings that are not needed/irrelevant
            redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)
            reordering = LongContextReorder()
            # Intialize the Reordering algorithm/function (For universal use)
            universal_pipeline = DocumentCompressorPipeline(transformers=[redundant_filter, reordering])
            assignment_retriever = ContextualCompressionRetriever(base_compressor=universal_pipeline, base_retriever=lotr)

        with st.spinner("Processing each benchmark... Please wait. This process will take roughly a few minutes"):
            # Process benchmarks
            vectordb_b = process_and_embed_benchmark(benchmarks)
            # Create benchmark retrieval (redundant and relevant embedding filter + reranker)
            benchmark_retriever = ContextualCompressionRetriever(base_compressor=universal_pipeline, 
                                                                base_retriever=vectordb_b.as_retriever(search_type="similarity", search_kwargs={"k": 10}))            

        if 'rubric_document' in st.session_state:
            # Use the cached rubric document
            vectordb_r = st.session_state['rubric_document']
        else: 
            with st.spinner("Processing the module rubric... Please wait. This process will take roughly a few minutes"):
                # Process rubric
                vectordb_r = process_and_embed_rubric(rubrics)

        # Create rubric retriever
        rubric_retriever = ContextualCompressionRetriever(base_compressor=universal_pipeline, 
                                                        base_retriever=vectordb_r.as_retriever(search_type="similarity", search_kwargs={"k": 10}))

        st.write(vectordb_a._collection.count())
        st.write(vectordb_b._collection.count())
        st.write(vectordb_r._collection.count())

        submit_button = st.button(label="Begin Feedback")

        if submit_button:
            with st.spinner("Generating Feedback... Please wait. This process will take roughly a few minutes"):
                # invoke all retrievers 
                assignment = assignment_retriever.invoke(QA_query)
                benchmark = benchmark_retriever.invoke(QA_query)
                rubric = rubric_retriever.invoke(QA_query)

                #Organize into list of documents with appropiate/necessary content
                assignments = [Document(page_content=assg.page_content, metadata=assg.metadata) for assg in assignment]
                rubrics = [Document(page_content=rub.page_content, metadata=rub.metadata) for rub in rubric]
                benchmarks = [Document(page_content=ben.page_content, metadata=ben.metadata) for ben in benchmark]

                st.write(assignments)
                st.write('---')
                st.write(benchmarks)
                st.write('---')
                st.write(rubrics)

                # Combine the content of benchmarks and rubrics for context
                benchmark_context = "\n".join([bench.page_content for bench in benchmarks])
                rubric_context = "\n".join([rub.page_content for rub in rubrics])

                # Compile context for benchmarks and rubrics together
                context = f"{benchmark_context}\n{rubric_context}"
                student_submission = "\n".join([assg.page_content for assg in assignments])

                # Create the final prompt
                final_prompt = f"""
                {system_prompt}
                
                Benchmarks and Rubrics:
                {context}
                
                Student Submission/s:
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

if __name__ == "__main__":
    embeddings = get_embeddings_model()
    _ = load_dotenv(find_dotenv())  # Load API key
    openai.api_key = os.getenv("OPENAI_API_KEY")
    google_api_key= os.getenv("GOOGLE_API_KEY")
    summary_prompt = os.getenv("SUMMARY_PROMPT")
    system_prompt = os.getenv("SYSTEM_PROMPT")
    QA_query = os.getenv("QNA_QUERY")  
    main2()