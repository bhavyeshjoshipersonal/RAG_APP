from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.document_loaders import OnlinePDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
import ollama
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.callbacks.manager import CallbackManagerForRetrieverRun

doc_path = r"C:\Rag_App\data\tsla-20241023-gen.pdf"
#doc_path = r"C:\Rag_App\data\BOI.pdf"
#doc_path = r"C:\Rag_App\data\HD_2024_AR_IRsite_v2.pdf"

model = "llama3.2"
# Ingest pdf file - load the file content into a list of Document objects
if doc_path:
    loader = UnstructuredPDFLoader(file_path = doc_path)
    data = loader.load()
    print("done loading....")
else:
    print("upload a pdf file...")

print(f"Number of elements in data: {len(data)}")
content = data[0].page_content
total_chars = sum(len(doc.page_content) for doc in data)
print(f"Total number of characters in data: {total_chars}")

# ...existing code...

# Extract text from the PDF and split it into chunks of text
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(data)
print("done splitting....")
print(f"Number of chunks: {len(chunks)}")

# ...existing code...

# Create embeddings for the chunks
result = ollama.pull("nomic-embed-text")
print(f"Model pulled: {result}")

#add embeddings to Chroma vector store

# Create embeddings for the chunks using OllamaEmbeddings
embedding_function = OllamaEmbeddings(model="nomic-embed-text")

# Add embeddings and documents to Chroma vector store
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_function,
    persist_directory="chroma_db"  # Optional: specify a directory to persist the DB
)

print("Embeddings added to Chroma vector store.")

#Retrieve documents based on a query
llm = ChatOllama(model=model)

QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is to generate five\ndifferent versions of the given user question to retrieve relevant documents from\na vector database. By generating multiple perspectives on the user question, your\ngoal is to help the user overcome some of the limitations of the distance-based\nsimilarity search. Provide these alternative questions separated by newlines.\nOriginal question: {question}""",
)

retriever = MultiQueryRetriever.from_llm(
    vectorstore.as_retriever(),
    llm,
    prompt=QUERY_PROMPT)

# RAG prompt
template = """Answer the question based ONLY on the following context:\n{context}\nQuestion: {question}\n"""

prompt = ChatPromptTemplate.from_template(template)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Print the generated query variations
print("Generated query variations:")

# Get and print the generated query variations
user_question = "what is the document about?"
# Construct a proper callback manager
run_manager = CallbackManagerForRetrieverRun.get_noop_manager()
variations = retriever.generate_queries(user_question, run_manager=run_manager)
for i, q in enumerate(variations):
    print(f"Variation {i+1}: {q}")

res = chain.invoke({"question": user_question})
print(f"Response: {res}")
