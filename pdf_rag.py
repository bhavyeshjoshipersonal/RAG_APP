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

doc_path = r"C:\Rag_App\data\tsla-20241023-gen.pdf"
#doc_path = r"C:\Rag_App\BOI.pdf"

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

#print(content[:100])  # Print the first 100 characters of the content

# Extract text from the PDF and split it into chunks of text
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(data)
print("done splitting....")
print(f"Number of chunks: {len(chunks)}")
#print(f"Example chunk: {chunks[4]}")

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
    template="""You are an AI language model assistant. Your task is to generate five
    different versions of the given user question to retrieve relevant documents from
    a vector database. By generating multiple perspectives on the user question, your
    goal is to help the user overcome some of the limitations of the distance-based
    similarity search. Provide these alternative questions separated by newlines.
    Original question: {question}""",
)

retriever = MultiQueryRetriever.from_llm(
    vectorstore.as_retriever(),
    llm,
    prompt=QUERY_PROMPT)

# RAG prompt
template = """Answer the question based ONLY on the following context:
{context}
Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

res = chain.invoke(input=("what is the document about?",))
print(f"Response: {res}")