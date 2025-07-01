from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma
import ollama
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.callbacks.manager import CallbackManagerForRetrieverRun

class PDFRAGPipeline:
    def __init__(self, doc_path, model="llama3.2", embedding_model="nomic-embed-text", persist_directory="chroma_db"):
        self.doc_path = doc_path
        self.model = model
        self.embedding_model = embedding_model
        self.persist_directory = persist_directory
        self.data = None
        self.chunks = None
        self.vectorstore = None
        self.llm = None
        self.retriever = None
        self.chain = None

    def load_pdf(self):
        loader = UnstructuredPDFLoader(file_path=self.doc_path)
        self.data = loader.load()
        print("done loading....")
        print(f"Number of elements in data: {len(self.data)}")
        total_chars = sum(len(doc.page_content) for doc in self.data)
        print(f"Total number of characters in data: {total_chars}")

    def split_text(self, chunk_size=1000, chunk_overlap=200):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.chunks = text_splitter.split_documents(self.data)
        print("done splitting....")
        print(f"Number of chunks: {len(self.chunks)}")

    def create_embeddings(self):
        result = ollama.pull(self.embedding_model)
        print(f"Model pulled: {result}")
        embedding_function = OllamaEmbeddings(model=self.embedding_model)
        self.vectorstore = Chroma.from_documents(
            documents=self.chunks,
            embedding=embedding_function,
            persist_directory=self.persist_directory
        )
        print("Embeddings added to Chroma vector store.")

    def setup_retriever(self):
        self.llm = ChatOllama(model=self.model)
        QUERY_PROMPT = PromptTemplate(
            input_variables=["question"],
            template="""You are an AI language model assistant. Your task is to generate five\ndifferent versions of the given user question to retrieve relevant documents from\na vector database. By generating multiple perspectives on the user question, your\ngoal is to help the user overcome some of the limitations of the distance-based\nsimilarity search. Provide these alternative questions separated by newlines.\nOriginal question: {question}""",
        )
        self.retriever = MultiQueryRetriever.from_llm(
            self.vectorstore.as_retriever(),
            self.llm,
            prompt=QUERY_PROMPT)

    def setup_chain(self):
        template = """Answer the question based ONLY on the following context:\n{context}\nQuestion: {question}\n"""
        prompt = ChatPromptTemplate.from_template(template)
        self.chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

    def get_query_variations(self, user_question):
        run_manager = CallbackManagerForRetrieverRun.get_noop_manager()
        variations = self.retriever.generate_queries(user_question, run_manager=run_manager)
        print("Generated query variations:")
        for i, q in enumerate(variations):
            print(f"Variation {i+1}: {q}")
        return variations

    def ask(self, user_question):
        res = self.chain.invoke({"question": user_question})
        print(f"Response: {res}")
        return res


def main():
    doc_path = r"C:\Rag_App\data\tsla-20241023-gen.pdf"
    #user_question = "who has signed this document on behalf of tesla and what is that person's position?"
    #user_question = "What is the total revenue for Tesla in 2023?"
    user_question = "What is the address, phone number and trading symbol of Tesla?"

    pipeline = PDFRAGPipeline(doc_path)
    pipeline.load_pdf()
    pipeline.split_text()
    pipeline.create_embeddings()
    pipeline.setup_retriever()
    pipeline.setup_chain()
    pipeline.get_query_variations(user_question)
    pipeline.ask(user_question)

if __name__ == "__main__":
    main()
