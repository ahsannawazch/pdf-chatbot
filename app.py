import os
import chainlit as cl
from langchain.schema.runnable.config import RunnableConfig
from chainlit.types import AskFileResponse
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter         # Implement  Semantic Chinking   2. llamaindex document knowledge graph
#from langchain_openai import OpenAIEmbeddings
#from langchain_pinecone import PineconeVectorStore
#from langchain_openai import ChatOpenAI
from langchain_cohere import ChatCohere, CohereEmbeddings, CohereRagRetriever
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import create_history_aware_retriever,create_retrieval_chain
from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

from dotenv import load_dotenv
load_dotenv()

# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# Loading PDF
def file_loader(file: AskFileResponse):
    loader = PyPDFLoader(file.path)
    pages = loader.load_and_split()
    return pages

# Splitting the docs
def doc_splitter(pages):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=70)   # paly with em
    chunks = splitter.split_documents(pages)

    for i, doc in enumerate(chunks):
        doc.metadata["source"] = f"source_{i}"
    
    return chunks

# Storing Embeddings 
def store_embeddings(chunks):
    embeddings = CohereEmbeddings()
    vectorstore = FAISS.from_documents(chunks,embeddings)
    return vectorstore



# If data is already in pinecone don't add more/repetitive stuff.    check later
# How to clear an index and add new data in it.
# How to append data in same index?
# Should I add multiple books in the same index?


# Model 
model = ChatCohere(cohere_api_key= COHERE_API_KEY)



@cl.on_chat_start
async def on_start_chat():
    elements = [
        cl.Image(name="image1",display="inline",path="llama.jpg")
    ]
    await cl.Message(content="Hello, How can I be of your assistance?", elements=elements).send()

    files = None

    # Wait for the user to upload a file
    while files is None:
        files = await cl.AskFileMessage(
            content="Please upload a PDF file to begin!\n"
            "The processing of the file may require a few moments or minutes to complete.",
            accept=["text/plain", "application/pdf"],
            max_size_mb=100,
            timeout=180,
        ).send()

    file = files[0]

    msg = cl.Message(content=f"Processing `{file.name}`...", disable_feedback=True)
    await msg.send()

    # Process the file and return pages
    pages = file_loader(file)

    # Split pages into chunks
    chunks = doc_splitter(pages)

    # Store Embeddings
    vectordb = store_embeddings(chunks)

    # Set vectorstore as retriever
    retriever = vectordb.as_retriever()                           # Play with top k and return source docs. later

    msg.content = f"Creating embeddings for `{file.name}`. . ."
    await msg.update()
















    #model = ChatOpenAI(model= "gpt-3.5-turbo")
    
   
    contextualize_query_system_message = """ Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""
    contextualize_query_prompt = ChatPromptTemplate.from_messages(
        [
        ("system", contextualize_query_system_message),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
        ]
    )
    history_aware_retriever = create_history_aware_retriever(model, retriever, contextualize_query_prompt)


    qa_system_message = """You are an assistant for question-answering tasks. \
    Use the following pieces of retrieved context to answer the question. \
    If you don't know the answer, just say that you don't know. \
    Use three sentences maximum and keep the answer concise.\

    {context}"""
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_message),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm=model, prompt=qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever,question_answer_chain)

     # Statefully tracking history
    store = {}

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key= "input",
        history_messages_key="chat_history",
        output_messages_key="answer",
        )

    cl.user_session.set("conversational_rag_chain",conversational_rag_chain)               #Might need to change quoted conversational_rag_chain to chain
    
    msg.content = f"`{file.name}` processed. You can now ask questions!"
    await msg.update()
   
    



 
##########################

@cl.on_message
async def on_message(message: cl.Message):

    conversational_rag_chain = cl.user_session.get("conversational_rag_chain")    

    #msg = cl.Message(content="")
    
#     conversational_rag_chain.invoke(
#     {"input": "Who is Ibn e Khaldoon?"},
#     config={
#         "configurable": {"session_id": "abc123"}
#     },  # constructs a key "abc123" in `store`.
# )["answer"]

    response =  await conversational_rag_chain.ainvoke(
        {"input": message.content},
        config={"configurable": {"session_id": "abc123"},
                "callbacks":[cl.AsyncLangchainCallbackHandler()]},         
    )
    answer = response["answer"]

    source_documents = response["context"]
    text_elements = []
    unique_pages = set()

    if source_documents:

        for source_idx, source_doc in enumerate(source_documents):
            source_name = f"source_{source_idx+1}"
            page_number = source_doc.metadata['page']
            #page_number = source_doc.metadata.get('page', "NA")  # NA or any default value
            page = f"Page {page_number}"
            text_element_content = source_doc.page_content
            #text_elements.append(cl.Text(content=text_element_content, name=source_name))
            if page not in unique_pages:
                unique_pages.add(page)
                text_elements.append(cl.Text(content=text_element_content, name=page))
            #text_elements.append(cl.Text(content=text_element_content, name=page))
        source_names = [text_el.name for text_el in text_elements]
        
        if source_names:
            answer += f"\n\n Sources:{', '.join(source_names)}"
        else:
            answer += "\n\n No sources found"

    await cl.Message(content=answer, elements=text_elements).send()


# ploomber below

    # if source_documents:
    #     for source_idx, source_doc in enumerate(source_documents):
    #         source_name = f"source {source_idx+1}"
    #         # Create the text element referenced in the message
    #         text_elements.append(
    #             cl.Text(content=source_doc.page_content, name=source_name)
    #         )
    #     source_names = [text_el.name for text_el in text_elements]

    #     if source_names:
    #         answer += f"\nSources: {', '.join(source_names)}"
    #     else:
    #         answer += "\nNo sources found"

    # await cl.Message(content=answer, elements=text_elements).send()




    