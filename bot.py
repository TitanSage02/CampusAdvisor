from socket import timeout
import streamlit as st

#Importations
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np

import os, tempfile, glob, random
from pathlib import Path
from getpass import getpass

from itertools import combinations
from langchain.memory import ConversationSummaryBufferMemory,ConversationBufferMemory # type: ignore

# LLM: Google_genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# LLM: HuggingFace
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.llms import HuggingFaceHub

# langchain prompts, memory, chains...
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import ConversationalRetrievalChain

from langchain_community.chat_message_histories import StreamlitChatMessageHistory

from operator import itemgetter
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain.schema import Document, format_document
from langchain_core.messages import AIMessage, HumanMessage, get_buffer_string

# Document loaders
from langchain_community.document_loaders import (
    TextLoader,
)

# Text Splitter
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter

# OutputParser
from langchain_core.output_parsers import StrOutputParser

# Chroma: vectorstore
from langchain_community.vectorstores import Chroma

# Contextual Compression
from langchain.retrievers.document_compressors import DocumentCompressorPipeline # type: ignore
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_transformers import EmbeddingsRedundantFilter,LongContextReorder
from langchain.retrievers.document_compressors import EmbeddingsFilter # type: ignore
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_loaders import TextLoader



# Envrionnments variables loading
from dotenv import load_dotenv
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

# Compilation of data
directory = "data"
def compile_txt_files_to_string(directory):
    compiled_content = []
    
    # Parcourir tous les fichiers dans le dossier donné
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)

            # Ouvrir et lire le contenu du fichier txt
            with open(filepath, "r", encoding="utf-8") as file:
                compiled_content.append(file.read())

    # Combiner tous les contenus en une seule chaîne
    return "\n\n\".join(compiled_content)

filepath = None
with open(f"{directory}/data.txt", "w", encoding="utf-8") as file:
    document = compile_txt_files_to_string(directory)
    content = file.write(document)

loader = TextLoader(f"{directory}/data.txt")
documents = loader.load()

# Create a RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    separators = ["\n\n", "\n\n\n", " "],
    chunk_size = 400,
    chunk_overlap= 80
)

# Text splitting
chunks = text_splitter.split_documents(documents=documents)
print(f"Number of chunks: {len(chunks)}")

def select_embeddings_model(LLM_service="Google"):
    """Connect to the embeddings API endpoint by specifying the name of the embedding model."""
    # if LLM_service == "OpenAI":
    #     embeddings = OpenAIEmbeddings(
    #         model='text-embedding-ada-002',
    #         api_key=openai_api_key)

    if LLM_service == "Google":
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=google_api_key,
            timeout=500 # type: ignore
        ) 
        
    if LLM_service == "HuggingFace":
        embeddings = HuggingFaceInferenceAPIEmbeddings(
            api_key="",
            model_name="sentence-transformers/all-MiniLM-L12-v2"
        )

    return embeddings

embedding = select_embeddings_model(LLM_service="HuggingFace")
#embeddings_HuggingFace = select_embeddings_model(LLM_service="HuggingFace")

def create_vectorstore(embeddings, documents, vectorstore_name):
    """Create a Chroma vector database."""
    LOCAL_VECTOR_STORE_DIR = Path("vector_store")
    persist_directory = (LOCAL_VECTOR_STORE_DIR.as_posix() + "/" + vectorstore_name)
    
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embedding,
        persist_directory=persist_directory
    )
    
    return vector_store


vector_store_google = create_vectorstore(
        embeddings=embedding,
        documents = chunks,
        vectorstore_name="Lead__"
    )

#print("vector_store_google: ",vector_store_google._collection.count()," chunks.")

def Vectorstore_backed_retriever(vectorstore,search_type="similarity", k=17, score_threshold=None):
    """create a vectorsore-backed retriever
    Parameters:
        search_type: Defines the type of search that the Retriever should perform.
            Can be "similarity" (default), "mmr", or "similarity_score_threshold"
        k: number of documents to return (Default: 5)
        score_threshold: Minimum relevance threshold for similarity_score_threshold (default=None)
    """
    search_kwargs={}

    if k is not None:
        search_kwargs['k'] = k
    
    if score_threshold is not None:
        search_kwargs['score_threshold'] = score_threshold

    retriever = vectorstore.as_retriever(
        search_type=search_type,
        search_kwargs=search_kwargs
    )
    return retriever


base_retriever_google = Vectorstore_backed_retriever(vector_store_google, "similarity", k=25)

def instantiate_LLM(LLM_provider="HuggingFace",api_key="", temperature=0.5, top_p=0.95, model_name=None):
    """Instantiate LLM in Langchain.
    Parameters:
        LLM_provider (str): the LLM provider; in ["OpenAI","Google","HuggingFace"]
        model_name (str): in ["gpt-3.5-turbo", "gpt-3.5-turbo-0125", "gpt-4-turbo-preview",
            "gemini-pro", "mistralai/Mistral-7B-Instruct-v0.2"].
        api_key (str): google_api_key or openai_api_key or huggingfacehub_api_token
        temperature (float): Range: 0.0 - 1.0; default = 0.5
        top_p (float): : Range: 0.0 - 1.0; default = 1.
    """
    # if LLM_provider == "OpenAI":
    #     llm = ChatOpenAI(
    #         api_key=api_key,
    #         model=model_name,
    #         temperature=temperature,
    #         model_kwargs={
    #             "top_p": top_p
    #         }
    #     )

    if LLM_provider == "Google":
        llm = ChatGoogleGenerativeAI(
            google_api_key=api_key,
            # model="gemini-pro",
            model=model_name,
            temperature=temperature,
            top_p=top_p,
            convert_system_message_to_human=True    
        ) # type: ignore

    if LLM_provider == "HuggingFace":
        llm = HuggingFaceHub(
            # repo_id="mistralai/Mistral-7B-Instruct-v0.2",
            repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
            huggingfacehub_api_token=api_key,
            model_kwargs={
                "temperature":temperature,
                "top_p": top_p,
                "do_sample": True,
                "max_new_tokens":1024
            },
        )
    return llm

def create_memory(model_name='gpt-3.5-turbo', memory_max_token=None):
    """Creates a ConversationSummaryBufferMemory for gpt-3.5-turbo
    Creates a ConversationBufferMemory for the other models."""

    if model_name=="gpt-3.5-turbo":
        if memory_max_token is None:
        #     memory_max_token = 1024 # max_tokens for 'gpt-3.5-turbo' = 4096
        # memory = ConversationSummaryBufferMemory(
        #     max_token_limit=memory_max_token,
        #     llm=ChatOpenAI(model_name="gpt-3.5-turbo",openai_api_key=openai_api_key,temperature=0.1),
        #     return_messages=True,
        #     memory_key='chat_history',
        #     output_key="answer",
        #     input_key="question"
        # )
            pass
    else:
        memory = ConversationBufferMemory(
            return_messages=True,
            memory_key='chat_history',
            output_key="answer",
            input_key="question",
        )
    return memory


def answer_template(language="french"):
    """Pass the standalone question along with the chat history and context (retrieved documents) to the `LLM` to get an answer."""

    template = f"""
    You're CampusAdvisor, you're a school guide. Your role is to help young graduates choose their studies according to their desires and opportunities.
    Answer the last question using only the following context (delimited by <context></context>).
    To answer this question, you'll need to use information about study programs, based on the admission criteria, career prospects and opportunities 
    offered indicated in the context provided. Be aware that the context provided is your brief; you should never give the impression that your answers 
    are based on the context provided. 

     
    Your answer should be written in the language of the ending.

    <chat_history>
        {{chat_history}}
    <chat_history>
    
    <context>
        {{context}}
    </context>

    Question: {{question}}

    Language: {language}.
    
    """

    # print("\n\n\n\nTemplate : {template}\n\n\n\n")

    return template


def _combine_documents(docs, document_prompt, document_separator="\n\n"):
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)

def custom_ConversationalRetrievalChain(
    llm,
    condense_question_llm,
    retriever,
    language="french",
    llm_provider="OpenAI",
    model_name='gpt-3.5-turbo',
):
    """Create a ConversationalRetrievalChain step by step.
    """
    ##############################################################
    # Step 1: Create a standalone_question chain
    ##############################################################

    # 1. Create memory: ConversationSummaryBufferMemory for gpt-3.5, and ConversationBufferMemory for the other models

    memory = create_memory(model_name)
    # memory = ConversationBufferMemory(memory_key="chat_history",output_key="answer", input_key="question",return_messages=True)

    # 2. load memory using RunnableLambda. Retrieves the chat_history attribute using itemgetter.
    loaded_memory = RunnablePassthrough.assign(
        chat_history=RunnableLambda(memory.load_memory_variables) | itemgetter("chat_history"),
    )


    # 3. Pass the follow-up question along with the chat history to the LLM, and parse the answer (standalone_question).
    condense_question_prompt = PromptTemplate(
        #input_variables=['chat_history', 'question'],
        input_variables=['question'],
        template = """Correct any errors in the input else return the input. Any answer other than the no-fault entry will be rejected by the system. Input: {question}\n
        Correct Input:"""
    )

    standalone_question_chain = {
        "standalone_question": {
            "question": lambda x: x["question"],
            "chat_history": lambda x: get_buffer_string(x["chat_history"]),
        }
        | condense_question_prompt
        | condense_question_llm
        | StrOutputParser(),
    }

    # 4. Combine load_memory and standalone_question_chain
    chain_question = loaded_memory | standalone_question_chain
    # chain_question = loaded_memory

    ####################################################################################
    #   Step 2: Retrieve documents, pass them to the LLM, and return the response.
    ####################################################################################

    # 5. Retrieve relevant documents
    retrieved_documents = {
        "docs": itemgetter("standalone_question") | retriever,
        "question": lambda x: x["standalone_question"],
    }

    # 6. Get variables ['chat_history', 'context', 'question'] that will be passed to `answer_prompt`

    DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")
    answer_prompt = ChatPromptTemplate.from_template(answer_template(language=language))
    
    # 3 variables are expected ['chat_history', 'context', 'question'] by the ChatPromptTemplate
    answer_prompt_variables = {
        "context": lambda x: _combine_documents(docs=x["docs"],document_prompt=DEFAULT_DOCUMENT_PROMPT),
        "question": itemgetter("question"),
        "chat_history": itemgetter("chat_history") # get it from `loaded_memory` variable
    }

    # 7. Load memory, format `answer_prompt` with variables (context, question and chat_history) and pass the `answer_prompt to LLM.
    # return answer, docs and standalone_question

    chain_answer = {
        "answer": loaded_memory | answer_prompt_variables | answer_prompt | llm,
        # return only page_content and metadata
        "docs": lambda x: [Document(page_content=doc.page_content, metadata=doc.metadata) for doc in x["docs"]],
        "standalone_question": lambda x:x["question"] # return standalone_question
    }

    # 8. Final chain
    conversational_retriever_chain = chain_question | retrieved_documents | chain_answer

    print("Conversational retriever chain created successfully!")
    # print("conversational_retriever_chain : " , conversational_retriever_chain)
    return conversational_retriever_chain,memory


chain_gemini, memory_gemini = custom_ConversationalRetrievalChain(
    llm = instantiate_LLM(
            LLM_provider="Google", api_key=google_api_key, temperature=0.5,model_name="gemini-1.5-pro"),
    
    condense_question_llm = instantiate_LLM(
            LLM_provider="Google", api_key=google_api_key, temperature=0.3, model_name="gemini-1.5-pro"),
    
    retriever=base_retriever_google,
    language="french",
    llm_provider="Google",
    model_name="gemini-1.5-pro"
)

# page config
st.set_page_config(page_title="CampusAdvisor", page_icon="👨🏾‍🎓")  

with st.sidebar:
    reset_button_key = "reset_button"
    reset_button = st.button("Reset Chat",key=reset_button_key)
    if reset_button:
        st.session_state.conversation = None
        st.session_state.chat_history = None
    # "[View the source code](https://github.com/TitanSage02/CampusAdvisor)"
st.image('logo.jpeg', width=200)

st.caption("🚀 Bienvenue CampusAdvisor, ton guide universitaire !")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "En quoi pouvons nous vous aider ?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    # Afficher un spinner pendant le traitement de la question
    with st.spinner('Traitement de la question en cours...'):
        response = chain_gemini.invoke({"question":prompt})
        
    msg = response['answer'].content
    st.session_state.messages.append({"role": "assistant", "content": msg})
    
    # Afficher un spinner pendant la génération de la réponse    
    with st.spinner('Génération de la réponse en cours...'):
        st.chat_message("assistant").write(msg)
        
    memory_gemini.save_context({"question": prompt}, {"answer": msg})
