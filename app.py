from dotenv import load_dotenv
import os
from pinecone import Pinecone, ServerlessSpec
from pinecone.grpc import PineconeGRPC as Pinecone
from langchain_pinecone import PineconeVectorStore
from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings
from flask import Flask,render_template,jsonify,request
from langchain_openai import OpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from src.prompt import *

load_dotenv()

app = Flask(__name__)

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY=os.environ.get('OPENAI_API_KEY')

embeddings = download_hugging_face_embeddings()

index_name = "medicalbot"

#embed each chunk and upsert the embeddings into Pinecone Index
docsearch=PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

#Create retrieve object to capture # of search results
retriever=docsearch.as_retriever(search_type="similarity",search_kwargs={"k":3})

#Initializing and Configuring OpenAI LLM
llm=OpenAI(temperature = 0.4, max_tokens = 500)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human","{input}"),
    ]
)

#Retrieving results from RAG using LLM
question_answer_chain = create_stuff_documents_chain(llm,prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/get", methods = ["GET","POST"])
def chat():
    msg = request.form["msg"]
    print("User Input:", msg)
    """ input=msg """
    print(input)
    response = rag_chain.invoke({"input":msg})
    print("Response:" , response["answer"])
    return str(response["answer"])

if __name__ == '__main__':
    app.run(host="0.0.0.0" , port = 8080 , debug= True)