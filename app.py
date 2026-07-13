from flask import Flask,render_template,jsonify,request
from src.helper import  downloading_embedding
from langchain_pinecone  import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os



app=Flask(__name__)
load_dotenv()
PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
GROQ_API_KEY=os.environ.get("GROQ_API_KEY")
os.environ['PINECONE_API_KEY']=str(PINECONE_API_KEY)
os.environ['GROQ_API_KEY']=str(GROQ_API_KEY)

embeddings=downloading_embedding()
index_name="medical-bot"
# embed each chunk upsert the embeddingd into your pinecone index
docsearch=PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)
retrivers=docsearch.as_retriever(search_type="similarity",search_kwargs={"k":3})



chatModel=ChatGroq(model="llama-3.3-70b-versatile")
prompt=ChatPromptTemplate.from_messages([
    ("system",system_prompt),
    ("human","{input}")
])
question_answer_chain=create_stuff_documents_chain(chatModel,prompt)
rag_chain=create_retrieval_chain(retrivers,question_answer_chain)




@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get",methods=["GET","POST"])
def chat():
    msg=request.form['msg']
    input=msg
    print(input)
    response =rag_chain.invoke({"input":msg})
    print("Response :",response["answer"])
    return str(response["answer"])


if __name__=='__main__':
    app.run(host="0.0.0.0",port=8080 ,debug=True)