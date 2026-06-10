# === CELL 0 (markdown) ===
#Importing dependencies

# === CELL 1 (code) ===
!pip install  langchain langgraph cassio

# === CELL 2 (code) ===
import cassio
 #connection of the Astra DB
ASTRA_DB_APPLICATION_TOKEN="AstraCS:JnQTGRKidFmXZKWvWnLAWZxD:10b82422b32b1b00c19965a26f394f998051a29aa7d450bac72761e462e1032c"
ASTRA_DB_ID="6e160db4-47b7-44ed-a1a0-bd8b721b1c0b"
cassio.init(token=ASTRA_DB_APPLICATION_TOKEN,database_id=ASTRA_DB_ID)

# === CELL 3 (code) ===
!pip install langchain_community

# === CELL 4 (code) ===
!pip install -U langchain langchain-community langchain-core langchain-text-splitters langgraph
!pip install -U langchain-groq langchain-huggingface langchainhub tiktoken

# === CELL 5 (markdown) ===
#RAG Part(Astra DB)

# === CELL 6 (code) ===
##build index

#data ingestion
from langchain_community.document_loaders import WebBaseLoader

#data transforamtion(chunking)
from langchain_text_splitters import RecursiveCharacterTextSplitter

#doc to index
urls=[
     "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",

]

#loaad the urls
docs=[WebBaseLoader(url).load() for url in urls]
doc_list =[item for sublist in docs for item in sublist]
print(doc_list)
text_splitter=RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=500,chunk_overlap=0)
docs_split= text_splitter.split_documents(doc_list)

# === CELL 7 (code) ===
docs_split

# === CELL 8 (code) ===
# text embedding
from langchain_huggingface import HuggingFaceEmbeddings
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# === CELL 9 (code) ===
#vector store db

from langchain_community.vectorstores import Cassandra
astra_vector_store=Cassandra(embedding=embeddings,
                             table_name="qa_mini_demo",
                             session=None,
                             keyspace=None)

# === CELL 10 (code) ===
astra_vector_store.add_documents(docs_split)
print("Inserted %i documents" % len(docs_split))


# === CELL 11 (code) ===
#retrieve the data
retriever=astra_vector_store.as_retriever()
retriever.invoke("whast is agent")

# === CELL 12 (markdown) ===
#Router

# === CELL 13 (code) ===
#langgraph application
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# === CELL 14 (code) ===
#data model
class RouteQuery(BaseModel):
  """ Route a user query to the most relevant datasource."""
  datasource: Literal["vectorstore","wiki_search"]= Field(
      ...,
      description="Given a user question choose to route it to wikipedia or a vectorstore.",

  )

# === CELL 15 (code) ===
#connecting to llm
from langchain_groq import ChatGroq
from google.colab import userdata

import os
groq_api_key=userdata.get('groq_api_key')
print(groq_api_key)

# === CELL 16 (code) ===
#addding a llm model
llm=ChatGroq(groq_api_key=groq_api_key,model_name="llama-3.3-70b-versatile")
llm

# === CELL 17 (code) ===
#adding router in llm

structured_llm_router=llm.with_structured_output(RouteQuery)

# === CELL 18 (code) ===
#promt
system="""You are an expert at routing a user question to a vectorstore or wikipedia.
The vectorstore contains documents related to agents, promt engineering,and  adversarial attacks.
Use the vectorstore  for questions on these topics .otherwise ,use wiki-search."""

route_promt=ChatPromptTemplate.from_messages(
    [
    ("system",system),
    ("human","{question}"),
    ]
)
question_route=route_promt | structured_llm_router

# === CELL 19 (code) ===
#checking which source it used

print(question_route.invoke(
    {"question":"what is agent ?"}
)
)

# === CELL 20 (code) ===
#checking which source it used
print(question_route.invoke(
    {"question":"what is Narendra modi ?"}
)
)

# === CELL 21 (markdown) ===
#Wikipedia part

# === CELL 22 (code) ===
!pip install langchain_community
!pip install wikipedia

# === CELL 23 (code) ===
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun
api_wrapper = WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper)

# === CELL 24 (code) ===
#testing the wiki
wiki.run("tell me about sharukh khan ")

# === CELL 25 (code) ===
#aiagent  application using Langgraph

from typing import List
from typing_extensions import TypedDict
from langchain_core.documents import Document

class GraphState(TypedDict):
  """Represents the  state of the graph.

  Attributes:
  question:queston
  genration: llm genration
  documents: list of documents
  """
  question: str
  generation: str
  documents: List[Document] # Changed from List[str] to List[Document]

# === CELL 26 (code) ===
from langchain_core.documents import Document


def retrieve(state):
    """retreive document

    Args:
    state(dic): the current graph state
    returns:
    state(dic):Now key added to state documents,that contains retrieved documents
    """

    print ("--Retreive--")
    question=state["question"]


    #retrievel
    documents= retriever.invoke(question)
    return{"documents":documents,"question": question}

# === CELL 27 (code) ===
from langchain_core.documents import Document

def wiki_search(state):
    """wiki search based on the re-phrased question.

    Args:
    state(dic): the current graph state
    returns:
    state(dic):Update documents key with appended web results"""

    print ("--Wikipedia--")
    question=state["question"]
    print(question)


    #wiki search
    wiki_content=wiki.invoke({"query": question})
    # The wiki.invoke() directly returns the content as a string, so we pass it as page_content
    wiki_document = Document(page_content=wiki_content)

    # Return a list of documents, consistent with the GraphState and retrieve function
    return {"documents": [wiki_document],"question":question}

# === CELL 28 (code) ===
##edge
def route_request(state):
    """ ROute question to wiki search or Rag

    args: state(dict):the current graph state
    returns: str:Next node to call
    """

    print("--ROUTE QUESTION--")
    question=state["question"]
    source=question_route.invoke({"question":question})

    if source.datasource=="wiki_search":
        print("--- ROUTE QUESTION TO Wiki search--")
        return "wiki_search"

    else:
        print("--- ROUTE QUESTION TO RAG--")
        return "vectorstore"

# === CELL 29 (code) ===
##making workflow

from langgraph.graph import END,StateGraph,START

workflow=StateGraph(GraphState)
#define the nodes
workflow.add_node("wiki_search",wiki_search)  #web search
workflow.add_node("retrieve",retrieve) #retrieve

#build the graph
workflow.add_conditional_edges(
    START,
    route_request,
    {
        "wiki_search":"wiki_search",
        "vectorstore":"retrieve",
    }
)

workflow.add_edge("retrieve",END)
workflow.add_edge("wiki_search",END)

#compile
app=workflow.compile()

# === CELL 30 (code) ===
#visualize

from IPython.display import Image,display
try:
    display(Image(app.get_graph().draw_mermaid_png()))
except Exception:
    # This requires some extra dependencies and its optional
    pass

# === CELL 31 (code) ===
from pprint import pprint

#chatbot interface
while True:
    question = input("\nAsk a question (type 'exit' to stop): ")

    if question.lower() == "exit":
        break

    inputs = {"question": question}
    final_output = None

    for output in app.stream(inputs):
        final_output = output

    for key, value in final_output.items():
        if "generation" in value:
            print("\nAnswer:\n")
            print(value["generation"])

        elif "documents" in value and value["documents"]:
            doc = value["documents"][0].model_dump()
            print("\nSource:\n")
            if "description" in doc.get("metadata", {}):
                print(doc["metadata"]["description"])
            else:
                print(doc["page_content"])

# === CELL 32 (code) ===


