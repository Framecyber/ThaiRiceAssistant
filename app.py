import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.graphs.networkx_graph import NetworkxEntityGraph
from langchain.chains import GraphQAChain
from neo4j import GraphDatabase
import networkx as nx
import os

# Initialize the LLM
llm = ChatGroq(
    model="Llama3-8b-8192",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=5,
    groq_api_key='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',  # Your API key here
)

# Set up Neo4j connection
uri = "neo4j+s://46084f1a.databases.neo4j.io"
user = "neo4j"
password = "FwnX0ige_QYJk8eEYSXSF0l081mWWGIS7TFg6t8rLZc"
driver = GraphDatabase.driver(uri, auth=(user, password))

def fetch_nodes(tx):
    query = "MATCH (n) RETURN id(n) AS id, labels(n) AS labels"
    result = tx.run(query)
    return result.data()

def fetch_relationships(tx):
    query = "MATCH (n)-[r]->(m) RETURN id(n) AS source, id(m) AS target, type(r) AS relation"
    result = tx.run(query)
    return result.data()

def populate_networkx_graph():
    G = nx.Graph()
    
    with driver.session() as session:
        nodes = session.read_transaction(fetch_nodes)
        relationships = session.read_transaction(fetch_relationships)
        
        for node in nodes:
            G.add_node(node['id'], labels=node['labels'])
        
        for relationship in relationships:
            G.add_edge(
                relationship['source'],
                relationship['target'],
                relation=relationship['relation']
            )
    
    return G

# Initialize Networkx Graph
networkx_graph = populate_networkx_graph()

# Initialize NetworkxEntityGraph
graph = NetworkxEntityGraph()

# Set the NetworkX graph to NetworkxEntityGraph
graph._graph = networkx_graph  # Directly assign the NetworkX graph

# Initialize the GraphQAChain with the NetworkXEntityGraph
chain = GraphQAChain.from_llm(
    llm=llm,
    graph=graph,
    verbose=True
)

# Streamlit app layout
st.title('Thai Rice Assistant Chatbot')

# User input
question = st.text_area("Ask a question about Thai rice:")

if st.button('Get Answer'):
    if question:
        response = chain.run(question)
        st.write("Answer:")
        st.write(response)
    else:
        st.error("Please enter a question.")
