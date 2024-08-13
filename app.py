import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.graphs import Neo4jGraph
from langchain_chains import GraphQAChain

# Initialize the LLM and Neo4j connection
llm = ChatGroq(
    model="Llama3-8b-8192",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=5,
    groq_api_key='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',  # Your API key here
)

graph = Neo4jGraph()

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
