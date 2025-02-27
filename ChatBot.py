from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

import streamlit as st

def rag_chain():
    model = ChatOllama(model="llama3")
    #
    prompt = PromptTemplate.from_template(
        """
        <s> [Instructions] You are a friendly assistant. Answer the question in Persian based only on the following context.
        If you don't know the answer, then reply, No Context availabel for this question {input}. [/Instructions] </s>
        [Instructions] Question: {input}
        Context: {context}
        Answer: [/Instructions]
        """
    )
    #Load vector store
    embedding = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
    vector_store = Chroma(persist_directory="./sql_chroma_db", embedding_function=embedding)

    #Create chain
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 3,
            "score_threshold": 0.5,
        },
    )

    document_chain = create_stuff_documents_chain(model, prompt)
    chain = create_retrieval_chain(retriever, document_chain)
    #
    return chain

def ask(query: str):
    #
    chain = rag_chain()
    # invoke chain
    result = chain.invoke({"input": query})
    # print results
    print(result["answer"])
    for doc in result["context"]:
        print("Source: ", doc)

    return result["answer"]


st.set_page_config(page_title="UniChatbot - NLP Final Proj 1403")
st.title("💬 UniChatbot")
st.caption("🚀 NLP Final Proj 1403")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    response = ask(prompt)
    msg = response
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)
