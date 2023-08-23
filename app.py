import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os
from langchain.embeddings.llamacpp import LlamaCppEmbeddings
OPENAI_API_KEY='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'
os.environ["OPENAI_API_KEY"] ='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'

# Sidebar contents
with st.sidebar:
    st.title('API-on')
    st.markdown('''
    ## About:
    im Apion, an API-on Greek culture. API is a way for software to communicate, and on means I’m always ready to chat with you. 😉

    Apion was also a Hellenized Egyptian scholar who wrote about many topics, such as Egypt, Pythagoras, Homer, and Apicius.
    
    I am a language model fine-tuned on Greek architecture and mythology and their impact on modern society. 
    I can also create graphic artworks based on prompts related to these topics.
    I hope you enjoy chatting with me and discovering new insights about the fascinating world of Greece.''')
    add_vertical_space(5)
    st.write('Developed Amity 43')


 

def main():
    st.header("what brings you here")

    load_dotenv()

    # upload a PDF file
    pdf = st.file_uploader("Upload  the attached PDF", type='pdf')

    # st.write(pdf)
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
            )
        chunks = text_splitter.split_text(text=text)

        # # embeddings
        store_name = pdf.name[:-4]
        st.write(f'{store_name}')
        # st.write(chunks)

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
            # st.write('Embeddings Loaded from the Disk')s
        else:
            embeddings = OpenAIEmbeddings()
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)

        # embeddings = OpenAIEmbeddings()
        # VectorStore = FAISS.from_texts(chunks, embedding=embeddings)

        # Accept user questions/query
        query = st.text_input("Ask all your law related questions here")
        # st.write(query)

        if query:
            docs = VectorStore.similarity_search(query=query, k=3)

            llm = OpenAI()
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                print(cb)
            st.write(response)

if __name__ == '__main__':
    main()