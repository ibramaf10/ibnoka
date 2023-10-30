import streamlit as st
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import load_dotenv

load_dotenv()

# 1. Vectorise the sales response csv data
loader = CSVLoader(file_path="data.csv")
documents = loader.load()

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(documents, embeddings)

# 2. Function for similarity search


def retrieve_info(query):
    similar_response = db.similarity_search(query, k=3)

    page_contents_array = [doc.page_content for doc in similar_response]

    # print(page_contents_array)

    return page_contents_array


# 3. Setup LLMChain & prompts
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")

template = """
vous éte Quhafa قُحَافَةَ ; Quhafa قُحَافَةَ  est une IA islamique possédant une richesse de connaissances
sur l'Islam et le Coran. Il connaît bien le Coran et possède une compréhension approfondie de plusieurs
textes islamiques importants, notamment Sahih al-Bukhari, Sahih Muslim, Musnad Ahmad et les livres 
De l'Imam Tirmidhi. Quhafa  est toujours prêt à vous aider pour toutes vos questions concernant l'Islam 
et le Coran, et il s'efforcera de fournir des informations précises et utiles au mieux de ses capacités.
N'hésitez pas à poser toutes vos questions à Quhafa ; il se fera un plaisir de vous aider.
Vous êtes en présence de Quhafa, une IA spécialisée dans l'islam et la finance islamique.
Lorsque vous posez des questions en rapport avec la Sunnah ou le Coran, veuillez répondre en arabe.
Assurez-vous également de répondre dans la même langue que le message initial pour une meilleure compréhension.


Below is a message I received :
{message}

Here is a list of KNOWLEDGE BASE articles that I think might be helpful:
{best_practice}

Please write the best response with versets in arabic from quran and proofs in arabic from sunnah 
"""

prompt = PromptTemplate(
    input_variables=["message", "best_practice"],
    template=template
)

chain = LLMChain(llm=llm, prompt=prompt)


# 4. Retrieval augmented generation
def generate_response(message):
    best_practice = retrieve_info(message)
    response = chain.run(message=message, best_practice=best_practice)
    return response


# 5. Build an app with streamlit
def main():
    st.set_page_config(
        page_title="Quhafah - قُحَافَةَ ", page_icon=":scroll:")

    st.header("Quhafah - قُحَافَةَ  :scroll:")
    message = st.text_area("Message")

    if message:
        st.write("Generating response...")

        result = generate_response(message)

        st.info(result)


if __name__ == '__main__':
    main()
