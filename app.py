import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain

load_dotenv()
# OpenAI API 키 설정
api_key = os.getenv("OPENAI_API_KEY")

# 문서 로드 함수
def load_docs():
    loader = TextLoader('text.txt')
    print(loader)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(documents)
    
    return splits


# 벡터 스토어 생성 함수
def create_vectorstore(splits):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    return vectorstore


# RetrievalQA 체인 생성 함수
def create_rag_chain(vectorstore):
    llm = ChatOpenAI(model_name="gpt-4o", temperature=1, openai_api_key=api_key)

    prompt = ChatPromptTemplate.from_template("""아래의 문맥을 사용하여 질문에 답하십시오.
    만약 답을 모른다면, 모른다고 말하고 답을 지어내지 마십시오.
    최대한 세 문장으로 답하고 가능한 한 간결하게 유지하십시오.
    
    서울시의 재활용 지원 정책에 대한 정보를 제공하는 챗봇입니다. 
    친절하고 정중한 어조로 대답하세요. 한국어로 대답하세요. 당신의 이름은 영어로 '서울리챗' 입니다. 
    '행정구역' 열에서 귀하의 지역을 찾아 해당 행의 정보만 정확하게 읽으세요. 
    각 지역에 필요한 정보는 여러 행에 존재합니다. 
    먼저 지역을 찾아 연속 5개의 행을 읽어 '행정구역'에서 요청한 지역과 일치하는 정보만 검색한다. 
    답변 작성 시 “**”, “##” 등의 기호를 삭제하세요. 한 항목을 작성한 후 줄바꿈을 시행하세요.
    Context: {context}
    Question: {input}
    Answer:""")

    embeddings = OpenAIEmbeddings()
    vector = FAISS.from_documents(documents, embeddings)
    retriever = vector.as_retriever()
    document_chain = create_stuff_documents_chain(llm, prompt)
    qa_chain = create_retrieval_chain(retriever, document_chain)

    return qa_chain

# 문서 로드 및 벡터 스토어 생성
documents = load_docs()
vectorstore = create_vectorstore(documents)

# RetrievalQA 체인 생성
qa_chain = create_rag_chain(vectorstore)

# 예제 질문에 대한 답변 생성
question = "강서구 지원정책 알려줘"
answer = qa_chain.invoke({"input": question})

print(answer)
