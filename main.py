import os #os level functionalities 
import regex as re #advanced regular expressions
import google.generativeai as genai #function and tools forworking with google generative ai model
from chromadb import Documents, EmbeddingFunction, Embeddings #databse or embedding handling function
import chromadb 
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from typing import List #type printing for lists
import warnings #for managing warnings in code 

warnings.filterwarnings("ignore")
#gemini chat model used
GOOGLE_API_KEY="AIzaSyC--JQ0LK80SkZf519Wwwdf3s2RxQNPyMg"
api_key = GOOGLE_API_KEY
if not api_key:
    raise ValueError("Google API Key not provided. Please provide GOOGLE_API_KEY as an environment variable")
#to load pdf (ingest the pdf)
from pypdf import PdfReader
#retrival augmented generation 1(L-T-E) 2pipeline(model import) important for rag combining model
def load_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Replace the path with your file path
pdf_text = load_pdf(file_path="A Handbook on Employee Relations and Labour Laws in India.pdf")
#split the text(divide in chunks)size=800 chunk overlay200 (if no content on llm)
def split_text(text: str) -> List[str]:
    split_text = re.split('\n \n', text)
    return [i for i in split_text if i != ""]

chunked_text = split_text(text=pdf_text)
#embedding#vectorize the text(tf-idf)#google model(001)KNN
class GeminiEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        gemini_api_key = GOOGLE_API_KEY
        if not gemini_api_key:
            raise ValueError("Google API Key not provided. Please provide GOOGLE_API_KEY as an environment variable") #llm custom query generate 
        genai.configure(api_key=gemini_api_key)
        model = "models/embedding-001"
        title = "Custom query"
        return genai.embed_content(model=model, content=input, task_type="retrieval_document", title=title)["embedding"] #read content

def create_chroma_db(documents: List, path: str, name: str): #embedding creation through chroma every time new collection is made
    chroma_client = chromadb.PersistentClient(path=path)
    collections = chroma_client.list_collections()
    if name in [collection.name for collection in collections]:
        db = chroma_client.get_collection(name=name, embedding_function=GeminiEmbeddingFunction())
    else:
        db = chroma_client.create_collection(name=name, embedding_function=GeminiEmbeddingFunction())
        for i, d in enumerate(documents):
            db.add(documents=d, ids=str(i))
    return db, name

db, name = create_chroma_db(documents=chunked_text, path="vectorstore", name="rag_ex")
#function creation
def load_chroma_collection(path, name): 
    chroma_client = chromadb.PersistentClient(path=path)
    db = chroma_client.get_collection(name=name, embedding_function=GeminiEmbeddingFunction())
    return db
#function calling 
db = load_chroma_collection(path="vectorstore", name="rag_ex")
#query will be vectorize and chunks will be converted to number #distance between query vector and chunk will be calculated and nearest select 
def get_relevant_passage(query, db, n_results):
    try:
        results = db.query(query_texts=[query], n_results=n_results)
        if 'documents' in results and results['documents']:
            passage = results['documents'][0]#o th element will bring element
            print(passage)#query,db,no of results will be selected n=3 so top 3 will be selected dictonery kind result = dic name , document = key
            return passage#like tuple or list inside dictionery
        else:
            raise ValueError("No documents found for the query.")
    except Exception as e:
        print(f"Error retrieving passage: {e}")
        return None
#rag prompt is having function name in which we are passing query and relevant passage
def make_rag_prompt(query, relevant_passage):
    escaped = relevant_passage.replace("'", "").replace('"', "").replace("\n", " ")#structure is feeding the model query,path,pasage to get ans 
    prompt = ("""You are Insaaf Insight, an AI-driven legal assistant designed to help users with their legal queries in Indian courts. 
    Your demeanor is professional and informative. You will answer users' questions with your knowledge and the context provided. 
    If a question does not make any sense, or is not factually coherent, explain why instead of answering incorrectly. 
    If you don't know the answer to a question, please don't share false information. Be open about your capabilities and limitations.
    Do not say thank you and do not mention that you are an AI Assistant \
    If the passage is irrelevant to the answer, you may ignore it.
    QUESTION: '{query}'
    PASSAGE: '{relevant_passage}'

    ANSWER:
    """).format(query=query, relevant_passage=escaped)

    return prompt
#rag prompt is saved and then passed in generate ans
def generate_ans(prompt):
    gemini_api_key = GOOGLE_API_KEY#t.env is having key
    if not gemini_api_key:
        raise ValueError("Google API Key not provided. Please provide GOOGLE_API_KEY as an environment variable")
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel('gemini-pro')
    answer = model.generate_content(prompt)#ans will be generated
    return answer.text
#query,istruction prompt,relevant passage is passed#json model will be created with ans name 
def generate_answer(db, query):
    relevant_text = get_relevant_passage(query, db, n_results=3)
    if not relevant_text:
        return "No relevant text found for the query."
    prompt = make_rag_prompt(query, relevant_passage="".join(relevant_text))#stored in relevant text if found 
    answer = generate_ans(prompt)#assigned in rag prompt 
    return answer#passed in generate ans and then return

# Example usage
if __name__ == "__main__":
    try:
        db = load_chroma_collection(path="vectorstore", name="rag_ex")
        answer = generate_answer(db, query="Who are you?")
        print(answer)
    except Exception as e:
        print(f"An error occurred: {e}")













#cdmyenv#Scripts/activate#cd..#cd.insaaf#cd.code#streamlit run frontend.py

#cd codepath copy   #conda activate run2
