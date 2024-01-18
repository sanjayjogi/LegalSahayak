from googletrans import Translator
import textwrap
from langchain.llms import HuggingFaceHub
from langchain.vectorstores import Pinecone
import pinecone
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.chains.question_answering import load_qa_chain
from InstructorEmbedding import INSTRUCTOR
from flask import Flask, request
import os
from dotenv import load_dotenv
load_dotenv()

instructor_embeddings = HuggingFaceInstructEmbeddings()

pinecone.init(
    api_key="",
    environment="gcp-starter"
)


app = Flask(__name__)

index_name = "instr"
Embeddings = instructor_embeddings
index = Pinecone.from_existing_index(index_name, Embeddings)

huggingfacehub_api_token = os.getenv('HUGGINGFACEHUB_API_TOKEN')

repo_id = "tiiuae/falcon-7b"
llm = HuggingFaceHub(huggingfacehub_api_token=huggingfacehub_api_token,
                     repo_id=repo_id,
                     model_kwargs={"temperature": 0.6, "max_new_tokens": 100})

chain = load_qa_chain(llm, chain_type="stuff")


def wrap_text_preserve_newlines(text, width=110):
    lines = text.split('\n')

    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]

    wrapped_text = '\n'.join(wrapped_lines)

    return wrapped_text


def process_llm_response(llm_response):
    return wrap_text_preserve_newlines(llm_response)
    # print('\nSources:')
    # for source in llm_response["source_documents"]:
    #     print(source.metadata['source'])


translator = Translator()


@app.route('/', methods=['GET'])
def welcome():
    json_data = request.get_json()
    query = json_data['query']

    query_translation = translator.translate(query, dest='en')

    if query_translation.src != 'en':
        docs = index.similarity_search(query_translation)
        llm_response = chain.run(input_documents=docs,
                                 question=query_translation)

    else:
        docs = index.similarity_search(query)
        llm_response = chain.run(input_documents=docs, question=query)

    answer = process_llm_response(llm_response)
    res = {'query': query, 'answer': answer}
    return res


if __name__ == '__main__':
    from waitress import serve
    serve(app, host="0.0.0.0", port=5000)
    app.run(debug=True)
