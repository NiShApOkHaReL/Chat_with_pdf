from langchain_community.document_loaders import PyPDFLoader

class DocumentTextExtractor:

    def extract_text(self, file_path:str):
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        return docs
        

