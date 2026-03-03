from langchain_community.document_loaders import DirectoryLoader,PyPDFLoader 

loader=DirectoryLoader(
    path='AI',
    glob='*.pdf',
    loader_cls=PyPDFLoader 
) 

docs=loader.load()

print(len(docs)) #all pdfs page in doc format
print(docs[0]) #First pdf first page