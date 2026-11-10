import uuid, chromadb
from my_init import embedding_model, ollama_host
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from fastapi import HTTPException

class ChromaNative:
    nResultsDefault = 1

    @classmethod
    def GetQueryResults(cls, ref):
        if cls.nResultsDefault > 1: return cls.ZipMethod(ref)
        try: return ref["distances"][0][0], ref["documents"][0][0]
        except Exception as e:
            print(str(e))
            return cls.ZipMethod(ref)

    @staticmethod
    def ZipMethod(ref):
        try:
            dist, docs = zip(ref["distances"][0], ref["documents"][0])
            pairs = [zip(dist[i], docs[i]) for i in range(len(dist[0]))]
            # [ (0.2, metal table), (0.8, wood table) ]
            print(f"Query Result: {pairs}")
            return pairs[0]
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"staric.ZipMethod failed:{e}")
    
    @staticmethod
    def GenIds(iterable):
        return [str(uuid.uuid4()) for _ in iterable]

    @staticmethod
    def SetEmbeddings():
        return OllamaEmbeddingFunction(url=ollama_host, model_name=embedding_model)
    
    @staticmethod
    def SetDB(client, emb):
        try:
            refDB = client.create_collection(name="ref", embedding_function=emb)
            catDB = client.create_collection(name="catalog", embedding_function=emb)
        except Exception as e:
            print(str(e))
            refDB = client.get_collection(name="ref")
            catDB = client.get_collection(name="catalog")
        print(f"server start db count: {refDB.count()}")
        return refDB, catDB
    #refDB.add(
        #documents=FURNITURE_REFERENCES,
        #embeddings=reference_response.embeddings, ids=GenChromaIds(FURNITURE_REFERENCES)
    #)
    #results = db.query(query_texts=["cloud deployment"], n_results=2)

chroma_client = chromadb.PersistentClient(path="/chroma_data")
embeddings = ChromaNative.SetEmbeddings()
refDB, catDB = ChromaNative.SetDB(chroma_client, embeddings)