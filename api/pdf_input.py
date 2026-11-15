from MyChunker import DoclingChunker, DoclingConverter
import httpx
from fastapi import HTTPException


class Pdf_DataUpdater:
    docling = DoclingChunker()
    chunks, metadata = (None, None)
    hpx = httpx.AsyncClient(timeout=0)

    @classmethod
    async def OverWriteByLocalFile(cls, FileName, m1='Splitter/TableSimple', m2='Splitter/nlp'):
        cov = DoclingConverter(FileName)
        DocIterator = DoclingItemsIterator(cls.docling.tokenizer)

        for i, chunk in enumerate(cls.docling.chunker.chunk(cov.document)):
            print(f"--- Chunk {i} ---")
            await DocIterator.Run(i, chunk, tableMethod=m1, textMethod=m2)

        cls.metadata = DocIterator.metadata
        cls.chunks = DocIterator.chunks


class DoclingItemsIterator:
    def __init__(self, tokenizer):
        self.MAX_TOKENS_LIMIT = 50
        self.tokenizer = tokenizer
        self.headings = None
        self.metadata, self.chunks = [],[]
        self.BaseUrl = 'http://splitter-svc:6001'
    
    async def MakeRequest(self, line, i, j, item, headings, method):
        response = await Pdf_DataUpdater.hpx.post(
            f'{self.BaseUrl}/{method}/{i}/{j}',
            json={'text':line, 'headings':headings, 'SplitMeth':None},
            params={'item':item.self_ref}
        )
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail="splitter-svc non 200")
        return response.json()
    
    def ValidateItemsVsLines(self, chunk):
        lines = chunk.text.splitlines()
        nLines, nItems = len(lines), len(chunk.meta.doc_items)
        if nLines != nItems: raise RuntimeError("Chunk lines and items mismatch")
        return chunk.meta.headings, lines
    
    def GatherItemInfo(self, item, line, chunk):
        label = str(item.label)
        nTokens = self.tokenizer.count_tokens(line)
        print(f"{type(item).__name__}-{label}-{item.self_ref} heading:{self.headings}")
        print(f"{nTokens} {len(chunk.text)} {chunk.text}")
        return label, nTokens

    async def Run(self, i, chunk, tableMethod=None, textMethod=None):
        self.headings, lines = self.ValidateItemsVsLines(chunk)
        for j, item in enumerate(chunk.meta.doc_items):
            label, nTokens = self.GatherItemInfo(item, lines[j], chunk)
            match label:
                case 'table': await self.HandleTable(lines[j], i, j, item, tableMethod)
                case 'text': await self.HandleText(nTokens, item, lines[j], i, j, textMethod)
                case _: continue

    async def HandleText(self, nTokens, item, line, i, j, method):
        if nTokens < self.MAX_TOKENS_LIMIT:
            strings = [line]
            records = [await self.MakeRequest(line, i, j, item, self.headings, method)]

        else: strings, records = await self.MakeRequest(line, i, j, item, self.headings, method)
        self.metadata.extend(records)
        self.chunks.extend(strings)

    async def HandleTable(self, line, i, j, item, method):
        array, RecordList = await self.MakeRequest(line, i, j, item, self.headings, method)
        self.metadata.extend(RecordList)
        self.chunks.extend(array)