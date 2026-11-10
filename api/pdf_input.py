from MyChunker import DoclingChunker, DoclingConverter
from SplitterStrategies import SplitterBase, TableSplitters, TextSplitters


class Pdf_DataUpdater:
    docling = DoclingChunker()
    FURNITURE_CATALOG, FURNITURE_REFERENCES = (None,None)

    @classmethod
    def OverWriteByLocalFile(cls, FileName, m1=TableSplitters.Simple, m2=TextSplitters.NlpMethod):
        cov = DoclingConverter(FileName)
        DocIterator = DoclingItemsIterator(cls.docling.tokenizer)

        for i, chunk in enumerate(cls.docling.chunker.chunk(cov.document)):
            print(f"--- Chunk {i} ---")
            DocIterator.Run(i, chunk, TABLE_SPLITTER=m1, TEXT_SPLITTER=m2)

        print(DocIterator.strings[0:15])
        cls.FURNITURE_CATALOG = DocIterator.chunks
        cls.FURNITURE_REFERENCES = DocIterator.strings


class DoclingItemsIterator:
    def __init__(self, tokenizer):
        self.MAX_TOKENS_LIMIT = 50
        self.tokenizer = tokenizer
        self.headings = None
        self.chunks, self.strings = [],[]
    
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

    def Run(self, i, chunk, TABLE_SPLITTER=TableSplitters.Simple, TEXT_SPLITTER=TextSplitters.NlpMethod):
        self.headings, lines = self.ValidateItemsVsLines(chunk)
        for j, item in enumerate(chunk.meta.doc_items):
            label, nTokens = self.GatherItemInfo(item, lines[j], chunk)
            match label:
                case 'table': self.HandleTable(lines[j], i, j, item, method=TABLE_SPLITTER)
                case 'text': self.HandleText(nTokens, chunk, item, lines[j], i, j, method=TEXT_SPLITTER)
                case _: continue

    def HandleText(self, nTokens, chunk, item, line, i, j, method=None):
        if nTokens < self.MAX_TOKENS_LIMIT:
            strings = [chunk.text]
            records = [SplitterBase.ChunkRecord(
                i, j, chunk.text, item, self.headings, None, nTokens
            )]
        else: strings, records = method(line, i, j, item, self.headings, self.tokenizer)
        self.chunks.extend(records)
        self.strings.extend(strings)

    def HandleTable(self, line, i, j, item, method=TableSplitters.Simple):
        array, RecordList = method(line, i, j, item, self.headings, self.tokenizer)
        self.chunks.extend(RecordList)
        self.strings.extend(array)