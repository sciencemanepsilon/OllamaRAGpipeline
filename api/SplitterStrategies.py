import spacy, re
from langchain_text_splitters import RecursiveCharacterTextSplitter

class SplitterBase:
    @staticmethod
    def ChunkRecord(i, j, text, item, headings, SplitMeth, tokens):
        return {
            "tokens": tokens,
            "chunk_id": f"{i}-{j}-{item.self_ref}",
            "chunk": text,
            "headings": headings[0],
            "split_method": SplitMeth
        }
    @classmethod
    def Main(cls, array, i, j, item, headings, tokenizer, MethString):
        return [cls.ChunkRecord(
            i, j, field, item, headings, MethString, tokenizer.count_tokens(field)
            ) for field in array if field
        ]

class TableSplitters(SplitterBase):
    StrSplit = "str.split(sep='. ')"
    StrLangChain = "langchain-text-splitter"
    #Splits table line into individual cell text elements
    table_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1, chunk_overlap=0,
        is_separator_regex=False, length_function=len,
        keep_separator=False, strip_whitespace=True, separators=[". "]
    )
    @classmethod
    def Simple(cls, line, i, j, item, headings, tokenizer):
        array = line.split('. ')
        RecordList = cls.Main(array, i, j, item, headings, tokenizer, cls.StrSplit)
        return array, RecordList

    @classmethod
    def LangChainMethod(cls, line, i, j, item, headings, tokenizer):
        return cls.Main(
            cls.table_splitter.split_text(line),
            i, j, item, headings, tokenizer, cls.StrLangChain
        )


class TextSplitters(SplitterBase):
    NlpSplit = "spacy.nlp(en_core_web_sm)"
    StrLangChain = "langchain-text-splitter-regex"
    # Named-entity-recognition "ner" is unnecessary here.
    nlp = spacy.load("en_core_web_sm", disable=["ner"])

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=0, is_separator_regex=True,
        length_function=len, keep_separator=False, strip_whitespace=True,
        separators=[r"(?<=\.) (?![a-z])", ".\n\n", ",\n\n", "\n\n", ".\n", "\n"]
    )
    @classmethod
    def LangChainMethod(cls, line, i, j, item, headings, tokenizer):
        StringList = cls.text_splitter.split_text(line)
        RecordList = cls.Main(
            StringList, i, j, item, headings, tokenizer, cls.StrLangChain
        )
        return StringList, RecordList
    @classmethod
    def NlpMethod(cls, line, i, j, item, headings, tokenizer):
        StringList, RecordList = [],[]
        for paragraph in cls.CreateParagraphs(line):
            paragraph = paragraph.strip()
            if not paragraph: continue
            strings, chunks = cls.RunByNlp(
                paragraph, i, j, item, headings, tokenizer
            )
            RecordList.extend(chunks)
            StringList.extend(strings)
        return StringList, RecordList

    @staticmethod
    def CreateParagraphs(line):
        NewLineFreeLine = line.replace("\r\n", "\n")
        return re.split(r"\n\s*\n+", NewLineFreeLine)
    
    @classmethod
    def RunByNlp(cls, paragraph, i, j, item, headings, tokenizer):
        strings, chunks = [],[]
        doc = cls.nlp(paragraph)
        for sentence in doc.sents:
            content = sentence.text.strip()
            print(f"sentence chunk: {content}")
            if not content: continue
            chunks.append(cls.ChunkRecord(
                i, j, content, item, headings,
                cls.NlpSplit, tokenizer.count_tokens(content)
            ))
            strings.append(content)
        return strings, chunks