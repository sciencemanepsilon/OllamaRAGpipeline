import spacy, re
from langchain_text_splitters import RecursiveCharacterTextSplitter

class SplitterBase:
    split_text = None
    StrLangChain = None

    @classmethod
    def LangChainMethod(cls, line, i, j, item, headings):
        array = cls.split_text(line)
        RecordList = cls.Main(array, i, j, item, headings, cls.StrLangChain)
        return array, RecordList

    @staticmethod
    def ChunkRecord(i, j, text, item, headings, SplitMeth):
        return {
            "chunk_id": f"{i}-{j}-{item}",
            "chunk": text,
            "headings": headings[0],
            "split_method": SplitMeth
        }
    @classmethod
    def Main(cls, array, i, j, item, headings, MethString):
        return [cls.ChunkRecord(
            i, j, field, item, headings, MethString) for field in array if field
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
    split_text = table_splitter.split_text

    @classmethod
    def Simple(cls, line, i, j, item, headings):
        array = line.split('. ')
        RecordList = cls.Main(array, i, j, item, headings, cls.StrSplit)
        return array, RecordList



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
    split_text = text_splitter.split_text
    

    @classmethod
    def NlpMethod(cls, line, i, j, item, headings):
        StringList, RecordList = [],[]
        for paragraph in cls.CreateParagraphs(line):
            paragraph = paragraph.strip()
            if not paragraph: continue
            strings, chunks = cls.RunByNlp(paragraph, i, j, item, headings)
            RecordList.extend(chunks)
            StringList.extend(strings)
        return StringList, RecordList

    @staticmethod
    def CreateParagraphs(line):
        NewLineFreeLine = line.replace("\r\n", "\n")
        return re.split(r"\n\s*\n+", NewLineFreeLine)
    
    @classmethod
    def RunByNlp(cls, paragraph, i, j, item, headings):
        strings, chunks = [],[]
        doc = cls.nlp(paragraph)
        for sentence in doc.sents:
            content = sentence.text.strip()
            print(f"sentence chunk: {content}")
            if not content: continue
            chunks.append(cls.ChunkRecord(
                i, j, content, item, headings, cls.NlpSplit
            ))
            strings.append(content)
        return strings, chunks