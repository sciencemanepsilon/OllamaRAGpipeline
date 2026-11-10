from pathlib import Path
from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer

class DoclingConverter:
    def __init__(self, pdf):
        pdf_path = Path(pdf)
        if not pdf_path.is_file(): raise FileNotFoundError(pdf_path)
        options = PdfPipelineOptions()
        options.do_ocr = False
        options.do_table_structure = True
        options.generate_page_images = False

        converter = DocumentConverter(format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=options)
        })
        result = converter.convert(pdf_path)
        if result.status != ConversionStatus.SUCCESS:
            raise RuntimeError(f"PDF conversion failed: {result.status}")

        self.document = result.document

class DoclingChunker:
    def __init__(self):
        #our_model = "sentence-transformers/all-MiniLM-L6-v2"
        our_model = "nomic-ai/nomic-embed-text-v1"
        self.tokenizer = HuggingFaceTokenizer.from_pretrained(
            model_name=our_model, max_tokens=8192
        )
        self.chunker = HybridChunker(
            tokenizer=self.tokenizer,
            #Whether to merge undersized chunks sharing same relevant metadata
            merge_peers=True, repeat_table_header=True
        )
    def CountTokens(self, text):
        return self.tokenizer.count_tokens(text)