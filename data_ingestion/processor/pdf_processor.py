import logging
from pathlib import Path
from docling_core.types.doc import ImageRefMode, PictureItem, TableItem
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import FigureElement, InputFormat, Table
import time



_log = logging.getLogger(__name__)

class PDFProcessor:
    def __init__(self, pdf_path, output_dir, markdown_file_name):
        self.pdf_path = Path(pdf_path)
        self.output_dir = Path(output_dir)
        self.markdown_file_name = Path(markdown_file_name)
        self.image_resolution_scale = 2.0  # Ajuste a escala de resolução da imagem


    

    def convert_pdf_to_markdown(self):
        """Converte um PDF para Markdown, exportando imagens e tabelas separadamente."""
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"Arquivo PDF '{self.pdf_path}' não encontrado.")

        logging.basicConfig(level=logging.INFO)

        pipeline_options = PdfPipelineOptions()
        pipeline_options.images_scale = self.image_resolution_scale
        pipeline_options.generate_page_images = True
        pipeline_options.generate_picture_images = True
    
        doc_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

        start_time = time.time()
        conv_res = doc_converter.convert(self.pdf_path)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        doc_filename = conv_res.input.file.stem

        # Salvar imagens de tabelas e figuras
        table_counter = 0
        picture_counter = 0

        for element, _ in conv_res.document.iterate_items():
            if isinstance(element, TableItem):
                table_counter += 1
                element_image_filename = self.output_dir / f"{doc_filename}-table-{table_counter}.png"
                with element_image_filename.open("wb") as fp:
                    element.get_image(conv_res.document).save(fp, "PNG")

            if isinstance(element, PictureItem):
                picture_counter += 1
                element_image_filename = self.output_dir / f"{doc_filename}-picture-{picture_counter}.png"
                with element_image_filename.open("wb") as fp:
                    element.get_image(conv_res.document).save(fp, "PNG")

        # Salvar Markdown
        markdown_path = self.output_dir / self.markdown_file_name
        conv_res.document.save_as_markdown(markdown_path, image_mode=ImageRefMode.REFERENCED)

        end_time = time.time() - start_time
        _log.info(f"Documento convertido e figuras exportadas em {end_time:.2f} segundos.")

        return markdown_path, markdown_path.name