from pathlib import Path
import re

import pdfplumber
import textract


class PdfData():
    WORD_PATTERN = re.compile(r"[^\W_]+", re.MULTILINE)

    def __init__(self,
                 filepath: Path,
                 encoding: str = 'utf_8'):
        self._encoding = encoding
        self._pdf_path = filepath
        self._page_count = 0

        self.pdf = pdfplumber.open(self.pdf_path)

    def process(self):
        self.count_words_in_pdf()
        self.get_images_coordinates()
        self.get_text_objects_coordinates()

    @property
    def pdf_path(self):
        return self._pdf_path

    @property
    def words_count(self):
        return self._words_count

    @property
    def page_count(self):
        return self._page_count

    @property
    def image_coordinates(self):
        return self._image_coordinates

    @property
    def text_objects_coordinates(self):
        return self._text_objects_coordinates

    def count_words_in_pdf(self) -> dict:
        if self.pdf_path is not None:
            self._count_words()
            return self.words_count

    def get_images_coordinates(self) -> dict:
        if self.pdf_path is not None:
            self._get_images_coordinates()
            return self.image_coordinates

    def get_text_objects_coordinates(self) -> dict:
        if self.pdf_path is not None:
            self._get_text_objects_coordinates()
            return self.text_objects_coordinates

    def _count_words(self) -> dict:
        filename = self.pdf_path
        words_dict = {}
        raw_text = textract.process(filename, encoding=self._encoding)
        pages = self._split_raw_text_on_pages(raw_text)
        self._page_count = len(pages)
        is_no_words = all(map(lambda x: x == '', pages))
        if is_no_words and len(pages) < 30:
            raw_text_from_images = textract.process(filename,
                                                    encoding=self._encoding,
                                                    method='tesseract')
            pages = self._split_raw_text_on_pages(raw_text_from_images)
        for num_page, page_text in enumerate(pages):
            words_dict[str(num_page+1)] = self._count_words_in_page_text(page_text)
        self._words_count = words_dict

    def _split_raw_text_on_pages(self, raw_text: bytes) -> list:
        pages = []
        raw_pages = raw_text.split(b'\x0c')
        for raw_page in raw_pages[:-1]:
            page = raw_page.decode(self._encoding)
            pages.append(page)
        if len(raw_pages[-1]) > 0:
            pages.append(raw_pages[-1].decode(self._encoding))
        return pages

    def _count_words_in_page_text(self, page_text: str) -> int:
        words_list = re.findall(self.WORD_PATTERN, page_text)
        return len(words_list)

    def _get_images_coordinates(self) -> dict:
        pdf_images = {}
        for page in self.pdf.pages:
            w = page.width
            h = page.height
            page_num = str(page.page_number)
            pdf_images[page_num] = []
            for image in page.images:
                pdf_images[page_num].append({
                    "x0": float(image["x0"]/w),
                    "x1": float(image["x1"]/w),
                    "y0": 1 - float(image["y1"]/h),  # from page top
                    "y1": 1 - float(image["y0"]/h),  # from page top
                })
        self._image_coordinates = pdf_images

    def _get_text_objects_coordinates(self) -> dict:
        pdf_text_objects = {}
        for page in self.pdf.pages:
            w = page.width
            h = page.height
            page_num = str(page.page_number)
            pdf_text_objects[page_num] = []
            for text_obj in page.extract_words(x_tolerance=10,
                                               y_tolerance=10,
                                               keep_blank_chars=True):
                pdf_text_objects[page_num].append({
                    "x0": float(text_obj["x0"]/w),
                    "x1": float(text_obj["x1"]/w),
                    "y0": float(text_obj["top"]/h),  # from page top
                    "y1": float(text_obj["bottom"]/h),  # from page top
                })
        self._text_objects_coordinates = pdf_text_objects
