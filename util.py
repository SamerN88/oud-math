import os
import time

import numpy as np
from reportlab.lib import pagesizes
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


class Stopwatch:
    def __init__(self):
        self.start_time = None
        self.last_lap = None

    def start(self):
        self.start_time = self.last_lap = time.time()

    def lap(self):
        now = time.time()
        elapsed = now - self.last_lap
        self.last_lap = now
        return elapsed

    def elapsed(self):
        return time.time() - self.last_lap

    def total_elapsed(self):
        return time.time() - self.start_time


class FixedDict:
    def __init__(self, contents):
        # Expects a dict
        self._contents = dict(**contents)

    def items(self):
        return self._contents.items()

    def keys(self):
        return self._contents.keys()

    def values(self):
        return self._contents.values()

    def copy(self):
        return FixedDict(self._contents)

    def __getitem__(self, key):
        return self._contents[key]

    def __setitem__(self, key, value):
        self._contents[key] = value

    def __len__(self):
        return len(self._contents)

    def __repr__(self):
        return repr(self._contents)


class Style:
    def __init__(self, text):
        self.text = text
        self.reset = '\033[0m'
        self.styles = []

    def add_style(self, style):
        self.styles.append(style)

    def bold(self):
        self.add_style('\033[1m')
        return self

    def italics(self):
        self.add_style('\033[3m')
        return self

    def red(self):
        self.add_style('\033[91m')
        return self

    def yellow(self):
        self.add_style('\033[93m')
        return self

    def green(self):
        self.add_style('\033[92m')
        return self

    def blue(self):
        self.add_style('\033[94m')
        return self

    def magenta(self):
        self.add_style('\033[95m')
        return self

    def cyan(self):
        self.add_style('\033[96m')
        return self

    def gray(self):
        self.add_style('\033[90m')
        return self

    def dark_gray(self):
        self.add_style('\033[90m')
        return self

    def light_gray(self):
        self.add_style('\033[37m')
        return self

    def str(self):
        return str(self)

    def __str__(self):
        # Apply all accumulated styles to the text
        return ''.join(self.styles) + self.text + self.reset

    def __add__(self, other):
        return str(self) + str(other)


class PDFUtil:
    _FONTS_DIR = os.path.join(os.path.dirname(__file__), 'assets')

    def __init__(self):
        # Define PDF config values
        page_width, page_height = map(round, pagesizes.letter)  # corresponds to US Letter 8.5 x 11 inches at 72 dpi (width=612, height=792)
        margin = 60  # in pt
        font_name = 'Helvetica'
        mono_font_name = 'Menlo-Regular'
        font_size = 10
        text_line_height = font_size * (2 / 3)
        normal_line_width = 1
        thin_line_width = 0.6
        dash_config = (1, 2)
        stroke_color = (0, 0, 0)
        line_cap = 1  # round line caps
        small_pad = 3  # in pt, used for padding between drawings
        big_pad = 5

        # Register fonts as needed
        mono_font_ttf_path = self._get_ttf_path(mono_font_name)
        pdfmetrics.registerFont(TTFont(mono_font_name, mono_font_ttf_path))

        # Save PDF config
        self.page_width = page_width
        self.page_height = page_height
        self.margin = margin
        self.font_name = font_name
        self.mono_font_name = mono_font_name
        self.font_size = font_size
        self.text_line_height = text_line_height
        self.normal_line_width = normal_line_width
        self.thin_line_width = thin_line_width
        self.dash_config = dash_config
        self.stroke_color = stroke_color
        self.line_cap = line_cap
        self.small_pad = small_pad
        self.big_pad = big_pad

    def initialize_canvas(self, output_pdf_path):
        return canvas.Canvas(output_pdf_path, pagesize=(self.page_width, self.page_height))

    def apply_default_pdf_config(self, pdf):
        font_name = self.font_name
        font_size = self.font_size
        normal_line_width = self.normal_line_width
        stroke_color = self.stroke_color
        line_cap = self.line_cap

        pdf.setFont(font_name, font_size)
        pdf.setLineWidth(normal_line_width)
        pdf.setStrokeColorRGB(*stroke_color)
        pdf.setLineCap(line_cap)

    @staticmethod
    def draw_vertical_bracket(pdf, x, y, width, height, direction):
        dxn = direction.strip().lower()
        if dxn not in {'left', 'right'}:
            raise ValueError(f'direction must be either "left" or "right", got "{direction}"')

        width *= (dxn == 'right') * 2 - 1
        half_width = width / 2

        pdf.line(x, y, x + half_width, y)  # lower tick
        pdf.line(x, y + height, x + half_width, y + height)  # upper tick
        pdf.line(x + half_width, y, x + half_width, y + height)  # main line
        pdf.line(x + half_width, y + (height / 2), x + width, y + (height / 2))  # middle tick

    @staticmethod
    def draw_horizontal_bracket(pdf, x, y, width, height, direction):
        dxn = direction.strip().lower()
        if dxn not in {'up', 'down'}:
            raise ValueError(f'direction must be either "up" or "down", got "{direction}"')

        height *= (dxn == 'up') * 2 - 1
        half_height = height / 2

        pdf.line(x, y, x, y + half_height)  # left tick
        pdf.line(x + width, y, x + width, y + half_height)  # right tick
        pdf.line(x, y + half_height, x + width, y + half_height)  # main line
        pdf.line(x + (width / 2), y + half_height, x + (width / 2), y + height)  # middle tick

    @staticmethod
    def mm_to_pt(mm):
        return pagesizes.mm * mm

    @staticmethod
    def pt_to_mm(pt):
        return pt / pagesizes.mm

    def _get_ttf_path(self, font_name):
        return os.path.join(self._FONTS_DIR, f'{font_name}.ttf')


def points_from_file(fp):
    with open(fp, 'r') as f:
        lines = f.readlines()

    points = []
    for line in lines:
        point_str = line.strip()

        # Supported formats: "(x, y)", "x y", "x, y"
        for ch in ['(', ')', ',']:
            point_str = point_str.replace(ch, '')

        if point_str == '':
            continue

        point = tuple(map(float, point_str.split()))
        points.append(point)

    # Convert the list of points to a 2D NumPy array
    return np.array(points)
