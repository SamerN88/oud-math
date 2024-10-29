import math
import textwrap
from datetime import datetime

import numpy as np

from util import PDFUtil, Stopwatch, Style


# TODO: Restructure project as follows:
#       - create file rib.py containing RibGenerator and RibMath
#       - create file bowl.py containing OudSpecs, PointsSelector (rename?), and Optimizer
#       - the rest remains as is


class RibGenerator:
    def __init__(self, model, num_ribs, *, verbose=False):
        self.model = model
        self.num_ribs = num_ribs

        # RibMath object, which contains all the core mathematical functionality to accurately generate the rib
        self.math = RibMath(self.model, self.num_ribs)

        # Specs object for the model being used
        self.oud_specs = OudSpecs(model)

        # Verbose config
        self.verbose = bool(verbose)
        self.print = print if verbose else (lambda *_, **__: None)

        # PDF config and utility functions
        self._pdf_util = PDFUtil()

    # Entry point
    def run(self, output_pdf_path):
        # [Verbose] Display model, number of ribs, and rib length
        self.print()
        self.print('-' * 75)
        self.print(Style('Running Rib Generator').bold().cyan(), Style(f'[verbose = {self.verbose}]').gray() + '\n')
        self.print(f'model = {self.model}')
        self.print(f'num_ribs = {self.num_ribs}')
        self.print()
        self.print('Rib Length:', Style(f'{self.math.l(self.model.H):.1f} mm').blue())

        stopwatch = Stopwatch()
        stopwatch.start()

        # Generate rib template PDF
        self._make_rib_template_pdf(output_pdf_path)

        # [Verbose] Display output PDF path and runtime
        self.print(f'\nDone. See file {Style(output_pdf_path).bold().green()}')
        self.print(f'Runtime: {stopwatch.total_elapsed():.3f} s')
        self.print('-' * 75)
        self.print()

    def _make_rib_template_pdf(self, output_pdf_path):
        # Get PDF config
        page_width = self._pdf_util.page_width
        page_height = self._pdf_util.page_height
        margin = self._pdf_util.margin
        text_line_height = self._pdf_util.text_line_height

        rib_length_mm = self.math.l(self.model.H)
        rib_length_pt = self._pdf_util.mm_to_pt(rib_length_mm)

        # Generate all rib points (pt); we use `c` to denote an arc length, to keep the notation consistent
        c_pt = np.arange(0, int(rib_length_pt) + 1)  # granularity is 1 pt
        half_rib_widths_pt = self._pdf_util.mm_to_pt(self.math.w_c(self._pdf_util.pt_to_mm(c_pt)) / 2)

        # Define box in which to draw each rib segment on each page
        box_origin = (margin, margin)
        box_width = page_width - 2 * margin
        box_height = page_height - 2 * margin

        # Break apart the points array into segments, one segment per page
        seg_size = box_height + 1
        segments = [half_rib_widths_pt[i: i + seg_size] for i in range(0, len(half_rib_widths_pt), seg_size)]
        if len(segments[-1]) == 1:
            # If the last segment has just a single point, then that single point will be covered by the box border,
            # leaving the last page of the PDF blank. So, we can just remove that point and eliminate the blank page.
            segments.pop()

        # Number of pages needed for the rib template (excluding the specs page)
        num_pages = len(segments)

        # Create a PDF with ReportLab
        pdf = self._pdf_util.initialize_canvas(output_pdf_path)

        # Generate first page, which is the specs page (model graph, dimensions, etc.)
        self.oud_specs.add_specs_page(pdf)

        # Iterate over pages and draw a segment of the rib on each page until the entire rib has been drawn
        x_mid = page_width / 2
        box_top = page_height - margin
        for page_idx in range(num_pages):
            page_num = page_idx + 1

            # Set PDF config for each page
            self._pdf_util.apply_default_pdf_config(pdf)

            # Draw box on page
            pdf.rect(*box_origin, box_width, box_height)

            # Write the number of ribs needed for the given specs
            pdf.drawCentredString(
                page_width / 2,
                page_height - (margin / 2) - text_line_height,
                f'Number of ribs: {self.num_ribs}'
            )

            # Write page number (specs page is not numbered, so we start numbering rib pages from 1)
            pdf.drawCentredString(page_width / 2, margin / 2, str(page_num))

            # Draw rib segment
            segment = segments[page_idx]
            for c in range(len(segment) - 1):  # we move 1 pt at a time, since the points granularity is 1 pt
                # hw = "half width"
                hw1 = segment[c]
                hw2 = segment[c + 1]

                # Convert the Cartesian coordinates to points on the PDF page (both left and right points).
                # Note that the rib is symmetric along its long axis.

                page_right_x1 = x_mid + hw1
                page_left_x1 = x_mid - hw1
                page_y1 = box_top - c

                page_right_x2 = x_mid + hw2
                page_left_x2 = x_mid - hw2
                page_y2 = box_top - (c + 1)

                # Draw both the right and left segments
                pdf.line(page_right_x1, page_y1, page_right_x2, page_y2)
                pdf.line(page_left_x1, page_y1, page_left_x2, page_y2)

            # If last page, draw the blunt end of the rib (the end at the neck-joint) using the last points
            if page_idx == num_pages - 1:
                pdf.line(page_right_x2, page_y2, page_left_x2, page_y2)

            # Create new page
            pdf.showPage()

        pdf.save()


class RibMath:
    def __init__(self, model, num_ribs):
        self.model = model  # oud bowl profile function (radius function)
        self.n = num_ribs  # number of ribs desired

        # Generate x distribution which will be used to generate tables
        #   - Resolution: 100 points per mm, capped at 1 million total points for performance (realistically should
        #     never reach 1 million points, unless oud soundboard is 10 meters long)
        resolution = min(math.ceil(self.model.H * 100), 1000000)
        x_distribution = np.linspace(0, self.model.H, resolution)

        # Generate arc length table and (arc length -> width) table

        x1 = x_distribution[:-1]
        x2 = x_distribution[1:]

        r1 = self.r(x1)
        r2 = self.r(x2)

        euclidean_distances = np.sqrt((x2 - x1) ** 2 + (r2 - r1) ** 2)

        arc_lengths = np.concatenate([[0], np.cumsum(euclidean_distances)])  # manually insert zero arc length at x=0
        widths = self.w_x(x_distribution)

        # Create tables
        self.l_table = np.column_stack([x_distribution, arc_lengths])
        self.l_w_table = np.column_stack([arc_lengths, widths])

    # Radius of bowl with respect to soundboard's long axis; simply the model's radius function
    def r(self, x):
        return self.model(x)

    # Arc length with respect to soundboard's long axis
    def l(self, x):
        return np.interp(x, self.l_table[:, 0], self.l_table[:, 1])

    # Width of rib with respect to soundboard's long axis
    def w_x(self, x):
        # 2r * tan(pi / 2n) gives us the "exterior chord", which is the true width of wood needed for the rib
        return 2 * self.r(x) * math.tan(math.pi / (2 * self.n))

    # Inverse arc length
    def l_inv(self, c):
        return np.interp(c, self.l_table[:, 1], self.l_table[:, 0])

    # Width of rib with respect to arc length
    def w_c(self, c):
        return np.interp(c, self.l_w_table[:, 0], self.l_w_table[:, 1])


class OudSpecs:
    def __init__(self, model):
        self.model = model

        # Set specs (part 1/2)
        self.full_length = model.H + model.NECK_LENGTH
        self.neck_joint_width = model.Z
        self.neck_length = model.NECK_LENGTH
        self.neck_thickness = model.Z / 2
        self.nut_width = model.Z * (2 / 3)
        self.soundboard_length = model.H

        # Generate model points
        x = np.linspace(0, model.H, 5000)
        y = model(x)
        points = np.column_stack((x, y))

        # Add neck joint point to guarantee its existence in extreme bowl curves
        neck_joint_point = np.array([self.soundboard_length, self.neck_thickness])
        points = np.vstack((points, neck_joint_point))

        # Add the neck point to the points (the farthest point up on the neck, where the nut is)
        neck_point = np.array([self.full_length, self.neck_thickness])
        points = np.vstack((points, neck_point))
        self._profile_points = points

        # Set specs (part 2/2)
        self.depth = max(y)
        self.soundboard_width = self.depth * 2

        self._pdf_util = PDFUtil()

    def add_specs_page(self, pdf):
        # Set PDF config for this page
        self._pdf_util.apply_default_pdf_config(pdf)

        self._draw_specs__header(pdf)  # draw header and footer
        self._draw_specs__profile_graph(pdf)  # draw graph of the oud profile
        self._draw_specs__face_diagram(pdf)  # draw diagram of the oud face

        # Create new page
        pdf.showPage()

    def make_specs_pdf(self, output_pdf_path):
        pdf = self._pdf_util.initialize_canvas(output_pdf_path)
        self.add_specs_page(pdf)
        pdf.save()

    def _draw_specs__header(self, pdf):
        page_width = self._pdf_util.page_width
        page_height = self._pdf_util.page_height
        margin = self._pdf_util.margin
        font_name = self._pdf_util.font_name
        font_size = self._pdf_util.font_size
        mono_font_name = self._pdf_util.mono_font_name
        text_line_height = self._pdf_util.text_line_height
        big_pad = self._pdf_util.big_pad
        tab = ' ' * 4

        # Write date
        date = datetime.now().strftime("%-d %b %Y")
        date_width = pdf.stringWidth(date, font_name, font_size)
        pdf.drawString(page_width - date_width - (margin / 2), page_height - (margin / 2), date)

        # Write model name and parameters

        pdf.saveState()
        leading = text_line_height + big_pad
        pdf.setFont(mono_font_name, font_size, leading=leading)

        # Find max char length (for a line of text)
        max_char_len = 1
        available_space = page_width - (2 * margin) - len(tab)
        while pdf.stringWidth('*' * max_char_len, mono_font_name, font_size) <= available_space:
            max_char_len += 1
        max_char_len -= 1

        # If needed, chop the model string into parts of appropriate length to be drawn on multiple lines
        model_str = str(self.model)
        model_str_parts = textwrap.wrap(model_str, max_char_len)
        if len(model_str_parts) == 0:
            model_str_parts.append('')

        # Write the model string (on multiple lines if needed)
        model_text = pdf.beginText(margin, page_height - margin)
        model_text.textLine(model_str_parts[0])
        for i, part in enumerate(model_str_parts[1:]):
            model_text.textLine(f'{tab}{part}')

        pdf.drawText(model_text)
        pdf.restoreState()

    def _draw_specs__profile_graph(self, pdf):
        page_width = self._pdf_util.page_width
        page_height = self._pdf_util.page_height
        text_line_height = self._pdf_util.text_line_height
        dash_config = self._pdf_util.dash_config
        thin_line_width = self._pdf_util.thin_line_width
        small_pad = self._pdf_util.small_pad
        big_pad = self._pdf_util.big_pad

        points_real = self._get_model_points()

        # Define graph width, and scale points accordingly
        graph_width_mm = 120
        points_graph = points_real * (graph_width_mm / self.full_length)  # scale
        points_graph = self._pdf_util.mm_to_pt(points_graph)  # convert mm to pt

        # Define lower left corner of graph drawing
        graph_width_pt = self._pdf_util.mm_to_pt(graph_width_mm)
        x_offset = (page_width - graph_width_pt) / 2
        y_offset = (page_height / 2) + 120

        # Get max depth in pt (on the graph's scale, not real life)
        max_depth_pt = max(points_graph[:, 1])

        # Draw graph axes
        pdf.line(x_offset, y_offset, x_offset + graph_width_pt, y_offset)  # x-axis
        # pdf.line(x_offset, y_offset, x_offset, y_offset + max_depth_pt + 40)  # y-axis

        # Draw model graph (profile of oud)
        for i in range(len(points_graph) - 1):
            x1, y1 = points_graph[i]
            x2, y2 = points_graph[i + 1]

            x1_page = x1 + x_offset
            x2_page = x2 + x_offset
            y1_page = y1 + y_offset
            y2_page = y2 + y_offset

            pdf.line(x1_page, y1_page, x2_page, y2_page)

        # Draw max depth dashed line
        argmax_depth_pt = np.argmax(points_graph[:, 1])  # index at which the max depth occurs
        x_max_depth = points_graph[:, 0][argmax_depth_pt] + x_offset
        pdf.saveState()
        pdf.setDash(*dash_config)
        pdf.setLineWidth(thin_line_width)
        pdf.line(x_max_depth, y_offset, x_max_depth, y_offset + max_depth_pt)
        pdf.restoreState()

        # Annotate max depth
        pdf.drawString(
            x_max_depth + small_pad,
            y_offset + (max_depth_pt / 2) - (text_line_height / 2),
            f'{round(self.depth)} mm'
        )

        # Draw neck thickness bracket
        x_neck_thickness_bracket = x_offset + graph_width_pt + big_pad
        neck_thickness_bracket_width = big_pad * 2
        neck_thickness_bracket_height = points_graph[-1, 1]
        pdf.saveState()
        pdf.setLineWidth(thin_line_width)
        self._pdf_util.draw_vertical_bracket(
            pdf,
            x=x_neck_thickness_bracket,
            y=y_offset,
            width=neck_thickness_bracket_width,
            height=neck_thickness_bracket_height,
            direction='right'
        )
        pdf.restoreState()

        # Annotate neck thickness
        pdf.drawString(
            x_neck_thickness_bracket + neck_thickness_bracket_width + small_pad,
            y_offset + (neck_thickness_bracket_height / 2) - (text_line_height / 2),
            f'{round(self.neck_thickness)} mm'
        )

        # Draw soundboard length bracket
        y_soundboard_length_bracket = y_offset - big_pad
        soundboard_length_bracket_width = points_graph[-2, 0]
        soundboard_length_bracket_height = big_pad * 2
        pdf.saveState()
        pdf.setLineWidth(thin_line_width)
        self._pdf_util.draw_horizontal_bracket(
            pdf,
            x=x_offset,
            y=y_soundboard_length_bracket,
            width=soundboard_length_bracket_width,
            height=soundboard_length_bracket_height,
            direction='down'
        )
        pdf.restoreState()

        # Annotate soundboard length
        pdf.drawCentredString(
            x_offset + (soundboard_length_bracket_width / 2),
            y_soundboard_length_bracket - soundboard_length_bracket_height - text_line_height - small_pad,
            f'{round(self.soundboard_length)} mm'
        )

    def _draw_specs__face_diagram(self, pdf):
        page_width = self._pdf_util.page_width
        font_name = self._pdf_util.font_name
        font_size = self._pdf_util.font_size
        text_line_height = self._pdf_util.text_line_height
        dash_config = self._pdf_util.dash_config
        thin_line_width = self._pdf_util.thin_line_width
        small_pad = self._pdf_util.small_pad
        big_pad = self._pdf_util.big_pad

        points_real = self._get_model_points()

        # Make the last point half of the nut width (as opposed to half of the neck joint width)
        points_real[-1, 1] = self.nut_width / 2

        # Define diagram height, and scale points accordingly
        diagram_height_mm = 100
        points_diagram = points_real * (diagram_height_mm / self.full_length)  # scale
        points_diagram = self._pdf_util.mm_to_pt(points_diagram)  # convert mm to pt

        # Define bottom center of diagram drawing
        x_offset = page_width / 2
        y_offset = 80

        # Draw soundboard's left edge
        for i in range(len(points_diagram) - 1):
            x1, y1 = points_diagram[i]
            x2, y2 = points_diagram[i + 1]

            x1_page = x_offset - y1
            y1_page = y_offset + x1
            x2_page = x_offset - y2
            y2_page = y_offset + x2

            pdf.line(x1_page, y1_page, x2_page, y2_page)

        # Draw soundboard's right edge
        for i in range(len(points_diagram) - 1):
            x1, y1 = points_diagram[i]
            x2, y2 = points_diagram[i + 1]

            x1_page = x_offset + y1
            y1_page = y_offset + x1
            x2_page = x_offset + y2
            y2_page = y_offset + x2

            pdf.line(x1_page, y1_page, x2_page, y2_page)

        # If the starting points of the two edges are not the same, connect them by a straight line
        if points_diagram[0, 1] != 0:
            x0, y0 = points_diagram[0]
            pdf.line(
                x_offset + y0,
                y_offset + x0,
                x_offset - y0,
                y_offset + x0
            )

        # Draw upper edge of nut
        x1_nut = x_offset - points_diagram[-1, 1]
        x2_nut = x_offset + points_diagram[-1, 1]
        y_nut = y_offset + points_diagram[-1, 0]
        pdf.line(x1_nut, y_nut, x2_nut, y_nut)

        # Draw nut bracket
        nut_bracket_height = big_pad * 2
        nut_width_pt = x2_nut - x1_nut
        pdf.saveState()
        pdf.setLineWidth(thin_line_width)
        self._pdf_util.draw_horizontal_bracket(
            pdf,
            x=x1_nut,
            y=y_nut + big_pad,
            width=nut_width_pt,
            height=nut_bracket_height,
            direction='up'
        )
        pdf.restoreState()

        # Annotate nut width
        pdf.drawCentredString(
            x_offset,
            y_nut + nut_bracket_height + big_pad + small_pad,
            f'{round(self.nut_width)} mm'
        )

        # Draw neck joint bracket
        nj_bracket_height = big_pad * 2
        diagram_H_pt = points_diagram[-2, 0]
        diagram_half_Z_pt = points_diagram[-2, 1]
        nj_width_pt = 2 * diagram_half_Z_pt
        pdf.saveState()
        pdf.setLineWidth(thin_line_width)
        self._pdf_util.draw_horizontal_bracket(
            pdf,
            x=x_offset - diagram_half_Z_pt,
            y=y_offset + diagram_H_pt - big_pad,
            width=nj_width_pt,
            height=nj_bracket_height,
            direction='down'
        )
        pdf.restoreState()

        # Annotate neck joint width
        pdf.drawCentredString(
            x_offset,
            y_offset + diagram_H_pt - big_pad - nj_bracket_height - small_pad - text_line_height,
            f'{round(self.neck_joint_width)} mm'
        )

        # Draw neck length markers and bracket
        x_neck_length_markers_left_end = x_offset - diagram_half_Z_pt - 40
        y_neck_length_lower_marker = y_offset + diagram_H_pt
        diagram_neck_length_pt = y_nut - y_neck_length_lower_marker
        neck_length_bracket_width = big_pad * 2
        pdf.saveState()
        pdf.setLineWidth(thin_line_width)
        self._pdf_util.draw_vertical_bracket(
            pdf,
            x=x_neck_length_markers_left_end - big_pad,
            y=y_neck_length_lower_marker,
            width=neck_length_bracket_width,
            height=diagram_neck_length_pt,
            direction='left'
        )
        pdf.setDash(*dash_config)
        pdf.line(
            x_neck_length_markers_left_end,
            y_neck_length_lower_marker,
            x_offset - diagram_half_Z_pt,
            y_neck_length_lower_marker
        )
        pdf.line(x_neck_length_markers_left_end, y_nut, x1_nut, y_nut)
        pdf.restoreState()

        # Annotate neck length
        neck_length_annotation = f'{round(self.neck_length)} mm'
        neck_length_annotation_width = pdf.stringWidth(neck_length_annotation, font_name, font_size)
        pdf.drawString(
            x_neck_length_markers_left_end - big_pad - neck_length_bracket_width - small_pad - neck_length_annotation_width,
            y_neck_length_lower_marker + (diagram_neck_length_pt / 2) - (text_line_height / 2),
            neck_length_annotation
        )

        # Draw soundboard width line
        diagram_soundboard_half_width = max(points_diagram[:, 1])
        x_soundboard_leftmost = x_offset - diagram_soundboard_half_width
        x_soundboard_rightmost = x_offset + diagram_soundboard_half_width
        argmax_soundboard_width = np.argmax(points_diagram[:, 1])
        y_soundboard_width_line = y_offset + points_diagram[argmax_soundboard_width, 0]
        pdf.saveState()
        pdf.setLineWidth(thin_line_width)
        pdf.setDash(*dash_config)
        pdf.line(x_soundboard_leftmost, y_soundboard_width_line, x_soundboard_rightmost, y_soundboard_width_line)
        pdf.restoreState()

        # Annotate soundboard width
        pdf.drawCentredString(
            x_offset,
            y_soundboard_width_line + small_pad,
            f'{round(self.soundboard_width)} mm'
        )

        # Draw soundboard length markers and bracket
        x_soundboard_length_markers_right_end = x_soundboard_rightmost + 15
        y_soundboard_length_upper_marker = y_offset + diagram_H_pt
        soundboard_length_bracket_width = big_pad * 2
        pdf.saveState()
        pdf.setLineWidth(thin_line_width)
        self._pdf_util.draw_vertical_bracket(
            pdf,
            x=x_soundboard_length_markers_right_end + big_pad,
            y=y_offset,
            width=soundboard_length_bracket_width,
            height=diagram_H_pt,
            direction='right'
        )
        pdf.setDash(*dash_config)
        pdf.line(
            x_soundboard_length_markers_right_end,
            y_soundboard_length_upper_marker,
            x_offset + diagram_half_Z_pt,
            y_soundboard_length_upper_marker
        )
        pdf.line(x_soundboard_length_markers_right_end, y_offset, x_offset, y_offset)
        pdf.restoreState()

        # Annotate soundboard length
        x_soundboard_length_annotation = x_soundboard_length_markers_right_end + big_pad + soundboard_length_bracket_width + small_pad
        y_soundboard_length_annotation = y_offset + (diagram_H_pt / 2) - (text_line_height / 2)
        pdf.drawString(
            x_soundboard_length_annotation,
            y_soundboard_length_annotation,
            f'{round(self.soundboard_length)} mm'
        )

    def _get_model_points(self):
        return self._profile_points.copy()
