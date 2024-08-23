import math
import os
import textwrap
from concurrent.futures import as_completed, ProcessPoolExecutor
from datetime import datetime

import numpy as np
from scipy.integrate import quad

from util import PDFUtil, Stopwatch, Style


class RibProcedure:
    def __init__(self, model, num_ribs, *, multiprocessing=True, verbose=False):
        self.model = model
        self.num_ribs = num_ribs
        self.math = RibMath(model, num_ribs)

        # Runtime config
        self.multiprocessing = multiprocessing
        self.verbose = verbose
        self.print = print if verbose else (lambda *_, **__: None)

        # Data required in the future
        self.x_distribution = None

        # Specs object for the model being used
        self.oud_specs = OudSpecs(model)

        # PDF config and utility functions
        self._pdf_util = PDFUtil()

    # Entry point
    def run(self, output_pdf_path):
        self.print('_' * 75)
        self.print(Style('Running rib procedure\n').bold().cyan())
        self.print(f'model = {self.model}')
        self.print(f'num_ribs = {self.num_ribs}')
        self.print()
        self.print(Style(f'multiprocessing = {self.multiprocessing}').light_gray())
        self.print(Style(f'verbose = {self.verbose}').light_gray())
        self.print('_' * 75)
        self.print()

        stopwatch = Stopwatch()
        stopwatch.start()

        # Setup

        if self.x_distribution is None:
            self._build_x_distribution()

        if self.math.l_inv_table_sorted is None:
            stopwatch.lap()
            self._build_l_inv_table()
            self.print(f'[{stopwatch.lap():.3f} s]')
        else:
            self.print(Style(f'Using cached inverse arc length table [size: {len(self.math.l_inv_table_sorted)}]').italics().gray())

        self.print()

        # Draw rib to scale in a PDF
        stopwatch.lap()
        self._make_rib_template_pdf(output_pdf_path)
        self.print(f'[{stopwatch.lap():.3f} s]')

        self.print(f'\nDone. See file {Style(output_pdf_path).bold().green()}')
        self.print(f'Total runtime: {stopwatch.total_elapsed():.3f} s')
        self.print()

    # Data creation ----------------------------------------------------------------------------------------------------

    def _build_x_distribution(self):
        H = self.model.H
        init_delta = 1e-12

        # Define slope thresholds and their respective interval masses
        slope_mass_pairs = [
            (1000000, 10000),
            (100, 1000),
        ]
        max_mass = slope_mass_pairs[0][1]

        # Initialize start and end distributions
        x_distributions = [
            np.linspace(0, init_delta, max_mass),
            np.linspace(H - init_delta, H, max_mass)
        ]

        # Generate dense distribution around start
        delta = init_delta
        for slope, mass in slope_mass_pairs:
            while abs(self.model.derivative(delta)) > slope:
                x_distributions.append(np.linspace(delta, delta * 10, mass))
                delta *= 10
        start = delta

        # Generate dense distribution around end
        delta = init_delta
        for slope, mass in slope_mass_pairs:
            while abs(self.model.derivative(H - delta)) > slope:
                x_distributions.append(np.linspace(H - delta * 10, H - delta, mass))
                delta *= 10
        end = H - delta

        # Generate sparse distribution in middle section
        x_distributions.append(np.linspace(start, end, max_mass))

        self.x_distribution = np.unique(np.concatenate(x_distributions))  # np.unique() sorts too

    def _build_l_inv_table(self):
        self.print(Style(f'Building inverse arc length table [size: ~{len(self.x_distribution)}]').blue())

        # Setup logging
        checkpoint = step_size = 10  # percent completion per log message

        def log_progress(num_points):
            nonlocal checkpoint
            completion_percent = num_points / len(self.x_distribution) * 100
            if completion_percent >= checkpoint:
                self.print(f'   {num_points} point{"" if num_points == 1 else "s"} generated [{round(completion_percent)}%]')
                checkpoint += step_size

        l_inv_table = {}

        if self.multiprocessing:
            num_cores = os.cpu_count() or 4
            self.print(Style(f'(using multiprocessing with {num_cores} cores)').light_gray())

            # Split x distribution into as many chunks as there are CPU cores (if it's sufficiently large)
            if len(self.x_distribution) > 100:
                x_chunks = [self.x_distribution[i::num_cores] for i in range(num_cores)]
            else:
                x_chunks = [self.x_distribution]

            with ProcessPoolExecutor(max_workers=num_cores) as executor:
                # Submit all jobs and get a list of Future objects
                futures = [executor.submit(self.math.compute_l_inv_table_entries, x_chunk) for x_chunk in x_chunks]

                self.print('Running processes in parallel...')
                for future in as_completed(futures):
                    l_inv_table.update(future.result())
                    log_progress(len(l_inv_table))
        else:
            for x in self.x_distribution:
                l_inv_table[self.math.l(x)] = x
                log_progress(len(l_inv_table))

        self.math.set_l_inv_table_sorted(l_inv_table)

    # PDF creation -----------------------------------------------------------------------------------------------------

    def _make_rib_template_pdf(self, output_pdf_path):
        self.print(Style(f'Drawing rib on PDF [output path: {output_pdf_path}]').blue())

        # Get PDF config
        page_width = self._pdf_util.page_width
        page_height = self._pdf_util.page_height
        margin = self._pdf_util.margin
        text_line_height = self._pdf_util.text_line_height

        rib_length_mm = self.math.l(self.model.H)
        rib_length_pt = self._pdf_util.mm_to_pt(rib_length_mm)

        self.print(Style(f'rib length: {rib_length_mm:.1f} mm ({rib_length_pt:.1f} pt)').light_gray())

        # Define box in which to draw each rib segment on each page
        box_origin = (margin, margin)
        box_width = page_width - 2 * margin
        box_height = page_height - 2 * margin

        # Number of pages needed for the rib template, i.e. excludes the first page (specs page)
        num_pages = round(rib_length_pt // box_height + 1)

        # Create a PDF with ReportLab
        pdf = self._pdf_util.initialize_canvas(output_pdf_path)

        # Generate first page, which is the specs page (model graph, dimensions, etc.)
        self.print(f'    creating page 1/{num_pages + 1}')
        self.oud_specs.add_specs_page(pdf)

        # Iterate over pages and draw a segment of the rib on each page until the entire rib has been drawn
        for page_idx in range(num_pages):
            page_num = page_idx + 1
            self.print(f'    creating page {page_num + 1}/{num_pages + 1}')  # accounts for specs page

            # Set PDF config for each page
            self._pdf_util.apply_default_pdf_config(pdf)

            # Draw box on page
            pdf.rect(*box_origin, box_width, box_height)

            # Write the total number of ribs needed for the given specs
            pdf.drawCentredString(
                page_width / 2,
                page_height - (margin / 2) - text_line_height,
                f'Total number of ribs: {self.num_ribs}'
            )

            # Write page number (specs page is not numbered, so we start numbering rib pages from 1)
            pdf.drawCentredString(page_width / 2, margin / 2, str(page_num))

            # Generate points for rib segment
            start_x = page_idx * box_height
            points = []
            for x in range(start_x, start_x + box_height + 1):  # step by 1 pt
                if x > rib_length_pt:
                    break

                x_mm = self._pdf_util.pt_to_mm(x)
                half_rib_width = 0.5 * self.math.omega(x_mm)
                y = self._pdf_util.mm_to_pt(half_rib_width)

                points.append((x, y))

            # Draw rib segment
            x_mid = page_width / 2
            box_top = page_height - margin
            for i in range(len(points) - 1):
                x1, y1 = points[i]
                x2, y2 = points[i + 1]

                # Convert the Cartesian coordinates to points on the PDF page (both left and right points).
                # Note that the rib is symmetric along its long axis.

                page_right_x1 = x_mid + y1
                page_left_x1 = x_mid - y1
                page_y1 = box_top - i

                page_right_x2 = x_mid + y2
                page_left_x2 = x_mid - y2
                page_y2 = box_top - (i + 1)

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
    def __init__(self, model, n, l_inv_table_sorted=None):
        self.model = model  # oud bowl profile function
        self.n = n  # number of ribs desired
        self.l_inv_table_sorted = l_inv_table_sorted  # sorted inverse arc length table (normally generated later)

    # Arc length of bowl with respect to long axis of oud
    def l(self, x):
        integrand = lambda t: math.sqrt(1 + self.model.derivative(t) ** 2)
        # Alert on error greater than 0.1 mm
        return quad(integrand, 0, x, epsabs=0.1)[0]

    # Width of rib with respect to long axis of oud
    def w(self, x):
        return 2 * self.model(x) * math.sin(math.pi / (2 * self.n))

    # Inverse arc length
    def l_inv(self, l):
        if self.l_inv_table_sorted is None:
            raise ValueError('`l_inv_table_sorted` must be set before calling `l_inv` (use `set_l_inv_table_sorted()`)')
        return np.interp(l, self.l_inv_table_sorted[:, 0], self.l_inv_table_sorted[:, 1])

    # Width of rib with respect to bowl arc length
    def omega(self, l):
        return self.w(self.l_inv(l))

    def compute_l_inv_table_entries(self, x_values):
        return {self.l(x): x for x in x_values}

    def set_l_inv_table_sorted(self, l_inv_table):
        if isinstance(l_inv_table, dict):
            l_inv_table = list(l_inv_table.items())
        l_inv_table = np.array(l_inv_table)
        l_inv_table_sorted = l_inv_table[l_inv_table[:, 0].argsort()]
        self.l_inv_table_sorted = l_inv_table_sorted


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

        # Add the neck point to the points (the farthest point up on the neck, where the nut is)
        neck_point = np.array([self.full_length, self.neck_thickness])
        points = np.row_stack((points, neck_point))
        self._profile_points = points

        # Set specs (part 2/2)
        self.depth = max(y)
        self.soundboard_width = self.depth * 2

        self._pdf_util = PDFUtil()

    def add_specs_page(self, pdf):
        # Set PDF config for this page
        self._pdf_util.apply_default_pdf_config(pdf)

        self._draw_specs__header_and_footer(pdf)  # draw header and footer
        self._draw_specs__profile_graph(pdf)  # draw graph of the oud profile
        self._draw_specs__face_diagram(pdf)  # draw diagram of the oud face

        # Create new page
        pdf.showPage()

    def make_specs_pdf(self, output_pdf_path):
        pdf = self._pdf_util.initialize_canvas(output_pdf_path)
        self.add_specs_page(pdf)
        pdf.save()

    def _draw_specs__header_and_footer(self, pdf):
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
        date = datetime.now().strftime("%d %b %Y")
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
        pdf.line(x_offset, y_offset, x_offset, y_offset + max_depth_pt + 40)  # y-axis

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

        # Draw neck thickness dashed line
        pdf.setDash(*dash_config)
        pdf.line(
            x_offset + graph_width_pt,
            y_offset,
            x_offset + graph_width_pt,
            y_offset + neck_thickness_bracket_height
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
