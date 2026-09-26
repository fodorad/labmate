import pymupdf

from paper2carousel.steps.figures import extract_figures
from tests.conftest import make_pdf


def test_raster_and_vector_figures_are_cropped_with_captions(tmp_path):
    pdf = make_pdf(tmp_path / "p.pdf", figure=True)
    figures = extract_figures(pdf, tmp_path / "run" / "figures", tmp_path / "run")
    assert [(f.id, f.number, f.page) for f in figures] == [("fig1", 1, 1), ("fig2", 2, 1)]
    assert figures[0].caption == "Figure 1: A synthetic architecture diagram."
    assert figures[0].path == "figures/fig1.png"
    crop = pymupdf.Pixmap(str(tmp_path / "run" / figures[0].path))
    assert crop.width > crop.height  # 240x160 pt box, landscape
    # the "Encoder" label right above the image is included, the body text is not
    with pymupdf.open(pdf) as doc:
        text = doc[0].get_text(clip=_box_of(doc[0], crop, figures[0]))
    assert "Encoder" in text


def _box_of(page, crop, fig):
    # reconstruct the crop box from the image position: pixels / (dpi/72)
    from paper2carousel.steps.figures import DPI, PADDING

    w, h = crop.width * 72 / DPI, crop.height * 72 / DPI
    img = page.get_image_rects(page.get_images()[0][0])[0]
    return pymupdf.Rect(
        img.x0 - PADDING, img.y1 + PADDING - h, img.x0 - PADDING + w, img.y1 + PADDING
    )


def test_papers_without_figures(tmp_path):
    assert extract_figures(make_pdf(tmp_path / "p.pdf"), tmp_path / "f", tmp_path) == []


def test_captions_without_graphics_and_repeated_numbers_are_skipped(tmp_path):
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 100), "Figure 1: caption with nothing above it.", fontsize=9)
    page = doc.new_page()
    shape = page.new_shape()
    shape.draw_rect(pymupdf.Rect(100, 100, 300, 250))
    shape.finish(color=(0, 0, 0))
    shape.commit()
    page.insert_text((72, 270), "Figure 1: a later mention of the same figure.", fontsize=9)
    shape = page.new_shape()
    shape.draw_rect(pymupdf.Rect(100, 300, 110, 310))  # too small to be a figure
    shape.finish(color=(0, 0, 0))
    shape.commit()
    page.insert_text((72, 330), "Figure 3: tiny.", fontsize=9)
    doc.save(tmp_path / "p.pdf")
    figures = extract_figures(tmp_path / "p.pdf", tmp_path / "f", tmp_path)
    assert [(f.id, f.page) for f in figures] == [("fig1", 2)]
