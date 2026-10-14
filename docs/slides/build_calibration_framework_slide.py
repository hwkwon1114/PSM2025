"""Build a one-slide summary of the ewcal calibration framework with native
PowerPoint (OMML) equations.  LaTeX -> OMML via pandoc (docx), then injected
into python-pptx text boxes as a14:m math paragraphs.

Run:  PANDOC=/path/to/pandoc ~/miniforge/envs/DHD/bin/python docs/slides/build_calibration_framework_slide.py
"""
import copy, os, subprocess, tempfile, zipfile
from pathlib import Path
from lxml import etree
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

OUT = Path(__file__).with_name('calibration_framework_slide.pptx')
PANDOC = os.environ.get('PANDOC', 'pandoc')
NS = {'m': 'http://schemas.openxmlformats.org/officeDocument/2006/math',
      'a': 'http://schemas.openxmlformats.org/drawingml/2006/main',
      'a14': 'http://schemas.microsoft.com/office/drawing/2010/main',
      'mc': 'http://schemas.openxmlformats.org/markup-compatibility/2006'}
NAVY, GREY, ACCENT = RGBColor(0x1F, 0x3A, 0x5F), RGBColor(0x44, 0x44, 0x44), RGBColor(0x2E, 0x75, 0xB6)

# ---------------------------------------------------------------- content
# Each card: (number, title, [items]); item = ('t', text) or ('eq', latex, fallback)
CARDS = [
    ('1', 'Parameters & search space', [
        ('t', 'Four per-pass growth parameters of the bilayer shell model (top/bottom growth, top/bottom orthotropy):'),
        ('eq', r'\boldsymbol{\theta}=\left(g_{\mathrm{top}},\,g_{\mathrm{bot}},\,o_{\mathrm{top}},\,o_{\mathrm{bot}}\right)', 'θ = (g_top, g_bot, o_top, o_bot)'),
        ('t', 'Optimized in the unit cube x ∈ [0,1]⁴: logarithmic in growth, linear in orthotropy.'),
        ('eq', r'x_{g}=\frac{\ln g-\ln g_{\min}}{\ln g_{\max}-\ln g_{\min}},\qquad x_{o}=\frac{o+1}{2}', 'x_g = (ln g − ln g_min)/(ln g_max − ln g_min), x_o = (o+1)/2'),
        ('t', 'Bounds: g_top ∈ [5×10⁻⁵, 10⁻²],  g_bot ∈ [10⁻⁵, 10⁻³],  o_top, o_bot ∈ [−1, 1].'),
    ]),
    ('2', 'Forward model: zigzag toolpath → bilayer shell', [
        ('t', 'Run6: 68-strip zigzag pass repeated 29×. Each hit on face f grows the target metrics, with hit-count (Voce) decay:'),
        ('eq', r'q_{n}=e^{-\beta n},\quad \beta=\ln 1.25,\qquad g_{1,2}^{\ell}=q_{n}\,g_{\ell}\left(1\pm o_{\ell}\right),\quad \ell\in\{\mathrm{top},\mathrm{bot}\}',
         'q_n = exp(−βn), β = ln 1.25;  g_{1,2}^ℓ = q_n g_ℓ (1 ± o_ℓ)'),
        ('eq', r'\bar{\mathbf{a}}_{\ell}\leftarrow\mathbf{T}^{\mathsf{T}}\bar{\mathbf{a}}_{\ell}\,\mathbf{T},\quad \mathbf{T}=\mathbf{D}_{m}^{-1}\mathbf{R}_{\phi}\,\mathrm{diag}\left(1+g_{1}^{\ell},\,1+g_{2}^{\ell}\right)\mathbf{R}_{\phi}^{\mathsf{T}}\mathbf{D}_{m}',
         'ā_ℓ ← Tᵀ ā_ℓ T,  T = D_m⁻¹ R_φ diag(1+g₁, 1+g₂) R_φᵀ D_m'),
        ('t', 'After all passes, solve elastic equilibrium once (native hybrid solver, ‖∇E‖ ≤ 5×10⁻¹⁴):'),
        ('eq', r'\mathbf{x}^{*}(\boldsymbol{\theta})=\arg\min_{\mathbf{x}}\,E_{\mathrm{bilayer}}\left(\mathbf{x};\,\bar{\mathbf{a}}_{\mathrm{top}},\,\bar{\mathbf{a}}_{\mathrm{bot}}\right)',
         'x*(θ) = argmin_x E_bilayer(x; ā_top, ā_bot)'),
    ]),
    ('3', 'Shape score against the 3D scan', [
        ('t', 'Rigid frame set by the material reference, not fitted to the scan: Kabsch rotation of deformed onto flat reference vertices; area-weighted centroid removed; scan centered.'),
        ('eq', r'\mathbf{R}=\arg\min_{\mathbf{R}\in SO(3)}\sum_{j}\left\|\mathbf{R}\left(\mathbf{p}_{j}-\bar{\mathbf{p}}\right)-\left(\mathbf{X}_{j}-\bar{\mathbf{X}}\right)\right\|^{2}',
         'R = argmin_{R∈SO(3)} Σ_j ‖R(p_j − p̄) − (X_j − X̄)‖²'),
        ('t', 'Symmetric RMS surface discrepancy (scan → simulated surface, and area-weighted simulated vertices → scan):'),
        ('eq', r'L_{\mathrm{sym}}(\boldsymbol{\theta})=\sqrt{\frac{1}{2}\left[\frac{1}{N}\sum_{i=1}^{N}d\left(\mathbf{y}_{i},S_{\boldsymbol{\theta}}\right)^{2}+\sum_{j}w_{j}\min_{i}\left\|\mathbf{p}_{j}-\mathbf{y}_{i}\right\|^{2}\right]}',
         'L_sym = sqrt(½[(1/N)Σ d(y_i,S_θ)² + Σ w_j min_i ‖p_j − y_i‖²])'),
        ('t', 'yᵢ: N scan points (2 mm grid);  S_θ: simulated surface;  pⱼ: simulated vertices;  wⱼ = Aⱼ / Σₖ Aₖ (reference dual areas).'),
    ]),
    ('4', 'Batch Bayesian optimization (BoTorch)', [
        ('t', 'Gaussian-process surrogate of the log-misfit with a Matérn-5/2 ARD kernel:'),
        ('eq', r'f(\mathbf{x})=-\ln L_{\mathrm{sym}}(\mathbf{x})\sim\mathcal{GP}\left(\mu,\,k\right)', 'f(x) = −ln L_sym(x) ~ GP(μ, k)'),
        ('eq', r'k(\mathbf{x},\mathbf{x}^{\prime})=\sigma^{2}\left(1+\sqrt{5}\,r+\frac{5}{3}r^{2}\right)e^{-\sqrt{5}\,r},\quad r^{2}=\sum_{d=1}^{4}\frac{\left(x_{d}-x_{d}^{\prime}\right)^{2}}{\ell_{d}^{2}}',
         'k = σ²(1 + √5 r + 5r²/3) exp(−√5 r)'),
        ('t', 'Each wave proposes q = 4 parallel solves by maximizing batch log expected improvement (qLogEI):'),
        ('eq', r'\{\mathbf{x}_{1},\dots,\mathbf{x}_{4}\}=\arg\max\;\ln\mathbb{E}\left[\max_{j}\left(f(\mathbf{x}_{j})-f^{*}\right)_{+}\right]',
         '{x_1..x_4} = argmax ln E[max_j (f(x_j) − f*)_+]'),
        ('t', 'Failed/timed-out solves are excluded from the GP; proposals keep ‖Δx‖ ≥ 0.08; stop at the evaluation budget or 6 consecutive failures.'),
    ]),
]
FOOTER = ('Implementation: calibration/ewcal (runner.py, material_frame.py, geometry.py, batch_bayes.py); '
          'shell solver src/simulations/Sim_Bilayer_Growth.cpp.   Run6 continuation (in progress, 14 Oct 2026): '
          'best L_sym = 1.18 mm (Attempt 187).')

# ---------------------------------------------------------------- LaTeX -> OMML
def latex_to_omml(latexes):
    with tempfile.TemporaryDirectory() as d:
        md = Path(d, 'eq.md'); docx = Path(d, 'eq.docx')
        md.write_text('\n\n'.join(f'$${s}$$' for s in latexes) + '\n')
        subprocess.run([PANDOC, str(md), '-o', str(docx)], check=True)
        root = etree.fromstring(zipfile.ZipFile(docx).read('word/document.xml'))
    maths = root.findall('.//m:oMathPara/m:oMath', NS)
    if len(maths) != len(latexes):
        raise RuntimeError(f'pandoc returned {len(maths)} equations for {len(latexes)} inputs')
    return maths

def styled_math(omath, size_pt):
    """Copy an m:oMath and give every math run PowerPoint run properties."""
    om = copy.deepcopy(omath)
    for r in om.iter('{%s}r' % NS['m']):
        rpr = etree.Element('{%s}rPr' % NS['a'], lang='en-US', sz=str(int(size_pt * 100)))
        etree.SubElement(rpr, '{%s}solidFill' % NS['a']).append(
            etree.Element('{%s}srgbClr' % NS['a'], val='1A1A1A'))
        etree.SubElement(rpr, '{%s}latin' % NS['a'], typeface='Cambria Math')
        etree.SubElement(rpr, '{%s}cs' % NS['a'], typeface='Cambria Math')
        mrpr = r.find('m:rPr', NS)
        (mrpr.addnext(rpr) if mrpr is not None else r.insert(0, rpr))
    return om

def add_math_paragraph(txbody, omath, fallback, size_pt, space_pt=3):
    """PowerPoint layout: <a:p><a:pPr/><a14:m><m:oMathPara>...</m:oMathPara></a14:m></a:p>.
    The fallback text is stored as an attribute-free marker for wrap_math_shape()."""
    a = NS['a']
    p = etree.SubElement(txbody, '{%s}p' % a)
    ppr = etree.SubElement(p, '{%s}pPr' % a, algn='ctr')
    etree.SubElement(etree.SubElement(ppr, '{%s}spcBef' % a), '{%s}spcPts' % a, val=str(space_pt * 100))
    etree.SubElement(etree.SubElement(ppr, '{%s}spcAft' % a), '{%s}spcPts' % a, val=str(space_pt * 100))
    a14m = etree.SubElement(p, '{%s}m' % NS['a14'], nsmap={'a14': NS['a14']})
    para = etree.SubElement(a14m, '{%s}oMathPara' % NS['m'], nsmap={'m': NS['m']})
    etree.SubElement(etree.SubElement(para, '{%s}oMathParaPr' % NS['m']), '{%s}jc' % NS['m'],
                     {'{%s}val' % NS['m']: 'centerGroup'})
    para.append(styled_math(omath, size_pt))
    FALLBACKS.append((fallback, size_pt))
    return p

FALLBACKS = []  # plain-text fallbacks, in paragraph order, for the shape being built

def wrap_math_shape(sp_el):
    """Wrap a text-box shape containing a14:m equations the way PowerPoint saves them:
    mc:AlternateContent / Choice(a14) = shape with native math, Fallback = same shape
    with the equations replaced by plain text (PowerPoint itself uses a picture here)."""
    a = NS['a']
    fallback = copy.deepcopy(sp_el)
    texts = iter(FALLBACKS)
    for p_fb in fallback.findall('.//a:p', NS):
        m = p_fb.find('a14:m', NS)
        if m is not None:
            text, size_pt = next(texts)
            p_fb.remove(m)
            r = etree.SubElement(p_fb, '{%s}r' % a)
            etree.SubElement(r, '{%s}rPr' % a, lang='en-US', sz=str(int(size_pt * 100)), i='1')
            etree.SubElement(r, '{%s}t' % a).text = text
    parent = sp_el.getparent(); idx = parent.index(sp_el)
    ac = etree.Element('{%s}AlternateContent' % NS['mc'], nsmap={'mc': NS['mc']})
    choice = etree.SubElement(ac, '{%s}Choice' % NS['mc'], Requires='a14', nsmap={'a14': NS['a14']})
    parent.remove(sp_el); choice.append(sp_el)
    etree.SubElement(ac, '{%s}Fallback' % NS['mc']).append(fallback)
    parent.insert(idx, ac)
    FALLBACKS.clear()

def add_text_paragraph(tf, text, size_pt, first):
    p = tf.paragraphs[0] if first else tf.add_paragraph()
    p.alignment = PP_ALIGN.LEFT
    p.space_before = Pt(0 if first else 3)
    run = p.add_run(); run.text = text
    run.font.size = Pt(size_pt); run.font.color.rgb = GREY; run.font.name = 'Calibri'

# ---------------------------------------------------------------- slide
def build():
    latexes = [it[1] for _, _, items in CARDS for it in items if it[0] == 'eq']
    maths = iter(latex_to_omml(latexes))

    prs = Presentation(); prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    s = prs.slides.add_slide(prs.slide_layouts[6])

    tb = s.shapes.add_textbox(Inches(0.4), Inches(0.2), Inches(12.5), Inches(0.6)).text_frame
    r = tb.paragraphs[0].add_run(); r.text = 'Calibration framework: growth parameters from the final 3D scan'
    r.font.size, r.font.bold, r.font.color.rgb, r.font.name = Pt(24), True, NAVY, 'Calibri'
    sub = s.shapes.add_textbox(Inches(0.4), Inches(0.75), Inches(12.5), Inches(0.35)).text_frame
    r = sub.paragraphs[0].add_run()
    r.text = 'θ  →  growth-metric forward simulation  →  material-frame shape score vs. scan  →  GP surrogate + batch qLogEI  →  next θ'
    r.font.size, r.font.color.rgb, r.font.name = Pt(15), ACCENT, 'Calibri'

    W, H, gx, gy, x0, y0 = 6.15, 2.85, 0.23, 0.15, 0.4, 1.2
    eq_pt, txt_pt = 14, 11.5
    for k, (num, title, items) in enumerate(CARDS):
        x, y = x0 + (k % 2) * (W + gx), y0 + (k // 2) * (H + gy)
        card = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(W), Inches(H))
        card.adjustments[0] = 0.04
        card.fill.solid(); card.fill.fore_color.rgb = RGBColor(0xF5, 0xF8, 0xFC)
        card.line.color.rgb = RGBColor(0xC5, 0xD3, 0xE6); card.line.width = Pt(1)
        card.shadow.inherit = False
        badge = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x + 0.12), Inches(y + 0.1), Inches(0.36), Inches(0.36))
        badge.fill.solid(); badge.fill.fore_color.rgb = ACCENT; badge.line.fill.background(); badge.shadow.inherit = False
        bt = badge.text_frame; bt.margin_left = bt.margin_right = bt.margin_top = bt.margin_bottom = 0
        bt.vertical_anchor = MSO_ANCHOR.MIDDLE
        br = bt.paragraphs[0].add_run(); br.text = num; bt.paragraphs[0].alignment = PP_ALIGN.CENTER
        br.font.size, br.font.bold, br.font.color.rgb = Pt(14), True, RGBColor(0xFF, 0xFF, 0xFF)
        ht = s.shapes.add_textbox(Inches(x + 0.55), Inches(y + 0.07), Inches(W - 0.7), Inches(0.4)).text_frame
        hr = ht.paragraphs[0].add_run(); hr.text = title
        hr.font.size, hr.font.bold, hr.font.color.rgb, hr.font.name = Pt(16), True, NAVY, 'Calibri'

        body = s.shapes.add_textbox(Inches(x + 0.15), Inches(y + 0.5), Inches(W - 0.3), Inches(H - 0.55))
        tf = body.text_frame; tf.word_wrap = True
        tf.margin_left = tf.margin_right = Inches(0.05); tf.margin_top = tf.margin_bottom = 0
        first = True
        for it in items:
            if it[0] == 't':
                add_text_paragraph(tf, it[1], txt_pt, first)
            else:
                if first:  # drop the empty default paragraph
                    tf._txBody.remove(tf.paragraphs[0]._p)
                add_math_paragraph(tf._txBody, next(maths), it[2], eq_pt)
            first = False
        wrap_math_shape(body._element)

    ft = s.shapes.add_textbox(Inches(0.4), Inches(7.05), Inches(12.5), Inches(0.35)).text_frame
    ft.word_wrap = True
    fr = ft.paragraphs[0].add_run(); fr.text = FOOTER
    fr.font.size, fr.font.color.rgb, fr.font.name = Pt(9.5), RGBColor(0x70, 0x70, 0x70), 'Calibri'
    prs.save(OUT)
    print('wrote', OUT)


if __name__ == '__main__':
    build()
