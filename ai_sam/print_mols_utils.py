
import io
import os

from cuspy import ConfigUtils
from tqdm import tqdm
import pandas as pd
from reportlab.graphics import renderPDF
from reportlab.pdfgen import canvas
from reportlab.platypus import Image
from svglib.svglib import svg2rlg
from rdkit.Chem import AllChem, Draw


class PrintMolsUtils:

    @classmethod
    def draw_mol(cls, smiles: str, ratio: float = 20):
        mol = AllChem.MolFromSmiles(smiles)
        AllChem.Compute2DCoords(mol)
        conf = mol.GetConformer()
        xs = []
        ys = []
        for i in range(mol.GetNumAtoms()):
            pos = conf.GetAtomPosition(i)
            xs.append(pos.x)
            ys.append(pos.y)
        width = max(max(xs) - min(xs), 1)
        height = max(max(ys) - min(ys), 1)
        draw = Draw.MolDraw2DSVG(int(width * ratio), int(height * ratio))
        draw.DrawMolecule(mol)
        draw.FinishDrawing()
        svg = draw.GetDrawingText()
        return svg, width * ratio, height * ratio

    @classmethod
    def draw_rxn(cls, rxn_smarts: str):
        rxn = AllChem.ReactionFromSmarts(rxn_smarts, useSmiles=True)
        rxn_svg = Draw.ReactionToImage(rxn, useSVG=True, subImgSize=(150, 150))

        width, height = cls.get_width_and_hight_from_svg(rxn_svg)
        return rxn_svg, width, height

    @classmethod
    def draw_mols(cls, smis: [str], mids: [int], temp_dp: str):
        svgs = []
        widths = []
        heights = []
        idxes = []
        for idx, (smi, mid) in enumerate(zip(smis, mids)):
            fig_fp = f'{mid}.svg'
            try:
                svg, width, height = cls.draw_mol(smi, ratio=20)
                with open(os.path.join(temp_dp, fig_fp), 'w', encoding='utf-8') as f:
                    f.write(svg)
            except:
                width = 100
                height = 100
                with open(os.path.join(temp_dp, fig_fp), 'w', encoding='utf-8') as f:
                    f.write('ERROR')
                svg = 'ERROR'
            idxes.append(mid)
            svgs.append(svg)
            widths.append(width)
            heights.append(height)

        return svgs, widths, heights

    @classmethod
    def get_line_mols_by_smis(cls, smis: [str], mids: [int], counts: [int], widths: [float], heights: [float],
                              svgs: [str], page_width: float):
        mols = []
        tot_width = 0
        for idx, (smi, mid, count, width, height, svg) in enumerate(zip(smis, mids, counts, widths, heights, svgs)):
            tot_width += width + 10
            if tot_width > page_width:
                yield mols
                mols = []
                tot_width = width + 10
            mol = {'id': mid, 'smiles': smi, 'svg': svg, 'width': width, 'height': height, 'count': count}
            mols.append(mol)
        if len(mols) > 0:
            yield mols

    @classmethod
    def get_page_mols_by_smis(cls, smis: [str], mids: [int], counts: [int], widths: [float], heights: [float],
                              svgs: [str], page_width: float, page_height: float):
        page_mols = []
        tot_height = 0
        for line_mols in cls.get_line_mols_by_smis(smis, mids, counts, widths, heights, svgs, page_width):
            heights = [mol['height'] for mol in line_mols]
            max_height = max(heights) + 20
            tot_height += max_height
            if tot_height > page_height:
                yield page_mols
                page_mols = []
                tot_height = max_height
            page_mols.append(line_mols)
        if len(page_mols) > 0:
            yield page_mols

    @classmethod
    def print_mols(cls, smis: [str], mids: [int], counts: [int], pdf_fp: str, temp_dp: str):
        page_width = 595.28
        page_height = 841.89
        c = canvas.Canvas(pdf_fp)
        c.setPageSize((page_width, page_height))
        svgs, widths, heights = cls.draw_mols(smis, mids, temp_dp)
        idx = 0
        for p, page_mols in enumerate(
                cls.get_page_mols_by_smis(smis, mids, counts, widths, heights, svgs, page_width, page_height)):
            height = 0
            # print(f"lines: {len(page_mols)}")
            c.setFont('Helvetica', 5)
            for line_mols in page_mols:
                width = 0
                for mol in line_mols:
                    # print(f"mols: {len(line_mols)}")
                    if mol['svg'] == 'ERROR':
                        c.drawString(width + 10, page_height - height - mol['height'] - 20, 'ERROR MOL')
                    else:
                        drawing = svg2rlg(os.path.join(temp_dp, f"{mol['id']}.svg"))
                        if drawing:
                            renderPDF.draw(drawing, c, width + 10, page_height - height - mol['height'] - 20)
                        else:
                            c.drawString(width + 10, page_height - height - mol['height'] - 20, 'ERROR SVG')
                    c.drawString(width + 10, page_height - height - 10, f"{idx}-{mol['id']}")
                    # c.drawString(width+10, page_height-height-20, f"{mol['count']}")
                    width += mol['width'] + 10
                    idx += 1
                heights = [mol['height'] for mol in line_mols]
                max_height = max(heights) + 20
                height += max_height
            c.showPage()
        c.save()
