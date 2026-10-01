"""Verificacoes documentais e renderizacao; nao executa inferencia ou testes."""
from pathlib import Path
import hashlib, json, re, subprocess
from pypdf import PdfReader
import pypdfium2 as pdfium
from PIL import Image, ImageDraw

B=Path(__file__).resolve().parent
R=B.parents[1]
BUILD=B/'build'
PDF=BUILD/'main.pdf'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    texts={str(p.relative_to(B)):p.read_text(encoding='utf-8') for p in B.rglob('*.tex') if 'build' not in p.parts}
    tex='\n'.join(texts.values())
    bib=(B/'referencias.bib').read_text(encoding='utf-8')
    keys=set(re.findall(r'@\w+\{([^,]+),',bib))
    cited={k.strip() for group in re.findall(r'\\cite(?:online)?\{([^}]+)\}',tex) for k in group.split(',')}
    assert cited==keys, (cited-keys,keys-cited)
    refs=set(re.findall(r'\\ref\{([^}]+)\}',tex)); labels=set(re.findall(r'\\label\{([^}]+)\}',tex))
    assert refs<=labels,refs-labels
    log=(BUILD/'main.log').read_text(encoding='utf-8',errors='replace')
    assert not re.search(r'Overfull|Undefined control|Missing character|undefined|Rerun to get|multiply defined',log), 'Diagnostico LaTeX pendente'
    reader=PdfReader(str(PDF)); pages=[p.extract_text() or '' for p in reader.pages]
    pdftext='\n\n'.join(pages)
    assert '??' not in pdftext
    assert not re.search(r'C:\\|OneDrive|OneDrie|api[_ -]?key|query_[0-9]',pdftext,re.I)
    assert all(len(t.strip())>20 for t in pages), 'Pagina quase vazia'
    # Same 72 original files and bytes, not merely existence checks.
    initial=json.loads((B/'local/inventario_inicial.json').read_text(encoding='utf-8'))
    for f in initial:
        p=R/f['path']; assert p.stat().st_size==f['bytes'] and sha(p)==f['sha256'],f['path']
    run=R/'outputs/validation_runs/fechamento_20260928_cuda'
    assert len([p for p in run.rglob('*') if p.is_file()])==len(initial)
    prov=json.loads((B/'PROVENIENCIA_FIGURAS_TABELAS.json').read_text(encoding='utf-8'))
    for p,h in prov['inputs'].items(): assert sha(R/p)==h,p
    for p,h in prov['outputs'].items(): assert sha(B/p)==h,p
    (BUILD/'texto_pdf.txt').write_text(pdftext,encoding='utf-8')
    render=BUILD/'paginas';render.mkdir(exist_ok=True)
    doc=pdfium.PdfDocument(str(PDF))
    thumbs=[]
    for i in range(len(doc)):
        page=doc[i];bitmap=page.render(scale=1.5);im=bitmap.to_pil().convert('RGB')
        im.save(render/f'pagina_{i+1:02d}.png')
        im.thumbnail((565,800))
        card=Image.new('RGB',(595,840),'#dfe3e7');card.paste(im,((595-im.width)//2,30))
        ImageDraw.Draw(card).text((15,8),f'PDF pagina {i+1}',fill='black');thumbs.append(card)
        bitmap.close();page.close()
    for start in range(0,len(thumbs),4):
        sheet=Image.new('RGB',(1190,1680),'#dfe3e7')
        for j,im in enumerate(thumbs[start:start+4]):sheet.paste(im,((j%2)*595,(j//2)*840))
        sheet.save(render/f'contato_{start//4+1:02d}.png')
    doc.close()
    result={'pdf_sha256':sha(PDF),'pages':len(pages),'references':len(keys),'original_files_unchanged':len(initial),'original_bytes':sum(f['bytes'] for f in initial),'latex_unresolved_or_overfull':False,'rendered_pages':len(pages),'visual_review':'pending human-facing assistant inspection; rendering alone is not review','page_text_lengths':[len(x) for x in pages]}
    (BUILD/'verificacao.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
