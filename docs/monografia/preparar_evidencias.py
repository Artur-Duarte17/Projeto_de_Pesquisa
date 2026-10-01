"""Deriva evidencias documentais; nao executa modelos nem altera resultados."""
from pathlib import Path
import csv, hashlib, json, statistics

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
RUN = ROOT / 'outputs/validation_runs/fechamento_20260928_cuda'
DEST = ROOT / 'docs/evidencias/fechamento_20260928_cuda'

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''): h.update(b)
    return h.hexdigest()

def rows(p):
    with p.open(encoding='utf-8-sig', newline='') as f: return list(csv.DictReader(f))

def main():
    DEST.mkdir(parents=True, exist_ok=True)
    (BASE/'local').mkdir(exist_ok=True)
    expected='5fc6acd8c42c75520c0eecefa5366f3a33e0627d217b0b0cf61dd556876f34c9'
    assert sha(RUN/'final_validation_manifest.json') == expected
    inventory=[{'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(RUN.rglob('*')) if p.is_file()]
    previous=BASE/'local/inventario_inicial.json'
    if previous.exists():
        assert json.loads(previous.read_text(encoding='utf-8')) == inventory, 'Resultados originais alterados'
    else: previous.write_text(json.dumps(inventory,indent=2),encoding='utf-8')
    sources=[]
    for rel in ['gallagher/evaluation/fusion_metrics.csv','lfw/evaluation/face_metrics.csv','holidays/evaluation/global_metrics.csv']:
        p=RUN/rel
        target=DEST/p.name
        target.write_bytes(p.read_bytes())
        sources.append({'source':p.relative_to(ROOT).as_posix(),'sha256':sha(p),'copy':target.name,'transformation':'copia identica, metricas agregadas apenas'})
    manifest=json.loads((RUN/'final_validation_manifest.json').read_text())
    derived={'notice':'Extrato documental derivado; nao e o manifesto original. Caminhos absolutos e entradas privadas omitidos.', 'original_manifest_sha256':expected,'git':manifest['git'],'runtime':manifest['runtime'],'configuration':{k:v for k,v in manifest['configuration'].items() if k!='run_root'},'sources':sources,'preserved_run_files':len(inventory),'preserved_run_bytes':sum(x['bytes'] for x in inventory)}
    (DEST/'proveniencia_derivada.json').write_text(json.dumps(derived,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps({'files':len(inventory),'bytes':sum(x['bytes'] for x in inventory),'aggregates':{x['copy']:rows(DEST/x['copy']) for x in sources}},ensure_ascii=False,indent=2))

if __name__=='__main__': main()
