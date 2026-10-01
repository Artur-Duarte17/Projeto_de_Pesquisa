"""Gera tabelas/figuras documentais de CSVs existentes, sem inferencia."""
from pathlib import Path
import csv, json, hashlib, statistics
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

B=Path(__file__).resolve().parent
R=B.parents[1]
RUN=R/'outputs/validation_runs/fechamento_20260928_cuda'
E=R/'docs/evidencias/fechamento_20260928_cuda'
F=B/'figuras'; F.mkdir(exist_ok=True)
T=B/'tabelas'; T.mkdir(exist_ok=True)
def read(p):
 with p.open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def fmt(x): return f'{float(x):.6f}'.replace('.',',')
agg=read(RUN/'gallagher/evaluation/fusion_metrics.csv')
per=read(RUN/'gallagher/evaluation/fusion_metrics_per_query.csv')
names={'face_only':'Somente face','global_context_only':'Somente global','fusion_0p9_0p1':'Fusão 0,9/0,1','fusion_0p7_0p3':'Fusão 0,7/0,3','fusion_0p5_0p5':'Fusão 0,5/0,5'}
base={x['query_id']:float(x['average_precision']) for x in per if x['method']=='face_only'}
assert len(per)==100 and len(base)==20
summaries=[]; deltas={}; local=[]
for a in agg:
 method=a['method']; subset=[p for p in per if p['method']==method]
 assert len(subset)==20 and {p['query_id'] for p in subset}==set(base)
 for col in ['average_precision','precision_at_5','precision_at_10','recall_at_5','recall_at_10']:
  assert abs(statistics.mean(float(p[col]) for p in subset)-float(a[col])) < 1e-12
 assert all(int(p['ranking_size'])==588 for p in subset)
 assert sum(int(p['num_relevant']) for p in subset)==884
 if method=='face_only': continue
 ds=[float(p['average_precision'])-base[p['query_id']] for p in subset]
 deltas[method]=ds
 summaries.append(dict(method=method,ganhos=sum(x>1e-12 for x in ds),empates=sum(abs(x)<=1e-12 for x in ds),perdas=sum(x< -1e-12 for x in ds),media=statistics.mean(ds),mediana=statistics.median(ds),minimo=min(ds),maximo=max(ds)))
 local.extend({'query_id':p['query_id'],'method':method,'delta_ap':d} for p,d in zip(subset,ds))
(B/'local/analise_pareada.json').write_text(json.dumps(local,indent=2),encoding='utf-8')
with (E/'comparacao_pareada_resumo.csv').open('w',encoding='utf-8',newline='') as f:
 w=csv.DictWriter(f,fieldnames=summaries[0].keys());w.writeheader();w.writerows(summaries)
lines=['\\begin{tabular}{lrrrrr}','\\toprule','Método & P@5 & P@10 & R@5 & R@10 & mAP \\\\','\\midrule']
for a in agg: lines.append(names[a['method']]+' & '+' & '.join(fmt(a[c]) for c in ['precision_at_5','precision_at_10','recall_at_5','recall_at_10','mAP'])+' \\\\')
lines+=['\\bottomrule','\\end{tabular}']
(T/'agregados.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
lines=['\\begin{tabular}{lrrrrr}','\\toprule','Método contra face & Ganhos & Empates & Perdas & Média $\\Delta$AP & Mediana \\\\','\\midrule']
for a in summaries: lines.append(names[a['method']]+f" & {a['ganhos']} & {a['empates']} & {a['perdas']} & {fmt(a['media'])} & {fmt(a['mediana'])} \\\\")
lines+=['\\bottomrule','\\end{tabular}']
(T/'pareados.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,ax=plt.subplots(figsize=(7.4,3.7))
ys=list(range(5)); vals=[float(a['mAP']) for a in agg]
ax.barh(ys,vals,color=['#205568','#899499','#487d8c','#66949e','#85aeb5'],height=.58)
ax.set_yticks(ys,[names[a['method']] for a in agg]);ax.invert_yaxis();ax.set_xlim(0,1.08);ax.set_xlabel('mAP — 20 consultas; pesos face/global')
for y,v in zip(ys,vals):ax.text(v+.012,y,fmt(v),va='center',fontsize=9)
ax.set_xticks([0,.2,.4,.6,.8,1]);ax.set_xticklabels(['0','0,2','0,4','0,6','0,8','1,0']);fig.tight_layout();fig.savefig(F/'map_gallagher.pdf');plt.close(fig)
fig,axs=plt.subplots(3,1,figsize=(7.4,5.0),sharex=True)
for ax,m in zip(axs,list(deltas)[1:]):
 ds=deltas[m]; counts={}
 for d in sorted(ds):
  key=round(d,12);counts[key]=counts.get(key,0)+1
 for d,n in counts.items(): ax.scatter([d]*n,list(range(1,n+1)),s=24,color='#205568',zorder=3)
 ax.axvline(0,color='#a54939',ls='--',lw=1);ax.set_ylabel(names[m].replace('Fusão ','')+'\ncontagem');ax.set_ylim(0,10);ax.set_yticks([1,5,9]);ax.grid(axis='x',alpha=.18)
axs[-1].set_xlabel('ΔAP por consulta = AP da fusão − AP facial (20 pontos em cada painel)')
axs[-1].set_xticks([-.8,-.6,-.4,-.2,0]);axs[-1].set_xticklabels(['−0,8','−0,6','−0,4','−0,2','0']);axs[-1].set_xlim(-.82,.085)
fig.tight_layout();fig.savefig(F/'deltas_pareados.pdf');plt.close(fig)
fig,ax=plt.subplots(figsize=(7.4,4.5));ax.set_xlim(0,10);ax.set_ylim(0,6);ax.axis('off')
boxes=[(0.2,4.55,2.7,1,'Fotografias locais\n(originais preservados)'),(3.55,4.55,2.7,1,'SCRFD + ArcFace\nrostos → vetores'),(7.0,4.55,2.7,1,'ResNet50\nimagem → vetor'),(.2,2.7,2.7,1,'Consulta\npessoa + imagem-fonte'),(3.55,2.7,2.7,1,'Índice facial\nmáximo por fotografia'),(7.0,2.7,2.7,1,'Índice global\ncosseno'),(3.55,.65,6.15,1.05,'Escores + pesos fixos → ranking de fotografias\nExclusão da fonte; avaliação contra anotações')]
for x,y,w,h,label in boxes:
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.05',fc='#edf3f4',ec='#205568',lw=1));ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=9)
ax.plot([1.55,1.55,8.35],[5.6,5.85,5.85],color='#205568',lw=1)
ax.annotate('',xy=(8.35,5.6),xytext=(8.35,5.85),arrowprops={'arrowstyle':'->','color':'#205568'})
ax.plot([1.55,1.55,8.35],[3.75,4.05,4.05],color='#205568',lw=1)
ax.annotate('',xy=(8.35,3.75),xytext=(8.35,4.05),arrowprops={'arrowstyle':'->','color':'#205568'})
for a,b in [((2.95,5.05),(3.5,5.05)),((4.9,4.5),(4.9,3.75)),((8.35,4.5),(8.35,4.1)),((2.95,3.2),(3.5,3.2)),((4.9,2.65),(4.9,1.75)),((8.35,2.65),(8.35,1.75))]:
 ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','color':'#205568'})
fig.tight_layout();fig.savefig(F/'fluxo_sistema.pdf');plt.close(fig)
provenance={'execution':'fechamento_20260928_cuda','inputs':{str(p.relative_to(R)):sha(p) for p in [RUN/'gallagher/evaluation/fusion_metrics.csv',RUN/'gallagher/evaluation/fusion_metrics_per_query.csv']},'operations':['medias recalculadas das 100 linhas','deltas pareados por query_id antes de retirar identificadores','contagens com tolerancia 1e-12','graficos sem fotos ou identificadores','diagrama conceitual baseado na arquitetura documentada'],'outputs':{str(p.relative_to(B)):sha(p) for p in list(T.glob('*.tex'))+list(F.glob('*.pdf'))},'checks':{'rows':100,'queries':20,'candidates':588,'relevance_relations':884},'summary':summaries}
(B/'PROVENIENCIA_FIGURAS_TABELAS.json').write_text(json.dumps(provenance,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print(json.dumps(summaries,indent=2))
