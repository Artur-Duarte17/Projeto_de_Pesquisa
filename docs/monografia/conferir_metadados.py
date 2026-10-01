"""Consulta metadados Crossref, sem substituir leitura das fontes."""
from pathlib import Path
import json, urllib.request, urllib.parse, concurrent.futures
BASE=Path(__file__).resolve().parent
SOURCES={
'Smeulders2000':'10.1109/34.895972',
'Dubey2022':'10.1109/TCSVT.2021.3080920',
'Babenko2014':'10.1007/978-3-319-10590-1_38',
'He2016':'10.1109/CVPR.2016.90',
'Schroff2015':'10.1109/CVPR.2015.7298682',
'Deng2019':'10.1109/CVPR.2019.00482',
'Zhang2015':'10.1109/CVPR.2015.7299113',
'Oh2015':'10.1109/ICCV.2015.440',
'Li2016':'10.1109/CVPR.2016.145',
'Atrey2010':'10.1007/s00530-010-0182-0',
'Jegou2008':'10.1007/978-3-540-88682-2_24',
'Manning2008':'10.1017/CBO9780511809071',
'Costache2008':'10.1117/12.766652',
'Messina2025':'10.1007/978-3-031-88708-6_28',
}
def get(item):
 key,value=item
 url='https://api.crossref.org/works/'+urllib.parse.quote(value,safe='/') if value.startswith('10.') else 'https://api.crossref.org/works?rows=1&query.bibliographic='+urllib.parse.quote(value)
 try:
  req=urllib.request.Request(url,headers={'User-Agent':'MonografiaDocumental/1.0'})
  data=json.load(urllib.request.urlopen(req,timeout=40))['message']
  if 'items' in data: data=data['items'][0]
  return key,data
 except Exception as e: return key,{'error':str(e)}
if __name__=='__main__':
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool: result=dict(pool.map(get,SOURCES.items()))
 (BASE/'cache/crossref.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
 for k,d in result.items(): print(k,json.dumps({x:d.get(x) for x in ['title','author','published','container-title','DOI','page','volume','issue','publisher']},ensure_ascii=False))
