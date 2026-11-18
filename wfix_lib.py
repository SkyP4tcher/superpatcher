import sys,os,struct
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from vbuf import *
from igzw import objects, deser
def fix(doc):
  dcs=[(s,o) for s,o,t in objects(doc) if t=='igModelDrawCallData']
  names=[]
  for s,o in dcs:
    v=struct.unpack_from('>I',doc.sections[s][2],o+8)[0]; ss,oo=deser(doc,v); st=doc.sections[ss][2]
    names.append(bytes(st[oo:st.index(b'\0',oo)]).decode())
  isbody=lambda nm: False
  body=sorted(set(x for x in names if isbody(x))); n=0; nv=0
  for nm,vb in zip(names,vertex_buffers(doc)):
    if isbody(nm): continue
    els={}
    for i in range(0,len(vb['elems']),12):
      e=vb['elems'][i:i+12]
      if e[0]==0x2c: break
      els[e[4]]=struct.unpack_from('>H',e,8)[0]
    s,o,_,_=mem(doc,vb['handle']); data=doc.sections[s][2]
    for k in range(vb['count']):
      b=o+k*vb['stride']; w=list(data[b+els[6]:b+els[6]+4]); ix=list(data[b+els[8]:b+els[8]+4])
      order=sorted(range(4),key=lambda j:-w[j])
      nw=[w[j] for j in order]; ni=[ix[j] for j in order]
      if nw!=w: nv+=1
      data[b+els[6]:b+els[6]+4]=bytes(nw); data[b+els[8]:b+els[8]+4]=bytes(ni)
    n+=1
  return body,n,nv
