import sys, os, struct
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from igzw import IgzDoc, objects, deser, unpack_ints

def tmhn(doc):
    r = doc.rec('TMHN')
    return [struct.unpack_from('>II', r[2], i * 8) for i in range(r[1])] if r else []

def rmhn_locs(doc):
    r = doc.rec('RMHN')
    return [deser(doc, v) for v in unpack_ints(r[2], r[1])] if r else []

def vertex_buffers(doc):
    """[(vb object, TMHN index of the data, vertex count, stride, elements)]"""
    objs = objects(doc)
    locs = rmhn_locs(doc)
    out = []
    for sec, off, t in objs:
        if t != 'igVertexBuffer': continue
        d = doc.sections[sec][2]
        count = struct.unpack_from('>I', d, off + 8)[0]
        fptr = struct.unpack_from('>I', d, off + 0x18)[0]
        fs, fo = deser(doc, fptr)
        fd = doc.sections[fs][2]
        stride = struct.unpack_from('>I', fd, fo + 8)[0]
        es, ep = struct.unpack_from('>II', fd, fo + 0xC)
        s1, o1 = deser(doc, ep)
        elems = bytes(doc.sections[s1][2][o1:o1 + (es & 0x07FFFFFF)])
        hidx = None
        for ls, lo in locs:
            if ls == sec and off <= lo < off + 0x34:
                hidx = struct.unpack_from('>I', d, lo)[0]
        out.append(dict(obj=(sec, off), handle=hidx, count=count, stride=stride, elems=elems, fmt=(fs, fo)))
    return out

def mem(doc, idx):
    size, ptr = tmhn(doc)[idx]
    s, o = deser(doc, ptr)
    return s, o, size & 0x07FFFFFF, size & ~0x07FFFFFF
