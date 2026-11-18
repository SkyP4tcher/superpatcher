"""Conversion of IGZ files from Wii U (cafe) to Xbox 360 (xenon)."""
import re, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from igzw import IgzDoc, split_strings, join_strings

LANGS_360 = ('es', 'it')
ALL_LANGS = ('da', 'de', 'en', 'es', 'fi', 'fr', 'it', 'mx', 'nl', 'no', 'pt', 'sv')
LNG_RE = re.compile(r'_([a-z]{2})(\.lng)?$')

def fix_tdep(doc):
    r = doc.rec('TDEP')
    if not r: return
    s = split_strings(r[2], r[1], 2)
    pairs = [(s[i], s[i + 1]) for i in range(0, len(s), 2)]
    keep = []
    for name, path in pairs:
        if path.endswith('.lng') or path.endswith('.pak'):
            m = LNG_RE.search(name)
            if m and m.group(1) in ALL_LANGS and m.group(1) not in LANGS_360:
                continue
        keep.append((name, path.replace('/BuildServer/cafe/', '/BuildServer/xenon/')))
    r[1] = len(keep)
    blob = join_strings([x for p in keep for x in p])
    r[2] = blob + b'\0' * (4 - len(blob) % 4)

def convert_generic(b):
    doc = IgzDoc(b)
    doc.platform = 5
    fix_tdep(doc)
    return doc

if __name__ == '__main__':
    pass

import struct
from igz import fnv1a
import tex

TEX_MAX = None
TEX_NOMIPS = False

def convert_textures(doc, name=''):
    ims = tex.image_info(doc)
    if not ims: return
    ex = doc.rec('EXID'); tm = doc.rec('TMHN')
    exb = bytearray(ex[2]); tmb = bytearray(tm[2])
    for im in ims:
        kind, plat = im['fmt']
        if plat != 'tile_cafe': raise ValueError('unexpected Wii U format ' + plat)
        bpb = tex.BPB[kind]
        sec = doc.sections[im['psec']][2]
        if not (im['poff'] == 0 and im['size'] == len(sec)):
            raise ValueError('pixels not in their own section')
        mips, _ = tex.gx2_untile(bytes(sec), im['w'], im['h'], im['levels'], bpb)
        linear = name.split('/')[-1].startswith('GuiStandard')
        w, h, lv = im['w'], im['h'], im['levels']
        # reduction to fit the 360 memory pools: the largest level is dropped
        while TEX_MAX and not linear and lv > 1 and max(w, h) > TEX_MAX:
            mips = mips[1:]; w = max(1, w // 2); h = max(1, h // 2); lv -= 1
        if TEX_NOMIPS and not linear and lv > 1:
            mips = mips[:1]; lv = 1
        if (w, h, lv) != (im['w'], im['h'], im['levels']):
            o = doc.sections[im['obj'][0]][2]
            struct.pack_into('>HH', o, im['obj'][1] + 0x34, w, h)
            struct.pack_into('>H', o, im['obj'][1] + 0x3A, lv)
            im = dict(im, w=w, h=h, levels=lv)
        if linear:
            assert im['levels'] == 1
            new = tex.xenos_linear(mips, im['w'], im['h'], bpb); suf = '_big_xenon'
        else:
            new = tex.xenos_tiled(mips, im['w'], im['h'], bpb); suf = '_tile_big_xenon'
        doc.sections[im['psec']][2] = bytearray(new)
        # pixel alignment: 8 KB on Wii U, 4 KB on Xbox 360 (also in the TMHN flag field)
        if doc.sections[im['psec']][1] == 0x2000: doc.sections[im['psec']][1] = 0x1000
        flags = 0x50000000 if im['flags'] == 0x58000000 else im['flags']
        struct.pack_into('>I', exb, im['exid_index'] * 8, fnv1a(kind + suf))
        struct.pack_into('>I', tmb, im['tmhn'] * 8, flags | len(new))
    ex[2] = bytes(exb); tm[2] = bytes(tmb)

def convert(b, name=''):
    doc = convert_generic(b)
    convert_textures(doc, name)
    return doc

import vbuf
from igzw import objects, deser, unpack_ints

# Alchemy element type -> Xenos D3DDECLTYPE; Alchemy usage -> D3DDECLUSAGE
DECL_TYPE = {0x02: 0x2a23b9, 0x05: 0x182886, 0x17: 0x1a2286, 0x1a: 0x1a2086, 0x23: 0x1a215a,
             0x28: 0x2a2187, 0x2a: 0x2c235f, 0x2b: 0x1a2360}
DECL_USAGE = {0: 0, 1: 3, 2: 6, 4: 10, 5: 5, 6: 1, 8: 2}
COLOR_WIIU, COLOR_360 = 0x04, 0x05

def xenos_decl(elems):
    out = bytearray()
    for i in range(0, len(elems), 12):
        e = elems[i:i + 12]
        if e[0] == 0x2c: break
        off = struct.unpack_from('>H', e, 8)[0]
        out += struct.pack('>HHIBBBB', e[1], off, DECL_TYPE[e[0]], 0, DECL_USAGE[e[4]], e[5], 0)
    out += bytes.fromhex('00ff000000ffffff000000ef')
    return bytes(out)

def handle_locs(doc):
    r = doc.rec('RMHN')
    return [deser(doc, v) for v in unpack_ints(r[2], r[1])] if r else []

# hashes of the vertex formats that differ on 360 (derived from pairs of files of both games)
VF_HASH = {(0x55316686, 'actors'): 0x5302e2a7, (0x1c4caaaa, 'actors'): 0xa468610f,
           (0x34eca639, 'models'): 0xdbde890c}

def convert_models(doc, name=''):
    objs = objects(doc)
    if not any(t == 'igVertexFormat' for _, _, t in objs): return
    tm = doc.rec('TMHN'); tmb = bytearray(tm[2])
    locs = handle_locs(doc)
    def handle_in(sec, off, size):
        for ls, lo in locs:
            if ls == sec and off <= lo < off + size:
                return struct.unpack_from('>I', doc.sections[sec][2], lo)[0]
        return None
    vbs = vbuf.vertex_buffers(doc)
    # 1. vertex data: color channels (Wii U A,B,G,R -> Xenos A,R,G,B)
    for vb in vbs:
        if vb['handle'] is None: continue
        s, o, size, flags = vbuf.mem(doc, vb['handle'])
        data = doc.sections[s][2]
        for i in range(0, len(vb['elems']), 12):
            e = vb['elems'][i:i + 12]
            if e[0] == 0x2c: break
            if e[0] == COLOR_WIIU:
                eo = struct.unpack_from('>H', e, 8)[0]
                for v in range(vb['count']):
                    p = o + v * vb['stride'] + eo
                    c = data[p:p + 4]
                    data[p:p + 4] = bytes((c[0], c[3], c[2], c[1]))
        if flags == 0x20000000:
            struct.pack_into('>I', tmb, vb['handle'] * 8, 0x10000000 | size)
    # 2. formats: color type and Xenos declaration in place of the GX2 block
    for sec, off, t in objs:
        if t != 'igVertexFormat': continue
        d = doc.sections[sec][2]
        es, ep = struct.unpack_from('>II', d, off + 0xC)
        ps, pp = struct.unpack_from('>II', d, off + 0x14)
        s1, o1 = deser(doc, ep); s2, o2 = deser(doc, pp)
        h = struct.unpack_from('>I', d, off + 0x38)[0]
        h2 = VF_HASH.get((h, name.split('/')[0]))
        if h2: struct.pack_into('>I', d, off + 0x38, h2)
        el = doc.sections[s1][2]
        n = es & 0x07FFFFFF
        for i in range(o1, o1 + n, 12):
            if el[i] == COLOR_WIIU: el[i] = COLOR_360
        decl = xenos_decl(bytes(el[o1:o1 + n]))
        old = ps & 0x07FFFFFF
        if len(decl) > old: raise ValueError('declaration larger than the GX2 block')
        pd = doc.sections[s2][2]
        pd[o2:o2 + old] = decl + b'\0' * (old - len(decl))
        struct.pack_into('>I', d, off + 0x14, (ps & ~0x07FFFFFF) | len(decl))
    # 3. index buffer: exact size and Xenos flags
    for sec, off, t in objs:
        if t != 'igIndexBuffer': continue
        d = doc.sections[sec][2]
        count = struct.unpack_from('>I', d, off + 8)[0]
        h = handle_in(sec, off, 0x34)
        if h is None: continue
        size, ptr = struct.unpack_from('>II', tmb, h * 8)
        if size >> 27 << 27 == 0x18000000:
            struct.pack_into('>I', tmb, h * 8, (count * 2 + 3) & ~3)
    tm[2] = bytes(tmb)
    for s in doc.sections:
        if s[0] == 0xf and s[1] == 64: s[1] = 16

def convert(b, name=''):
    doc = convert_generic(b)
    convert_textures(doc, name)
    convert_models(doc, name)
    return doc

import reloc

def _name(doc, sec, off):
    v = struct.unpack_from('>I', doc.sections[sec][2], off + 8)[0]
    s, o = deser(doc, v); st = doc.sections[s][2]
    return bytes(st[o:st.index(b'\0', o)])

def convert_materials(doc):
    objs = [o for o in objects(doc) if o[0] == 0]
    types = {o[1]: o[2] for o in objs}
    d = doc.sections[0][2]
    consts = [o[1] for o in objs if o[2] == 'igGraphicsShaderConstant']
    shared = [o for o in consts if struct.unpack_from('>I', d, o + 4)[0] > 1]
    ranges = []; redirect = {}
    for sec, off, t in objs:
        if not t.startswith('igShaderConstantValue') or t == 'igShaderConstantValueList': continue
        tgt = struct.unpack_from('>I', d, off + 8)[0]
        ts, to = deser(doc, tgt)
        if ts != 0 or types.get(to) != 'igGraphicsShaderConstant': continue
        if struct.unpack_from('>I', d, to + 4)[0] != 1: continue
        nm = _name(doc, 0, to)
        cands = [o for o in shared if o < off and _name(doc, 0, o) == nm]
        if not cands: continue
        redirect[to] = cands[-1]
        ranges.append((to, to + 16))
    if ranges:
        reloc.remove_ranges(doc, ranges, redirect)

def convert(b, name=''):
    doc = convert_generic(b)
    convert_textures(doc, name)
    convert_models(doc, name)
    return doc

def tmet_names(doc):
    r = doc.rec('TMET')
    return set(s.decode('latin1') for s in r[2].split(b'\0') if s)

from igzw import pack_ints as _pack

def _str_at(doc, v):
    s, o = deser(doc, v); st = doc.sections[s][2]
    return bytes(st[o:st.index(b'\0', o)])

def convert_package(doc):
    """igStringRefList of the packages: removes language files missing from the 360 game."""
    s0 = doc.sections[0][2]
    n = struct.unpack_from('>I', s0, 0x1C + 8)[0]
    arr = struct.unpack_from('>I', s0, 0x1C + 0x14)[0] & 0x07FFFFFF
    vals = [_str_at(doc, struct.unpack_from('>I', s0, arr + 4 * i)[0]) for i in range(n)]
    pairs = [(vals[i], vals[i + 1]) for i in range(0, n, 2)]
    keep = []
    for kind, path in pairs:
        m = LNG_RE.search(path.decode('latin1').rsplit('.', 1)[0])
        if kind == b'lang_file' and m and m.group(1) not in LANGS_360: continue
        keep.append((kind, path))
    nl = struct.unpack_from('>I', s0, arr + 4 * n + 0x14)[0] & 0x07FFFFFF
    name, hsh = struct.unpack_from('>II', s0, nl)
    pkgname = _str_at(doc, name)
    # string table: order of first appearance, each with terminator and padding to 2
    strs = bytearray(); where = {}
    def add(s):
        if s not in where:
            if len(strs) % 2: strs.append(0)
            where[s] = len(strs); strs.extend(s + b'\0')
        return 0x08000000 | where[s]
    ptrs = [add(x) for p in keep for x in p]
    namep = add(pkgname)
    cnt = len(ptrs)
    out = bytearray()
    out += struct.pack('>6I', 0, 0, 1, 1, 0x80000004, 0x18) + struct.pack('>I', 0x1C)
    out += struct.pack('>6I', 1, 1, cnt, cnt, 0x80000000 | 4 * cnt, 0x34)
    out += b''.join(struct.pack('>I', p) for p in ptrs)
    nlo = len(out)
    out += struct.pack('>6I', 2, 0, 1, 1, 0x80000008, nlo + 0x18) + struct.pack('>II', namep, hsh)
    doc.sections[0][2] = out
    doc.sections[1][2] = strs
    for r in doc.records:
        if r[0] == 'RVTB': locs = [0, 0x1C, nlo]
        elif r[0] == 'RSTR': locs = [0x34 + 4 * i for i in range(cnt)] + [nlo + 0x18]
        elif r[0] == 'ROFS': locs = [0x14, 0x18, 0x30, nlo + 0x14]
        else: continue
        body = _pack(locs); r[1] = len(locs); r[2] = body + b'\0' * ((-len(body)) % 4)
    on = doc.rec('ONAM'); on[2] = struct.pack('>I', nlo)

_convert_prev = convert
def convert(b, name=''):
    doc = _convert_prev(b, name)
    if tmet_names(doc) == {'igObjectList', 'igStringRefList', 'igNameList'}:
        convert_package(doc)
    return doc
