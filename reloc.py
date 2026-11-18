"""Removal of objects from pool 0 of an IGZ, updating every reference."""
import struct, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from igzw import unpack_ints, pack_ints, RELOC

def ser(sec, off):
    return (sec << 27) | off

def remove_ranges(doc, ranges, redirect):
    """ranges: [(start, end)] in pool 0; redirect: {old object offset: new target (old offset)}."""
    ranges = sorted(ranges)
    def removed(o):
        return any(s <= o < e for s, e in ranges)
    def shift(o):
        return o - sum(min(e, o) - s for s, e in ranges if s < o)
    def remap_target(sec, off):
        if sec != 0: return sec, off
        for s, e in ranges:
            if s <= off < e:
                if s not in redirect: raise ValueError('pointer to removed object without replacement %x' % s)
                off = redirect[s] + (off - s); break
        return 0, shift(off)
    lists = {}
    for r in doc.records:
        if r[0] in RELOC and r[0] != 'ONAM':
            lists[r[0]] = [((v >> 27), v & 0x07FFFFFF) for v in unpack_ints(r[2], r[1])]
    d0 = doc.sections[0][2]
    # pointer values (ROFS) before moving the data
    newvals = {}
    for sec, off in lists.get('ROFS', []):
        if sec == 0 and removed(off): continue
        data = doc.sections[sec][2]
        v = struct.unpack_from('>I', data, off)[0]
        ts, to = remap_target(v >> 27, v & 0x07FFFFFF)
        newvals[(sec, off)] = (v & 0xF8000000 if False else 0) | ser(ts, to)
    for (sec, off), v in newvals.items():
        struct.pack_into('>I', doc.sections[sec][2], off, v)
    # ONAM
    on = doc.rec('ONAM')
    if on:
        v = struct.unpack_from('>I', on[2], 0)[0]
        ts, to = remap_target(v >> 27, v & 0x07FFFFFF)
        on[2] = struct.pack('>I', ser(ts, to)) + on[2][4:]
    # position lists
    for r in doc.records:
        if r[0] in lists:
            locs = [(s, shift(o) if s == 0 else o) for s, o in lists[r[0]] if not (s == 0 and removed(o))]
            vals = sorted(ser(s, o) for s, o in locs)
            r[1] = len(vals)
            body = pack_ints(vals)
            r[2] = body + b'\0' * ((-len(body)) % 4 if True else 0)
    # pool 0 data
    nd = bytearray(); pos = 0
    for s, e in ranges:
        nd += d0[pos:s]; pos = e
    nd += d0[pos:]
    doc.sections[0][2] = nd
