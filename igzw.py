"""Reading and rewriting of IGZ v9 (big-endian) files record by record."""
import struct

ALIGN8 = {'EXNM', 'EXID', 'TMHN'}

class IgzDoc:
    def __init__(self, b):
        assert b[:4] == b'IGZ\x01'
        self.head = bytearray(b[:0x800])
        self.sections = []  # [pool, align, data]
        o = 0x14
        descs = []
        while True:
            name, off, ln, al = struct.unpack_from('>IIII', b, o)
            if off == 0: break
            descs.append((name, off, ln, al)); o += 16
        self.records = []  # [tag, count, data]
        nfix = struct.unpack_from('>I', b, 0x10)[0]
        p = descs[0][1]
        for _ in range(nfix):
            tag = b[p:p + 4][::-1].decode('latin1')
            cnt, ln, ds = struct.unpack_from('>III', b, p + 4)
            data = b[p + ds:p + ln]
            self.records.append([tag, cnt, data])
            p += ln
        for name, off, ln, al in descs[1:]:
            self.sections.append([name, al, bytearray(b[off:off + ln])])
        self.fix_pool, self.fix_align = descs[0][0], descs[0][3]
        self.raw_tail = {}

    @property
    def platform(self): return struct.unpack_from('>I', self.head, 0xC)[0]
    @platform.setter
    def platform(self, v): struct.pack_into('>I', self.head, 0xC, v)

    def rec(self, tag):
        for r in self.records:
            if r[0] == tag: return r
        return None

    def serialize(self, strip_trailing=False):
        out = bytearray(0x800)
        out[:0x800] = self.head
        start = 0x800
        fix = bytearray()
        for tag, cnt, data in self.records:
            p = start + len(fix)
            ds = 16
            if tag in ALIGN8 and (p + ds) % 8: ds += 4
            body = bytes(data)
            ln = ds + len(body)
            ln = (ln + 3) & ~3
            rec = bytearray(tag.encode('latin1')[::-1] + struct.pack('>III', cnt, ln, ds))
            rec += b'\0' * (ds - 16) + body
            rec += b'\0' * (ln - len(rec))
            fix += rec
        descs = [(self.fix_pool, start, len(fix), self.fix_align)]
        out += fix
        for name, al, data in self.sections:
            off = len(out)
            descs.append((name, off, len(data), al))
            out += data
        struct.pack_into('>I', out, 0x10, len(self.records))
        o = 0x14
        for d in descs:
            struct.pack_into('>IIII', out, o, *d); o += 16
        return bytes(out)

def split_strings(data, count, per=1):
    parts = data.split(b'\0')
    return [p.decode('latin1') for p in parts[:count * per]]

def join_strings(strs):
    return b''.join(s.encode('latin1') + b'\0' for s in strs)

def unpack_ints(data, count):
    """Nibble-packed integers (RVTB, ROFS, ...): increasing offsets, multiples of 4."""
    out = []; idx = 0; high = False; prev = 0
    def nib():
        nonlocal idx, high
        if not high:
            v = data[idx] & 0xF; high = True
        else:
            v = data[idx] >> 4; high = False; idx += 1
        return v
    for i in range(count):
        c = nib(); val = c & 7; sh = 3
        while c & 8:
            c = nib(); val |= (c & 7) << sh; sh += 3
        prev = prev + val * 4
        out.append(prev)
    return out

def deser(doc, v):
    """Serialized offset (pool in the high bits) -> (section index, offset)."""
    return (v >> 27), v & 0x07FFFFFF

def tmet(doc):
    r = doc.rec('TMET')
    return [s.decode('latin1') for s in r[2].split(b'\0') if s][:r[1]]

def objects(doc):
    r = doc.rec('RVTB')
    types = tmet(doc)
    out = []
    for v in unpack_ints(r[2], r[1]):
        sec, off = deser(doc, v)
        data = doc.sections[sec][2]
        t = int.from_bytes(data[off:off + 4], 'big')
        out.append((sec, off, types[t] if t < len(types) else '?%d' % t))
    return out

def pack_ints(vals):
    """Inverse of unpack_ints: increasing values, multiples of 4."""
    nibs = []; prev = 0
    for v in vals:
        d = (v - prev) // 4; prev = v
        while True:
            n = d & 7; d >>= 3
            if d: nibs.append(n | 8)
            else: nibs.append(n); break
    if len(nibs) % 2: nibs.append(0)
    out = bytes(nibs[i] | (nibs[i + 1] << 4) for i in range(0, len(nibs), 2))
    return out

RELOC = ('RVTB', 'RSTR', 'ROFS', 'RPID', 'REXT', 'RHND', 'RMHN', 'RNEX', 'ROOT', 'ONAM')
