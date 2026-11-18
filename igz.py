"""Big-endian IGZ v9 parser: sections and fixup records."""
import struct

class Igz:
    def __init__(self, b):
        self.b = b
        assert b[:4] == b'IGZ\x01'
        self.ver, self.hash, self.plat, self.nfix = struct.unpack_from('>IIII', b, 4)
        self.sections = []
        o = 0x14
        while True:
            name, off, ln, al = struct.unpack_from('>IIII', b, o)
            if off == 0: break
            self.sections.append((name, off, ln, al)); o += 16
        self.fix = {}
        p = self.sections[0][1]
        for _ in range(self.nfix):
            tag = b[p:p + 4][::-1].decode('latin1')
            cnt, ln, ds = struct.unpack_from('>III', b, p + 4)
            self.fix[tag] = (cnt, b[p + ds:p + ln])
            p += ln
    def strings(self, tag):
        if tag not in self.fix: return []
        cnt, d = self.fix[tag]
        out = []; i = 0
        for _ in range(cnt):
            e = d.index(b'\0', i); out.append(d[i:e].decode('latin1'))
            i = e + 1
            if tag in ('TSTR', 'TMET', 'EXNM') and (i & 1): i += 1
        return out
    def ptr(self, v):
        pool, off = v >> 27, v & 0x07FFFFFF
        return self.sections[pool + 1][1] + off

def fnv1a(s):
    h = 0x811c9dc5
    for c in s.encode():
        h = ((h ^ c) * 0x01000193) & 0xFFFFFFFF
    return h

FORMATS = {}
for base in ('dxt1', 'dxt3', 'dxt5', 'dxn', 'r8g8b8a8', 'b8g8r8a8', 'a8r8g8b8', 'r5g6b5', 'a8', 'l8', 'a8l8', 'r4g4b4a4', 'b5g6r5', 'ati1', 'ati2', 'bc4', 'bc5', 'l8a8', 'r8', 'r8g8', 'rgba8888', 'r16g16b16a16f', 'r32f', 'a4r4g4b4', 'x8r8g8b8', 'etc1', 'l4a4', 'dxt1a'):
    for suf in ('', '_big_xenon', '_xenon', '_tile_cafe', '_cafe', '_big_cafe', '_tile_xenon', '_big', '_tile', '_big_wii', '_ps3', '_big_ps3', '_tile_big_cafe', '_linear_cafe', '_srgb_tile_cafe', '_srgb_big_xenon'):
        FORMATS[fnv1a(base + suf)] = base + suf

def exid(g):
    cnt, d = g.fix['EXID']
    return [struct.unpack_from('>II', d, i * 8) for i in range(cnt)]

def objects(g):
    cnt, d = g.fix['RVTB']
    # nibble-packed offsets (as in igLibrary): delta * 4 per object
    out = []; pos = 0; shift = 0; i = 0
    nib = []
    for byte in d:
        nib += [byte & 0xF, byte >> 4]
    acc = 0; sh = 0; k = 0; prev = 0; res = []
    while len(res) < cnt and k < len(nib):
        n = nib[k]; k += 1
        acc |= (n & 7) << sh; sh += 3
        if not (n & 8):
            prev += acc * 4 if not res else acc * 4
            res.append(prev); acc = 0; sh = 0
    return res
