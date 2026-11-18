"""Alchemy textures: GX2 (Wii U) and Xenos (Xbox 360) tiling for DXT."""
import struct, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from igzw import IgzDoc, objects, deser
from igz import fnv1a

FMT = {}
for k in ('dxt1', 'dxt3', 'dxt5', 'dxn', 'r8g8b8a8', 'b8g8r8a8', 'a8r8g8b8', 'r5g6b5', 'b5g6r5', 'a8', 'l8', 'a8l8', 'l8a8', 'r4g4b4a4', 'a4r4g4b4', 'r16g16b16a16f', 'r32f', 'ati1', 'ati2', 'bc4', 'bc5', 'x8r8g8b8', 'r8', 'r8g8'):
    for suf in ('_tile_cafe', '_cafe', '_linear_cafe', '_tile_big_xenon', '_big_xenon', '_tile_xenon', '_xenon', '_srgb_tile_cafe', '_srgb_tile_big_xenon'):
        FMT[fnv1a(k + suf)] = (k, suf[1:])
BPB = {'dxt1': 8, 'dxt3': 16, 'dxt5': 16, 'dxn': 16}

def image_info(doc):
    out = []
    ex = doc.rec('EXID')
    exid = [struct.unpack_from('>II', ex[2], i * 8) for i in range(ex[1])] if ex else []
    tm = doc.rec('TMHN')
    tmhn = [struct.unpack_from('>II', tm[2], i * 8) for i in range(tm[1])] if tm else []
    for sec, off, t in objects(doc):
        if t != 'igImage2': continue
        d = doc.sections[sec][2]
        w, h, dep, lv = struct.unpack_from('>HHHH', d, off + 0x34)
        fh = struct.unpack_from('>I', d, off + 0x40)[0]
        pix = struct.unpack_from('>I', d, off + 0x4C)[0]
        fmt = FMT.get(exid[fh & 0x7FFFFFFF][0]) if fh & 0x80000000 else None
        size_flags, ptr = tmhn[pix]
        psec, poff = deser(doc, ptr)
        out.append(dict(obj=(sec, off), w=w, h=h, depth=dep, levels=lv, fmt=fmt, exid_index=fh & 0x7FFFFFFF,
                        tmhn=pix, size=size_flags & 0x07FFFFFF, flags=size_flags & ~0x07FFFFFF,
                        psec=psec, poff=poff))
    return out

# ---------------- GX2 (R600 addrlib), tile mode 4 (2D thin1) and 2 (1D thin1)
PIPES, BANKS, PIPE_INTERLEAVE = 2, 4, 256

def pixel_index(x, y, bpp):
    x0, x1, x2 = x & 1, (x >> 1) & 1, (x >> 2) & 1
    y0, y1, y2 = y & 1, (y >> 1) & 1, (y >> 2) & 1
    if bpp == 8: b = (x0, x1, x2, y1, y0, y2)
    elif bpp == 16: b = (x0, x1, x2, y0, y1, y2)
    elif bpp in (32, 96): b = (x0, x1, y0, x2, y1, y2)
    elif bpp == 64: b = (x0, y0, x1, x2, y1, y2)
    else: b = (y0, x0, x1, x2, y1, y2)
    return sum(v << i for i, v in enumerate(b))

def gx2_addr_micro(x, y, bpp, pitch, height):
    mtb = 64 * bpp // 8
    mto = mtb * ((x // 8) + (y // 8) * (pitch // 8))
    return mto + (bpp * pixel_index(x, y, bpp) >> 3)

def gx2_addr_macro(x, y, bpp, pitch, height, swizzle=0):
    elem = (bpp * pixel_index(x, y, bpp)) >> 3
    pipe = ((y >> 3) ^ (x >> 3)) & 1
    bank = (((y // 32) ^ (x >> 3)) & 1) | ((((y // 16) ^ (x >> 4)) & 1) << 1)
    bank_pipe = pipe + PIPES * bank
    pipe_sw = (swizzle >> 8) & 1; bank_sw = (swizzle >> 9) & 3
    bank_pipe ^= (pipe_sw + PIPES * bank_sw)
    bank_pipe %= PIPES * BANKS
    pipe = bank_pipe % PIPES; bank = bank_pipe // PIPES
    mtp, mth = 8 * BANKS, 8 * PIPES
    mtbytes = (bpp * mth * mtp + 7) >> 3
    mto = ((x // mtp) + (pitch // mtp) * (y // mth)) * mtbytes
    total = elem + (mto >> 3)
    off_high = (total & ~0xFF) << 3
    return off_high | (bank << 9) | (pipe << 8) | (total & 0xFF)

def gx2_levels(w, h, levels, bpb, force1d=False):
    """(blocks_w, blocks_h, pitch, aligned height, mode, offset, size) per level."""
    bpp = bpb * 8
    out = []; off = 0
    for lv in range(levels):
        lw = max(1, w >> lv); lh = max(1, h >> lv)
        bw = (lw + 3) // 4; bh = (lh + 3) // 4
        mode = 4
        if force1d or bw < 32 or bh < 16: mode = 2
        if mode == 4:
            pitch = (bw + 31) & ~31; ah = (bh + 15) & ~15; align = 2048
        else:
            pitch = (bw + 7) & ~7; ah = (bh + 7) & ~7; align = 256
        size = pitch * ah * bpb
        off = (off + align - 1) // align * align
        out.append((bw, bh, pitch, ah, mode, off, size))
        off += size
    return out

def gx2_layout(w, h, levels, bpb, size):
    for f1 in (False, True):
        lv = gx2_levels(w, h, levels, bpb, f1)
        end = lv[-1][5] + lv[-1][6]
        if end == size or (end <= size and size - end < 2048 and not f1):
            return lv
    raise ValueError('unknown GX2 layout %dx%d l%d size %x' % (w, h, levels, size))

def gx2_untile(data, w, h, levels, bpb, swizzle=0):
    lv = gx2_layout(w, h, levels, bpb, len(data))
    res = []
    for bw, bh, pitch, ah, mode, off, size in lv:
        out = bytearray(bw * bh * bpb)
        for y in range(bh):
            for x in range(bw):
                a = gx2_addr_macro(x, y, bpb * 8, pitch, ah, swizzle) if mode == 4 else gx2_addr_micro(x, y, bpb * 8, pitch, ah)
                s = off + a
                out[(y * bw + x) * bpb:(y * bw + x + 1) * bpb] = data[s:s + bpb]
        res.append(bytes(out))
    return res, lv

# ---------------- Xenos
def xenos_offset(x, y, pitch, log2_bytes):
    pitch = (pitch + 31) & ~31
    macro = ((x >> 5) + (y >> 5) * (pitch >> 5)) << (log2_bytes + 7)
    micro = ((x & 7) + ((y & 0xE) << 2)) << log2_bytes
    off = macro + ((micro & ~0xF) << 1) + (micro & 0xF) + ((y & 1) << 4)
    return (((off & ~0x1FF) << 3) + ((y & 16) << 7) + ((off & 0x1C0) << 2) +
            (((((y & 8) >> 2) + (x >> 3)) & 3) << 6) + (off & 0x3F))

def swap16(b):
    a = bytearray(b); a[0::2], a[1::2] = b[1::2], b[0::2]; return bytes(a)

def log2c(v):
    n = 0
    while (1 << n) < v: n += 1
    return n

def xenos_tiled(mips, w, h, bpb):
    """Linear levels -> tiled Xenos buffer with packed mip tail."""
    l2 = 3 if bpb == 8 else 4
    tile = 32 * 32 * bpb
    out = bytearray(); packed_base = None; packed_idx = 0
    for l, m in enumerate(mips):
        lw = max(1, w >> l); lh = max(1, h >> l)
        bw = (lw + 3) // 4; bh = (lh + 3) // 4
        if min(lw, lh) <= 16 and len(mips) > 1:
            if packed_base is None:
                packed_base = len(out); out += b'\0' * tile
            base = packed_base
            off = (16 >> packed_idx) // 4 if packed_idx < 3 else 0
            if log2c(lw) > log2c(lh): xo, yo = 0, off
            else: xo, yo = off, 0
            if packed_idx >= 3: xo = yo = 0
            packed_idx += 1
            pitch = 32
        else:
            base = len(out)
            tw = (bw + 31) // 32; th = (bh + 31) // 32
            out += b'\0' * (tile * tw * th)
            pitch = tw * 32; xo = yo = 0
        for y in range(bh):
            for x in range(bw):
                s = base + xenos_offset(x + xo, y + yo, pitch, l2)
                out[s:s + bpb] = swap16(m[(y * bw + x) * bpb:(y * bw + x + 1) * bpb])
    return bytes(out)

def xenos_linear(mips, w, h, bpb):
    m = mips[0]
    bw = (w + 3) // 4; bh = (h + 3) // 4
    pitch = (bw + 31) & ~31
    out = bytearray(pitch * bh * bpb)
    for y in range(bh):
        row = m[y * bw * bpb:(y + 1) * bw * bpb]
        out[y * pitch * bpb:y * pitch * bpb + len(row)] = swap16(row)
    return bytes(out)
