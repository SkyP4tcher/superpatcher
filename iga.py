"""Reader for IGA v11 archives (big-endian, Wii U)."""
import struct, zlib, sys, os

def read_iga(path):
    b = open(path, 'rb').read()
    ver, toc, n, ss = struct.unpack_from('>IIII', b, 4)
    nl, nm, nsm = struct.unpack_from('>III', b, 0x1C)
    nto = struct.unpack_from('>Q', b, 0x28)[0]
    nts = struct.unpack_from('>I', b, 0x30)[0]
    eo = 0x38 + n * 4
    entries = [struct.unpack_from('>IIII', b, eo + i * 16) for i in range(n)]
    bo = eo + n * 16
    large = struct.unpack_from('>%dI' % nl, b, bo); bo += nl * 4
    med = struct.unpack_from('>%dH' % nm, b, bo); bo += nm * 2
    small = b[bo:bo + nsm]
    # names: table of u32 offsets then strings (build path, logical path)
    nt = b[nto:nto + nts]
    offs = struct.unpack_from('>%dI' % n, nt, 0)
    names = []
    for o in offs:
        e = nt.index(b'\0', o); s1 = nt[o:e]
        e2 = nt.index(b'\0', e + 1); s2 = nt[e + 1:e2]
        names.append(s2.decode('latin1') if s2 else s1.decode('latin1'))
    def block(idx, ln):
        secs = (ln + ss - 1) // ss
        if secs <= 0x7F: v = small[idx]; return v & 0x7F, bool(v >> 7)
        if secs <= 0x7FFF: v = med[idx]; return v & 0x7FFF, bool(v >> 15)
        v = large[idx]; return v & 0x7FFFFFFF, bool(v >> 31)
    files = []
    for i, (a, off, ln, mode) in enumerate(entries):
        files.append((names[i], off, ln, mode))
    def data(f):
        name, off, ln, mode = f
        if mode == 0xFFFFFFFF:
            return b[off:off + ln]
        comp = mode >> 28
        bi = mode & 0x0FFFFFFF
        out = bytearray(); k = 0
        while len(out) < ln:
            sec, c = block(bi + k, ln)
            p = off + sec * ss
            want = min(0x8000, ln - len(out))
            if c:
                sz = struct.unpack_from('<H', b, p)[0]
                if comp == 1:
                    out += zlib.decompressobj(-15).decompress(b[p + 2:p + 2 + sz])
                elif comp == 2:
                    import lzma
                    pr = b[p + 2:p + 7]
                    lc = pr[0] % 9; lp = (pr[0] // 9) % 5; pb = pr[0] // 45
                    dsz = int.from_bytes(pr[1:5], 'little')
                    dec = lzma.LZMADecompressor(lzma.FORMAT_RAW, filters=[{'id': lzma.FILTER_LZMA1, 'dict_size': dsz, 'lc': lc, 'lp': lp, 'pb': pb}])
                    out += dec.decompress(b[p + 7:p + 7 + sz], max_length=want)
                else:
                    raise ValueError('compression %d' % comp)
            else:
                out += b[p:p + want]
            k += 1
        return bytes(out[:ln])
    return files, data

if __name__ == '__main__':
    files, data = read_iga(sys.argv[1])
    out = sys.argv[2] if len(sys.argv) > 2 else None
    for f in files:
        d = data(f)
        if out:
            p = os.path.join(out, f[0].replace(':', '_').replace('`', '_'))
            os.makedirs(os.path.dirname(p), exist_ok=True)
            open(p, 'wb').write(d)
    print(len(files), 'files')
