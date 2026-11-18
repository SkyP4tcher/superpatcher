"""Writer for big-endian IGA v11 archives (Xbox 360 game format)."""
import struct, zlib, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from igz import fnv1a

BLOCK = 0x8000

def _deflate(chunk):
    c = zlib.compressobj(9, zlib.DEFLATED, -15)
    return c.compress(chunk) + c.flush()

def iga_hashes(path):
    """Hash of every entry, in file order (like iga_names)."""
    b = open(path, 'rb').read()
    n = struct.unpack_from('>I', b, 0xC)[0]
    return list(struct.unpack_from('>%dI' % n, b, 0x38))

def iga_names(path):
    """Name table entries of an IGA: [(build path, logical name)] in file order."""
    b = open(path, 'rb').read()
    n = struct.unpack_from('>I', b, 0xC)[0]
    nto = struct.unpack_from('>Q', b, 0x28)[0]; nts = struct.unpack_from('>I', b, 0x30)[0]
    nt = b[nto:nto + nts]
    out = []
    for i in range(n):
        o = struct.unpack_from('>I', nt, 4 * i)[0]
        e = nt.index(b'\0', o); e2 = nt.index(b'\0', e + 1)
        out.append((nt[o:e].decode('latin1'), nt[e + 1:e2].decode('latin1')))
    return out

def write_iga(path, files, sector=0x20, flags=1):
    """files: [(logical name, build path, data, 'zlib'|'stored')] in the order of the data in the archive
    (for an existing pak: offset order, see entry_list)."""
    n = len(files)
    # hash: FNV-1a of the lowercase logical name; in sound bank archives (flag 3) only of the last
    # part of the name (checked on every flag 3 pak of the 360 and Wii U games)
    key = (lambda nm: nm.split('/')[-1].lower()) if flags == 3 else (lambda nm: nm.lower())
    hfun = lambda i: files[i][4] if len(files[i]) > 4 and files[i][4] is not None else fnv1a(key(files[i][0]))
    hashed = sorted(range(n), key=lambda i: (hfun(i), i))
    small, medium, large = [], [], []
    blobs = {}; modes = {}
    for i, (name, build, data, kind) in enumerate(f[:4] for f in files):
        if kind == 'stored':
            blobs[i] = data; modes[i] = 0xFFFFFFFF; continue
        secs = (len(data) + sector - 1) // sector
        table = small if secs <= 0x7F else (medium if secs <= 0x7FFF else large)
        lim = 0x7F if table is small else (0x7FFF if table is medium else 0x7FFFFFFF)
        flag = lim + 1
        start = len(table)
        body = bytearray()
        for p in range(0, max(len(data), 1), BLOCK):
            chunk = data[p:p + BLOCK]
            comp = _deflate(chunk)
            sec_idx = len(body) // sector
            if len(comp) + 2 < len(chunk):
                body += struct.pack('<H', len(comp)) + comp
                table.append(flag | sec_idx)
            else:
                body += chunk
                table.append(sec_idx)
            body += b'\0' * ((-len(body)) % sector)
        table.append(len(body) // sector)  # end sector, as in the game files
        blobs[i] = bytes(body); modes[i] = (1 << 28) | start
    toc = n * 4 + n * 16 + len(large) * 4 + len(medium) * 2 + len(small)
    pos = 0x38 + toc
    pos += (-pos) % sector
    offsets = {}
    for i in range(n):
        offsets[i] = pos
        pos += len(blobs[i]); pos += (-pos) % sector
    nto = pos
    names = bytearray(); noffs = []
    for i in hashed:
        noffs.append(4 * n + len(names))
        names += files[i][1].encode('latin1') + b'\0' + files[i][0].encode('latin1') + b'\0' + b'\0' * 4
    nt = b''.join(struct.pack('>I', o) for o in noffs) + names
    hashes = [hfun(i) for i in hashed]
    div = 0xFFFFFFFF // n if n else 0
    slop = max((abs(h // div - k) for k, h in enumerate(hashes)), default=0)
    out = bytearray()
    out += struct.pack('>IIIIIIIIIIQII', 0x1A414749, 11, toc, n, sector, div, slop,
                       len(large), len(medium), len(small), nto, len(nt), flags)
    out += b''.join(struct.pack('>I', h) for h in hashes)
    # first field of the entry: position (in the hash-sorted table) of the entry that follows in the data,
    # << 8; 0xFFFFFF00 for the last one. Checked on every pak of the 360 game.
    slot = {i: k for k, i in enumerate(hashed)}
    for i in hashed:
        nxt = (slot[i + 1] << 8) if i + 1 < n else 0xFFFFFF00
        out += struct.pack('>IIII', nxt, offsets[i], len(files[i][2]), modes[i])
    out += b''.join(struct.pack('>I', v) for v in large)
    out += b''.join(struct.pack('>H', v) for v in medium)
    out += bytes(small)
    for i in range(n):
        out += b'\0' * (offsets[i] - len(out)); out += blobs[i]
    out += b'\0' * (nto - len(out)); out += nt
    open(path, 'wb').write(out)

def replace_iga(src, out, new_data):
    """Rewrites an IGA keeping its header, hashes, ordinals and names; replaces only the data.

    new_data: {logical name: bytes}. Entries not listed stay as they were. Uncompressed entries only
    (the sound bank paks, flag 3, where the hash is that of the last part of the name).
    """
    b = open(src, 'rb').read()
    n = struct.unpack_from('>I', b, 0xC)[0]
    sector = struct.unpack_from('>I', b, 0x10)[0]
    nto = struct.unpack_from('>Q', b, 0x28)[0]; nts = struct.unpack_from('>I', b, 0x30)[0]
    eo = 0x38 + n * 4
    ents = [list(struct.unpack_from('>IIII', b, eo + i * 16)) for i in range(n)]
    names = iga_names(src)
    if any(e[3] != 0xFFFFFFFF for e in ents):
        raise ValueError('replace_iga only handles archives with uncompressed entries')
    out_b = bytearray(b[:min(e[1] for e in ents)])
    for i in sorted(range(n), key=lambda i: ents[i][1]):
        data = new_data.get(names[i][1], b[ents[i][1]:ents[i][1] + ents[i][2]])
        out_b += b'\0' * ((-len(out_b)) % sector)
        ents[i][1] = len(out_b); ents[i][2] = len(data)
        out_b += data
    out_b += b'\0' * ((-len(out_b)) % sector)
    new_nto = len(out_b)
    out_b += b[nto:nto + nts]
    for i, e in enumerate(ents):
        struct.pack_into('>IIII', out_b, eo + i * 16, *e)
    struct.pack_into('>Q', out_b, 0x28, new_nto)
    open(out, 'wb').write(out_b)


def entry_list(path):
    """Entries of an IGA in data order: [(logical name, build path, data, uncompressed)]."""
    import iga
    b = open(path, 'rb').read()
    n = struct.unpack_from('>I', b, 0xC)[0]
    ents = [struct.unpack_from('>IIII', b, 0x38 + n * 4 + i * 16) for i in range(n)]
    files, data = iga.read_iga(path)
    names = iga_names(path)
    order = sorted(range(n), key=lambda i: ents[i][1])
    return [(names[i][1], names[i][0], data(files[i]), ents[i][3] == 0xFFFFFFFF) for i in order]


def check_chain(path):
    """True if the first field of the entries follows the data order as on the 360 game."""
    b = open(path, 'rb').read()
    n = struct.unpack_from('>I', b, 0xC)[0]
    ents = [struct.unpack_from('>IIII', b, 0x38 + n * 4 + i * 16) for i in range(n)]
    byoff = sorted(range(n), key=lambda i: ents[i][1])
    return all(ents[byoff[k]][0] == ((byoff[k + 1] << 8) if k + 1 < n else 0xFFFFFF00) for k in range(n))
