"""Multiple XMA2 streams (6-8-10 channel music): reading, checking and writing.

An FSB with more than 2 channels holds several XMA streams (stereo, or mono for an odd channel count)
in the same flow of 2048-byte packets. Conventions of the 360 game:
- blocks of 16 packets; in every block each stream has at least one packet and the same number of
  frames as the others; at the start of the block one packet per stream, in order, with the first
  frame at 0;
- the last byte of the packet header says how many packets to skip to find the next packet of the
  same stream (also across the block boundary);
- packets left empty at the end of a block are all zero; the last packet of each stream has the
  skip that would lead to the stream's position in a following block;
- chunk 6: number of streams, number of blocks + 2, 0, then the decoded samples at the end of each
  block (frames of the block * 512, the same for every stream).
"""
import struct, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xma
from xma import PKT, PB, BLOCK_PACKETS, SPF

def demux(data, ns):
    """Owner of every packet (None for empty ones) following the skips."""
    npk = len(data) // PKT
    hdr = [xma.packet_header(struct.unpack_from('>I', data, k * PKT)[0]) for k in range(npk)]
    owner = [None] * npk
    for s in range(min(ns, npk)):
        owner[s] = s
    for k in range(npk):
        s = owner[k]
        if s is None:
            if data[k * PKT:(k + 1) * PKT].strip(b'\0'):
                raise ValueError('packet %d has no stream and is not empty' % k)
            continue
        j = k + 1 + hdr[k][3]
        if j < npk:
            if owner[j] is not None and owner[j] != s:
                raise ValueError('packet %d assigned to two streams' % j)
            owner[j] = s
    return owner, hdr


def split_streams(data, ns):
    """For every stream: list of blocks, each as the bytes of its contiguous packets."""
    owner, hdr = demux(data, ns)
    out = [[] for _ in range(ns)]
    for b0 in range(0, len(owner), BLOCK_PACKETS):
        for s in range(ns):
            ks = [k for k in range(b0, min(b0 + BLOCK_PACKETS, len(owner))) if owner[k] == s]
            out[s].append(b''.join(data[k * PKT:(k + 1) * PKT] for k in ks))
    return out


def stream_frames(blocks, nch, rate):
    """Frames (xma.parse_frame dicts) of every block of a stream; the state carries across blocks."""
    offs = xma.band_offsets(rate)
    chs = [dict(reuse=0, saved=None, ti=0, step=1) for _ in range(nch)]
    res = []
    for bd in blocks:
        npk = len(bd) // PKT
        bits = ''.join(xma.bits_of(bd[k * PKT + 4:(k + 1) * PKT]) for k in range(npk))
        first = xma.packet_header(struct.unpack_from('>I', bd, 0)[0])[1]
        br = xma.BitReader(bits, first)
        frames = []
        while br.pos + 15 <= len(bits):
            try:
                f = xma.parse_frame(br, nch, offs, chs)
            except EOFError:
                break
            if f is None:
                k = (br.pos - 15) // PB + 1; found = False
                while k < npk:
                    h = xma.packet_header(struct.unpack_from('>I', bd, k * PKT)[0])
                    if h[0]:
                        br.pos = k * PB + h[1]; found = True; break
                    k += 1
                if not found: break
                continue
            frames.append(f)
        res.append(frames)
    return res


def stream_layout(nch_total):
    """Channels of every stream: stereo pairs, the last one mono for an odd channel count."""
    out = []; c = 0
    while c < nch_total:
        n = 2 if nch_total - c >= 2 else 1
        out.append(n); c += n
    return out


def _frame_starts(fbits):
    """Bit position (in the stream block) of the start of every frame, with the rules of xma.pack."""
    pos = 0; starts = []; per = {}
    for fb in fbits:
        k = pos // PB
        if per.get(k, 0) >= 63:
            pos = (k + 1) * PB; k += 1
        per[k] = per.get(k, 0) + 1
        starts.append(pos); pos += len(fb)
    return starts


def _pack_block(fbits):
    """Frames (bit strings) of a block of a stream -> (packets, time each packet is needed = index of
    the frame holding its first bit)."""
    data, counts = xma.pack(fbits)
    assert len(counts) == 1
    pk = [bytearray(data[k * PKT:(k + 1) * PKT]) for k in range(len(data) // PKT)]
    st = _frame_starts(fbits)
    need = [sum(1 for x in st if x <= k * PB) - 1 for k in range(len(pk))]
    return pk, need


def _npk(fbits):
    """Packets needed by a sequence of frames in a single block (None if it does not fit)."""
    pos = 0; starts = {}
    for fb in fbits:
        k = pos // PB
        if starts.get(k, 0) >= 63:
            pos = (k + 1) * PB; k += 1
        starts[k] = starts.get(k, 0) + 1
        pos += len(fb)
    return (pos + PB - 1) // PB


def interleave(streams_frames):
    """streams_frames: for every stream the list of frame bit strings (same count).
    Returns (data, frames per block)."""
    ns = len(streams_frames)
    nf = len(streams_frames[0])
    assert all(len(s) == nf for s in streams_frames)
    out_blocks = []; counts = []; i = 0
    while i < nf:
        # how many frames per stream fit in a 16-packet block
        lo, hi = 1, min(nf - i, 63 * BLOCK_PACKETS)
        def fits(n):
            return sum(_npk(s[i:i + n]) for s in streams_frames) <= BLOCK_PACKETS
        if not fits(1):
            raise ValueError('frames too large for a block')
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if fits(mid): lo = mid
            else: hi = mid - 1
        n = lo
        out_blocks.append([_pack_block(s[i:i + n]) for s in streams_frames]); counts.append(n); i += n
    # packet order: the first of every stream in order, then by the frame in which the decoder needs
    # it (on a tie, the stream with the lower index)
    seq = []
    for blk in out_blocks:
        order = [(s, 0) for s in range(ns)]
        order += sorted(((s, k) for s, (pk, nd) in enumerate(blk) for k in range(1, len(pk))),
                        key=lambda sk: (blk[sk[0]][1][sk[1]], sk[0]))
        seq.append([(s, blk[s][0][k]) for s, k in order])
    flat = []
    for bi, blk in enumerate(seq):
        for s, p in blk:
            flat.append((s, p))
        if bi + 1 < len(seq):
            for _ in range(BLOCK_PACKETS - len(blk)):
                flat.append((None, bytearray(PKT)))
    # skip: packets between this one and the next of the same stream
    for k, (s, p) in enumerate(flat):
        if s is None: continue
        j = k + 1
        while j < len(flat) and flat[j][0] != s: j += 1
        # the last packet of a stream points where the next block would start
        skip = j - k - 1 if j < len(flat) else len(flat) + s - k - 1
        if skip > 255: raise ValueError('skip above 255')
        h = struct.unpack_from('>I', p, 0)[0] & ~0xFF | skip
        struct.pack_into('>I', p, 0, h)
    return b''.join(bytes(p) for _, p in flat), counts


def seek_chunk(ns, counts):
    vals = [ns, len(counts) + 2, 0]; acc = 0
    for c in counts:
        acc += c * SPF; vals.append(acc)
    return struct.pack('>%dI' % len(vals), *vals)


def check(data, nch, rate=xma.RATE):
    """Re-reads a multistream, rewrites the frames and redoes the interleaving: (identical, frames per
    block). Also a test of the packet order, which depends only on the frames."""
    lay = stream_layout(nch)
    bl = split_streams(data, len(lay))
    offs = xma.band_offsets(rate)
    fr = [stream_frames(bl[s], lay[s], rate) for s in range(len(lay))]
    fb = [[xma.write_frame(f, lay[s], offs) for b in fr[s] for f in b] for s in range(len(lay))]
    out, counts = interleave(fb)
    return out == data, counts


if __name__ == '__main__':
    # python3 xma_ms.py check file.fsb ...  : identical re-read and rewrite of a multistream FSB
    import fsb
    for p in sys.argv[2:]:
        g = fsb.parse(open(p, 'rb').read())
        nch = dict(g['chunks']).get(1, bytes([g['channels']]))[0]
        ok, counts = check(g['data'], nch, g['rate'])
        c6 = seek_chunk(len(stream_layout(nch)), counts) == dict(g['chunks']).get(6)
        print('%s: %d channels, %s, chunk 6 %s, %d blocks' % (p, nch, 'identical' if ok else 'DIFFERENT',
                                                           'same' if c6 else 'DIFFERENT', len(counts)))
