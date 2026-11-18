"""FSB5 files of the sound banks: reading, writing and IMA ADPCM re-encoding (FMOD/Xbox variant).

Every "Compressed/wavs/..." entry of a bank pak is an FSB5 with a single sample.
IMA format used by FMOD (codec 7), checked on the Wii U files:
- 36-byte block per channel: little endian int16 (first sample), step index (u8), a zero byte,
  then 32 bytes = 64 nibbles, low nibble first;
- the block gives 64 samples: the header one plus 63 nibbles, the 64th nibble is always 0;
- stereo: 72-byte block, the two headers side by side and then groups of 4 bytes alternating
  left/right;
- more than 2 channels (6): block of 36 * channels bytes with the first int16 samples of every channel,
  then the 16-bit step indexes, then groups of 2 bytes in turn per channel (checked on the music of a
  driver challenge: the last sample of each block joins the first of the next block);
- the sample count written in the header is blocks * 64, the data is aligned to 32 bytes.
"""
import struct, sys
import numpy as np

STEP = [7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 19, 21, 23, 25, 28, 31, 34, 37, 41, 45, 50, 55, 60, 66,
        73, 80, 88, 97, 107, 118, 130, 143, 157, 173, 190, 209, 230, 253, 279, 307, 337, 371, 408,
        449, 494, 544, 598, 658, 724, 796, 876, 963, 1060, 1166, 1282, 1411, 1552, 1707, 1878, 2066,
        2272, 2499, 2749, 3024, 3327, 3660, 4026, 4428, 4871, 5358, 5894, 6484, 7132, 7845, 8630,
        9493, 10442, 11487, 12635, 13899, 15289, 16818, 18500, 20350, 22385, 24623, 27086, 29794,
        32767]
IDX = [-1, -1, -1, -1, 2, 4, 6, 8] * 2
FREQS = {1: 8000, 2: 11000, 3: 11025, 4: 16000, 5: 22050, 6: 24000, 7: 32000, 8: 44100, 9: 48000}
CHANNELS = {0: 1, 1: 2, 2: 6, 3: 8}
CODEC_PCM16, CODEC_IMA, CODEC_XMA, CODEC_MPEG = 2, 7, 10, 11


def parse(d):
    """Returns a dict with the FSB5 fields (version 1, single sample)."""
    if d[:4] != b'FSB5':
        raise ValueError('not an FSB5')
    ver, ns, shs, nts, ds, codec = struct.unpack_from('<IIIIII', d, 4)
    if ver != 1 or ns != 1:
        raise ValueError('FSB5 version %d with %d samples: not supported' % (ver, ns))
    hs = 0x3C
    v = struct.unpack_from('<Q', d, hs)[0]
    p = hs + 8; more = v & 1; chunks = []
    while more:
        c = struct.unpack_from('<I', d, p)[0]; p += 4
        more = c & 1; size = (c >> 1) & 0xFFFFFF; typ = c >> 25
        chunks.append((typ, d[p:p + size])); p += size
    if p != hs + shs:
        raise ValueError('unexpected sample table')
    rate = FREQS.get((v >> 1) & 0xF)
    for typ, cd in chunks:
        if typ == 2: rate = struct.unpack('<I', cd)[0]
    do = hs + shs + nts
    return dict(head=d[:hs], codec=codec, channels=CHANNELS[(v >> 5) & 3], freqidx=(v >> 1) & 0xF,
                offset=((v >> 7) & 0x7FFFFFF) * 32, samples=v >> 34, chunks=chunks, rate=rate,
                names=d[hs + shs:do], data=d[do:do + ds])


def build(f):
    """Rebuilds the FSB5 from a dict like the one from parse()."""
    chcode = {1: 0, 2: 1, 6: 2, 8: 3}[f['channels']]
    v = (f['samples'] << 34) | ((f['offset'] // 32) << 7) | (chcode << 5) | (f['freqidx'] << 1)
    sh = bytearray()
    for i, (typ, cd) in enumerate(f['chunks']):
        sh += struct.pack('<I', (typ << 25) | (len(cd) << 1) | (1 if i + 1 < len(f['chunks']) else 0)) + cd
    if f['chunks']: v |= 1
    sh = struct.pack('<Q', v) + sh
    head = bytearray(f['head'])
    struct.pack_into('<IIIIII', head, 4, 1, 1, len(sh), len(f['names']), len(f['data']), f['codec'])
    return bytes(head) + bytes(sh) + f['names'] + f['data']


def _dec_channel(blocks):
    """blocks: list of 36-byte blocks of a channel -> int samples (64 per block)."""
    out = []
    for b in blocks:
        pred, idx = struct.unpack_from('<hB', b, 0)
        out.append(pred)
        k = 0
        for byte in b[4:36]:
            for n in (byte & 15, byte >> 4):
                k += 1
                if k == 64: break
                st = STEP[idx]; diff = st >> 3
                if n & 1: diff += st >> 2
                if n & 2: diff += st >> 1
                if n & 4: diff += st
                pred = pred - diff if n & 8 else pred + diff
                pred = -32768 if pred < -32768 else (32767 if pred > 32767 else pred)
                idx += IDX[n]; idx = 0 if idx < 0 else (88 if idx > 88 else idx)
                out.append(pred)
    return out


def ima_decode(data, channels, samples):
    """FMOD IMA data -> int16 array (samples, channels)."""
    bs = 36 * channels; nb = samples // 64
    chans = [[] for _ in range(channels)]
    for k in range(nb):
        blk = data[k * bs:(k + 1) * bs]
        if channels == 1:
            chans[0].append(blk)
        elif channels > 2:
            # more than 2 channels (6-channel music): first the initial int16 samples of every channel,
            # then the 16-bit step indexes, then groups of 2 bytes (4 nibbles) in turn per channel
            body = blk[4 * channels:]
            for c in range(channels):
                h = blk[2 * c:2 * c + 2] + bytes([blk[2 * channels + 2 * c], 0])
                chans[c].append(h + b''.join(body[(j * channels + c) * 2:(j * channels + c) * 2 + 2]
                                             for j in range(16)))
        else:
            L = bytearray(blk[0:4]); R = bytearray(blk[4:8])
            for j in range(8, 72, 8):
                L += blk[j:j + 4]; R += blk[j + 4:j + 8]
            chans[0].append(bytes(L)); chans[1].append(bytes(R))
    return np.array([_dec_channel(c) for c in chans], dtype=np.int16).T


def _enc_channel(x, idx0=0):
    """int16 samples of a channel (multiple of 64) -> list of 36-byte blocks."""
    blocks = []; idx = idx0
    x = [int(v) for v in x]
    for k in range(0, len(x), 64):
        pred = x[k]
        blk = bytearray(struct.pack('<hBB', pred, idx, 0))
        nib = []
        for s in x[k + 1:k + 64]:
            st = STEP[idx]; diff = s - pred; n = 0
            if diff < 0: n = 8; diff = -diff
            if diff >= st: n |= 4; diff -= st
            st2 = st >> 1
            if diff >= st2: n |= 2; diff -= st2
            if diff >= st2 >> 1: n |= 1
            st = STEP[idx]; d = st >> 3
            if n & 1: d += st >> 2
            if n & 2: d += st >> 1
            if n & 4: d += st
            pred = pred - d if n & 8 else pred + d
            pred = -32768 if pred < -32768 else (32767 if pred > 32767 else pred)
            idx += IDX[n]; idx = 0 if idx < 0 else (88 if idx > 88 else idx)
            nib.append(n)
        nib.append(0)
        for i in range(0, 64, 2):
            blk.append(nib[i] | (nib[i + 1] << 4))
        blocks.append(bytes(blk))
    return blocks


def ima_encode(pcm):
    """int16 array (samples, channels) with a multiple of 64 samples -> FMOD IMA data, aligned to 32."""
    n, ch = pcm.shape
    assert n % 64 == 0 and ch in (1, 2)
    cb = [_enc_channel(pcm[:, c]) for c in range(ch)]
    out = bytearray()
    for k in range(n // 64):
        if ch == 1:
            out += cb[0][k]
        else:
            L, R = cb[0][k], cb[1][k]
            out += L[0:4] + R[0:4]
            for j in range(4, 36, 4):
                out += L[j:j + 4] + R[j:j + 4]
    out += b'\0' * ((-len(out)) % 32)
    return bytes(out)


def resample(pcm, src, dst, loop):
    """Resamples to exactly 64 samples per block. Returns (int16 pcm, effective rate).

    Looping sounds: (periodic) FFT resampling to a length that is a multiple of 64, so the loop
    point stays continuous; the rate written is the one matching the length.
    Other sounds: polyphase filter at the requested rate and zero tail up to the whole block."""
    from scipy import signal
    from math import gcd
    x = pcm.astype(np.float64)
    if loop:
        m = max(64, int(round(len(x) * dst / src / 64)) * 64)
        y = signal.resample(x, m, axis=0)
        rate = int(round(src * m / len(x)))
    else:
        g = gcd(src, dst)
        y = signal.resample_poly(x, dst // g, src // g, axis=0)
        y = np.concatenate([y, np.zeros(((-len(y)) % 64, y.shape[1]))])
        rate = dst
    return np.clip(np.round(y), -32768, 32767).astype(np.int16), rate


def reencode(d, rate, name=''):
    """FSB5 IMA -> FSB5 IMA at the given rate (if lower than the original)."""
    f = parse(d)
    if f['codec'] != CODEC_IMA:
        raise ValueError('codec %d: only IMA (7) is re-encoded' % f['codec'])
    if rate >= f['rate']:
        return d
    pcm = ima_decode(f['data'], f['channels'], f['samples'])
    pcm, eff = resample(pcm, f['rate'], rate, 'loop' in name.lower())
    f['data'] = ima_encode(pcm)
    f['samples'] = len(pcm)
    f['freqidx'] = 0
    f['chunks'] = [(t, c) for t, c in f['chunks'] if t != 2] + [(2, struct.pack('<I', eff))]
    # the name string carries the source rate ("x.wav,0,28000,kIma4,...")
    old = b',%d,' % f['rate']; new = b',%d,' % eff
    if len(old) == len(new) and old in f['names']:
        f['names'] = f['names'].replace(old, new, 1)
    return build(f)


def write_wav(path, pcm, rate):
    import wave
    w = wave.open(path, 'wb')
    w.setnchannels(pcm.shape[1]); w.setsampwidth(2); w.setframerate(rate)
    w.writeframes(pcm.astype('<i2').tobytes()); w.close()


if __name__ == '__main__':
    # python3 fsb.py file.fsb output.wav : decodes an IMA FSB5 to wav
    f = parse(open(sys.argv[1], 'rb').read())
    pcm = ima_decode(f['data'], f['channels'], f['samples'])
    write_wav(sys.argv[2], pcm, f['rate'])
    print(f['channels'], 'channels', f['rate'], 'Hz', f['samples'], 'samples')
