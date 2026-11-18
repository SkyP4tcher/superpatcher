#!/usr/bin/env python3
"""superpatcher: adds to the recompiled Skylanders SuperChargers Xbox 360 the characters, vehicles and
driver challenges found only in the Wii U version, converting them from the Wii U files.

SuperChargers only. It contains no game files or keys and decrypts nothing: it works on the Wii U game
files already extracted by the user and on the user's Xbox 360 game. The languages are the ones on the
Xbox 360 game (the European Wii U release has all of them).

Usage:
  python3 superpatcher.py --wiiu <Wii U files folder> --x360 <recompiled 360 game folder>
                          --output <folder> [--test-copy <new folder>] [--languages es,it]

--wiiu        extracted Wii U game folder (with content/archives) or directly its archives folder.
              Read only.
--x360        recompiled installation folder (with rom/archives) or directly the archives folder of
              the Xbox 360 game. Read only.
--output      receives archives/ with the converted paks and a summary.
--test-copy   creates a separate installation: links to the original files, converted paks in place,
              executable with enlarged memory slots (only for the recognized build),
              portal_menu_nintendo = true and a .bat that uses separate save files.
--languages   forces the languages instead of reading them from the Xbox 360 game.

Requires Python 3 with numpy, and ffmpeg (xma2 decoder) to verify the audio.
"""
import argparse, os, re, shutil, struct, sys, tempfile, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import iga, igawrite, conv, fsb, xma, xma_ms, wfix_lib
from igz import Igz, fnv1a
from igzw import IgzDoc, objects

ALL_LANGS = conv.ALL_LANGS

# ---------------------------------------------------------------- content
TOY_PAKS = ['Bowser', 'Bowser_AbilityUpgrades', 'DonkeyKong', 'DonkeyKong_AbilityUpgrades',
            'ClownPlane', 'ClownPlane_blueprints', 'DKCart', 'DKCart_blueprints']
DC_PAKS = ['APA4_DC_Bowser', 'APL1_DC_DonkeyKong', 'DriverMabu_DCBowser', 'VehicleMabuAir_DCBowser']
VIDEOS = ['SG_Bowser.bk2', 'SG_DonkeyKong.bk2', 'SG_ClownCruiser.bk2', 'SG_BarrelBlaster.bk2']
HERO_SLOT = 3 * 1024 * 1024
VEHICLE_SLOT = 1536 * 1024
# sound banks loaded together: (slot, main paks, prefix of the per-language voice paks)
AUDIO_GROUPS = {
    'Bowser': (HERO_SLOT, ['CHAMP_L15_Bowser_bank', 'CHAMP_L15_Bowser_bank_v', 'CHAMP_L15_Bowser_Giant_bank_v'],
               'CHAMP_L15_Bowser_bank_Lv'),
    'DonkeyKong': (HERO_SLOT, ['CHAMP_L15_DonkeyKong_bank', 'CHAMP_L15_DonkeyKong_bank_v'],
                   'CHAMP_L15_DonkeyKong_bank_Lv'),
    'ClownPlane': (VEHICLE_SLOT, ['CHAMP_V15_BowserPlane_bank'], 'CHAMP_V15_BowserPlane_bank_Lv'),
    'DKCart': (VEHICLE_SLOT, ['CHAMP_V15_DKCart_bank'], 'CHAMP_V15_DKCart_bank_Lv'),
}
AUDIO_FRAC = 0.93
DC_AUDIO = ['MAP_DriverChallenge_Bowser_bank', 'MAP_DriverChallenge_DonkeyKong_bank', 'MUSIC_ActionPack_DC_Bowser_bank']
DC_GENBANK = ['APA4_DC_Bowser_genbank', 'APL1_DC_DonkeyKong_genbank']
DC_NOISE = 0.15
MAPS = {
    'APA4': ('maps/ActionPacks/APA4/APA4_DriverChallenge_Bowser.igz', ('driverchallenge_bowser', 'dc_bowser', 'dcbowser')),
    'APL1': ('maps/ActionPacks/APL1/APL1_DriverChallenge_DonkeyKong.igz', ('driverchallenge_donkeykong', 'dc_donkeykong')),
}
STUB = 2184                     # size of the empty maps on the Xbox 360 game
PERMANENT_PKG = 'packages/generated/packageXmls/permanent_pkg.igz'
SYSTEM_DATA = 'LooseData/systemData.igz'
DC_ICONS = ('Bowser_tga__material', 'DonkeyKong_tga__material')
SECTOR_AUDIO = 0x800

# recompiled executable: hero and vehicle memory slots. The code moves between Core of Light builds,
# so each change is found by its surrounding bytes, which must match exactly once.
# (signature, [(offset in the signature, old byte, new byte)])
EXE_NAME = 'superchargersrecompiled.exe'
EXE_PATCH = [
    (bytes.fromhex('41bb0017980047891c0a'), [(3, 0x17, 0x27)]),                 # hero data 0x179800 -> 0x279800
    (bytes.fromhex('48c7415800201c00b8001c2000'), [(6, 0x1C, 0x2C), (10, 0x1C, 0x2C)]),  # vehicle slot 0x1C2000 -> 0x2C2000
]


def patch_exe(b):
    """Enlarges the memory slots in the executable. Returns (patched bytes or None, message)."""
    b = bytearray(b)
    todo = []
    for sig, changes in EXE_PATCH:
        done = bytearray(sig)
        for k, _, new in changes:
            done[k] = new
        if b.count(sig) == 1:
            todo.append((b.find(sig), changes))
        elif b.count(sig) == 0 and b.count(bytes(done)) == 1:
            continue                     # already patched
        else:
            return None, ('executable NOT modified: memory slot code not recognized in this Core of Light '
                          'build, the game may run out of memory loading the added characters and vehicles')
    if not todo:
        return None, 'executable: memory slots already enlarged'
    for pos, changes in todo:
        for k, old, new in changes:
            assert b[pos + k] == old
            b[pos + k] = new
    return bytes(b), 'executable: hero and vehicle memory slots enlarged'
TOML_NAME = 'superchargersrecomp.toml'
VRAM_NAME = 'VramBudget.igx'
VRAM_XENON = {'_character': '14', '_vehicle': '14'}   # MB per character and vehicle (was 8 and 10)


def patch_vram(b):
    """Xbox 360 video memory budget per character and vehicle: the added toys' textures are at full
    quality. Only the values in the XENON entry change."""
    t = b.decode('utf-8')
    m = re.search(r'(<object refname="VramBudget_IG_CORE_PLATFORM_XENON".*?</object>)', t, re.S)
    if not m:
        return b
    blk = m.group(1)
    for k, v in VRAM_XENON.items():
        blk = re.sub(r'(<var name="%s" value=")[^"]*(")' % k, r'\g<1>%s\g<2>' % v, blk)
    return (t[:m.start()] + blk + t[m.end():]).encode('utf-8')


def log(*a):
    print(*a, flush=True)


def lang_of(name):
    m = re.search(r'_([a-z]{2})$', name.rsplit('.', 1)[0])
    return m.group(1) if m and m.group(1) in ALL_LANGS else None


def wanted_lang(name, langs):
    lg = lang_of(name)
    return lg is None or lg in langs


# ---------------------------------------------------------------- Wii U files
def find_wiiu(path):
    for p in (os.path.join(path, 'content', 'archives'), os.path.join(path, 'archives'), path):
        if os.path.exists(os.path.join(p, 'Bowser.pak')):
            return p
    raise SystemExit('Bowser.pak not found in %s: extracted SuperChargers Wii U is needed' % path)


# ---------------------------------------------------------------- Xbox 360 game
def find_archives(path):
    for p in (os.path.join(path, 'rom', 'archives'), os.path.join(path, 'archives'), path):
        if os.path.exists(os.path.join(p, 'permanent.pak')):
            return p
    raise SystemExit('permanent.pak not found in ' + path)


def langs_360(arch):
    names = [n for _, n in igawrite.iga_names(os.path.join(arch, 'permanent.pak'))]
    langs = sorted(set(m.group(1) for n in names for m in [re.fullmatch(r'LooseData/systemData_([a-z]{2})\.lng', n)] if m))
    if not langs:
        raise SystemExit('Xbox 360 game languages not recognized')
    return langs


# ---------------------------------------------------------------- IGZ and pak
def convert_entry(name, data, langs):
    if not wanted_lang(name, langs):
        return None
    if data[:4] == b'IGZ\x01':
        doc = conv.convert(data, name)
        if name.startswith('actors/Vehicles/'):
            wfix_lib.fix(doc)        # the 360 rewrites the first weight to 255: the largest weight goes there
        return doc.serialize()
    return data


def pak_params(path):
    b = open(path, 'rb').read(0x38)
    return struct.unpack_from('>I', b, 0x10)[0], struct.unpack_from('>I', b, 0x34)[0]


def write_pak(dst, files, sector, flags):
    igawrite.write_iga(dst + '.part', files, sector=sector, flags=flags)
    os.replace(dst + '.part', dst)


def convert_pak(src, dst, langs):
    sector, flags = pak_params(src)
    files = []
    for logical, bpath, data, stored in igawrite.entry_list(src):
        d = convert_entry(logical, data, langs) if '/cafe/' in bpath else data
        if d is None:
            continue
        files.append((logical, bpath.replace('/BuildServer/cafe/', '/BuildServer/xenon/'), d, 'stored' if stored else 'zlib'))
    write_pak(dst, files, sector, flags)
    return files


def patch_dc_flag(igzb):
    """The added characters' driver challenges have a flag (+0x24) that the 360 game uses to hide them."""
    g = Igz(igzb); t = g.strings('TSTR'); cnt, d = g.fix['EXNM']
    P = [struct.unpack_from('>II', d, i * 8) for i in range(cnt)]
    doc = IgzDoc(igzb)
    sec_off = g.sections[1][1]
    out = bytearray(igzb); done = []
    for s, o, ty in objects(doc):
        if ty != 'CDriverChallengeData':
            continue
        w = struct.unpack_from('>10I', out, sec_off + o)
        icon = t[P[(w[4] & 0xffffff) + 8][1] & 0x7fffffff]
        if icon in DC_ICONS:
            if w[9] not in (0, 0x01000000):
                raise SystemExit('systemData: unexpected flag value for challenge %s' % icon)
            struct.pack_into('>I', out, sec_off + o + 0x24, 0)
            done.append(icon)
    if sorted(done) != sorted(DC_ICONS):
        raise SystemExit('systemData: driver challenges of the added characters not found')
    return bytes(out)


def build_permanent(arch, wiiu, dst, langs):
    src = os.path.join(arch, 'permanent.pak')
    sector, flags = pak_params(src)
    have = set(n.lower() for _, n in igawrite.iga_names(src))
    wl = igawrite.entry_list(os.path.join(wiiu, 'permanent.pak'))
    files = []; added = []
    for logical, bpath, data, stored in igawrite.entry_list(src):
        if logical == PERMANENT_PKG:
            data = conv.convert([e for e in wl if e[0] == PERMANENT_PKG][0][2], PERMANENT_PKG).serialize()
        elif logical == SYSTEM_DATA:
            data = patch_dc_flag(data)
        files.append((logical, bpath, data, 'stored' if stored else 'zlib'))
    for logical, bpath, data, stored in wl:
        if logical.lower() in have or not wanted_lang(logical, langs):
            continue
        d = conv.convert(data, logical).serialize() if data[:4] == b'IGZ\x01' else data
        files.append((logical, bpath.replace('/BuildServer/cafe/', '/BuildServer/xenon/'), d, 'stored' if stored else 'zlib'))
        added.append(logical)
    write_pak(dst, files, sector, flags)
    return files, added


def plan_map(arch, wiiu, name, langs):
    """Entries to replace and to add to the track pak for the driver challenge."""
    mapname, keys = MAPS[name]
    wu = igawrite.entry_list(os.path.join(wiiu, name + '.pak'))
    x = igawrite.entry_list(os.path.join(arch, name + '.pak'))
    xi = {e[0].lower(): e for e in x}
    wi = {e[0].lower(): e for e in wu}
    stubs = [e[0] for e in x if len(e[2]) == STUB and e[0].lower() in wi and len(wi[e[0].lower()][2]) > STUB
             and any(k in e[0].lower() for k in keys)]
    if stubs != [mapname]:
        raise SystemExit('%s: empty challenge map not found on the Xbox 360 game (%s)' % (name, stubs))
    pkg = 'packages/generated/maps/actionpacks/%s/%s_pkg.igz' % (name.lower(), name.lower())
    refs = set()
    for s in stubs + ([pkg] if pkg in wi else []):
        refs |= set(t.decode('latin1').lower() for t in re.findall(rb'[\w./@!,`{}-]{6,}', wi[s.lower()][2]))
    add = []
    while True:                      # closure: added files reference others too
        new = []
        for e in wu:
            n = e[0].lower()
            if n in xi or e[0] in add or not wanted_lang(e[0], langs):
                continue
            base = n.rsplit('/', 1)[-1].rsplit('.', 1)[0]
            if any(k in n for k in keys) or any(base in r for r in refs):
                new.append(e[0])
        if not new:
            break
        add += new
        for n in new:
            refs |= set(t.decode('latin1').lower() for t in re.findall(rb'[\w./@!,`{}-]{6,}', wi[n.lower()][2]))
    return wu, x, stubs, add, pkg


def build_map(arch, wiiu, name, dst, langs):
    wu, x, stubs, add, pkg = plan_map(arch, wiiu, name, langs)
    wi = {e[0].lower(): e for e in wu}
    sector, flags = pak_params(os.path.join(arch, name + '.pak'))
    files = []
    for logical, bpath, data, stored in x:
        if logical in stubs or (logical.lower() == pkg and pkg in wi):
            data = convert_entry(logical, wi[logical.lower()][2], langs)
        files.append((logical, bpath, data, 'stored' if stored else 'zlib'))
    for n in add:
        logical, bpath, data, stored = wi[n.lower()]
        d = convert_entry(logical, data, langs) if '/cafe/' in bpath else data
        files.append((logical, bpath.replace('/BuildServer/cafe/', '/BuildServer/xenon/'), d, 'stored' if stored else 'zlib'))
    write_pak(dst, files, sector, flags)
    return files, stubs, add


# ---------------------------------------------------------------- audio
def mp3_decode(data, channels, samples, work):
    import subprocess
    src = os.path.join(work, 'in.mp3'); raw = os.path.join(work, 'out.raw')
    open(src, 'wb').write(data)
    r = subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'mp3', '-i', src,
                        '-f', 's16le', '-acodec', 'pcm_s16le', raw], capture_output=True, text=True)
    if r.returncode or r.stderr.strip():
        raise SystemExit('ffmpeg mp3: ' + r.stderr.strip())
    y = np.fromfile(raw, '<i2').reshape(-1, channels)
    if len(y) >= samples:
        return y[:samples]
    return np.concatenate([y, np.zeros((samples - len(y), channels), np.int16)])


def source_pcm(f, work):
    if f['codec'] == fsb.CODEC_IMA:
        return fsb.ima_decode(f['data'], f['channels'], f['samples'])
    if f['codec'] == fsb.CODEC_PCM16:
        return np.frombuffer(f['data'][:f['samples'] * 2 * f['channels']], '<i2').reshape(-1, f['channels'])
    if f['codec'] == fsb.CODEC_MPEG:
        return mp3_decode(f['data'], f['channels'], f['samples'], work)
    raise SystemExit('audio codec %d not supported' % f['codec'])


def load_bank(path, work, dc=False):
    out = []
    for logical, bpath, data, stored in igawrite.entry_list(path):
        f = fsb.parse(data)
        loop = 'loop' in logical.lower()
        if dc:
            loop = loop or logical.split('/')[-1].startswith('INTER_') and not logical.lower().endswith('intro')
        out.append(dict(logical=logical, f=f, loop=loop, pcm=xma.prepare_pcm(source_pcm(f, work), f['rate'], loop)))
    return out


def _encode_stream(args):
    pcm, noise, loop = args
    return xma.encode_frames(pcm, xma.Params(noise=noise), loop)


def encode_fsb(e, noise, pool=None):
    """XMA FSB of a sound; above 2 channels one stream per channel pair (xma_ms)."""
    pcm = e['pcm']; nch = pcm.shape[1]
    if nch <= 2:
        data, c6, _ = xma.encode(pcm, xma.Params(noise=noise), e['loop'])
    else:
        lay = xma_ms.stream_layout(nch); parts = []; c = 0
        for n in lay:
            parts.append(np.ascontiguousarray(pcm[:, c:c + n])); c += n
        mp = pool.map if pool else map
        frames = list(mp(_encode_stream, [(p, noise, e['loop']) for p in parts]))
        data, counts = xma_ms.interleave(frames)
        c6 = xma_ms.seek_chunk(len(lay), counts)
    return xma.to_fsb(e['f'], data, c6, len(pcm), e['f']['channels'])


def fsb_size(d):
    return (len(d) + SECTOR_AUDIO - 1) // SECTOR_AUDIO * SECTOR_AUDIO


def _encode_refine(args):
    e, noise, refine = args
    d = encode_fsb(e, noise)
    if refine:                       # raise quality while it stays in the same sectors
        size = fsb_size(d); n = noise
        for _ in range(8):
            n *= 0.85
            d2 = encode_fsb(e, n)
            if fsb_size(d2) > size:
                break
            d = d2
    return d


def verify_fsb(d, pcm, work):
    """Identical re-read and rewrite of every stream, ffmpeg decode without errors and with the right
    length. Returns (ok, message, minimum SNR)."""
    g = fsb.parse(d)
    if g['codec'] != fsb.CODEC_XMA:
        return False, 'codec %d' % g['codec'], 0
    lay = xma_ms.stream_layout(g['channels'])
    if len(lay) == 1:
        ok, _ = xma.check_stream(g['data'], g['channels'])
        if not ok:
            return False, 'rewrite differs', 0
        streams = [g['data']]
    else:
        bl = xma_ms.split_streams(g['data'], len(lay))
        offs = xma.band_offsets(xma.RATE)
        fr = [xma_ms.stream_frames(bl[s], lay[s], xma.RATE) for s in range(len(lay))]
        fb = [[xma.write_frame(f, lay[s], offs) for b in fr[s] for f in b] for s in range(len(lay))]
        data, counts = xma_ms.interleave(fb)
        if data != g['data'] or xma_ms.seek_chunk(len(lay), counts) != dict(g['chunks'])[6]:
            return False, 'multistream rewrite differs', 0
        streams = []
        for blocks in bl:
            s = bytearray(b''.join(blocks))
            for k in range(0, len(s), xma.PKT):
                s[k + 3] = 0
            streams.append(bytes(s))
    snrs = []; c = 0
    for n, sd in zip(lay, streams):
        y, err = xma.ffmpeg_decode(sd, n, g['samples'], work)
        if err:
            return False, 'ffmpeg: ' + err[:200], 0
        if len(y) != g['samples']:
            return False, 'length %d instead of %d' % (len(y), g['samples']), 0
        snrs.append(xma.snr_db(pcm[:, c:c + n], y)); c += n
    return True, '', min(snrs)


def write_bank(src, dst, entries, encoded, work, min_snr):
    """Audio pak on the structure of the Wii U pak (hashes, names and chain kept), after verification."""
    new = {}; snrs = []
    for e, d in zip(entries, encoded):
        ok, msg, snr = verify_fsb(d, e['pcm'], work)
        silent = float(np.max(np.abs(e['pcm']))) < 1.0
        if not ok or (snr < min_snr and not silent):
            raise SystemExit('%s %s: audio check failed (%s, SNR %.1f)' % (os.path.basename(src), e['logical'], msg, snr))
        new[e['logical']] = d
        if not silent:
            snrs.append(snr)
    igawrite.replace_iga(src, dst + '.part', new)
    os.replace(dst + '.part', dst)
    return (min(snrs), float(np.mean(snrs))) if snrs else (0, 0)


def pak_bytes(encoded, src):
    """Size of the pak written by replace_iga: header and names of the source, data in sectors."""
    b = open(src, 'rb').read(0x38)
    n = struct.unpack_from('>I', b, 0xC)[0]
    head = 0x38 + n * 20; head += (-head) % SECTOR_AUDIO
    return head + sum(fsb_size(d) for d in encoded) + struct.unpack_from('>I', b, 0x30)[0]


def build_toy_audio(wiiu, out, langs, pool, work):
    report = []
    for g, (slot, paks, lv) in AUDIO_GROUPS.items():
        names = paks + ['%s_%s' % (lv, l) for l in langs]
        src = {p: load_bank(os.path.join(wiiu, p + '.pak'), work) for p in names}
        def group_size(enc):
            size = lambda p: pak_bytes(enc[p], os.path.join(wiiu, p + '.pak'))
            return sum(size(p) for p in paks) + max(size('%s_%s' % (lv, l)) for l in langs)
        def encode_all(noise, refine=False):
            return {p: list(pool.map(_encode_refine, [(e, noise, refine) for e in src[p]], chunksize=1)) for p in names}
        lo, hi = 0.1, 4.0            # common noise for the group: the lowest that fits in the slot
        for _ in range(8):
            mid = (lo * hi) ** 0.5
            if group_size(encode_all(mid)) > AUDIO_FRAC * slot:
                lo = mid
            else:
                hi = mid
        enc = encode_all(hi, refine=True)
        total = group_size(enc)
        if total > slot:
            raise SystemExit('%s: audio too large for the slot (%d > %d)' % (g, total, slot))
        for p in names:
            mn, avg = write_bank(os.path.join(wiiu, p + '.pak'), os.path.join(out, p + '.pak'), src[p], enc[p], work, 3)
            report.append('%-36s %3d sounds  SNR min %.1f mean %.1f dB' % (p, len(src[p]), mn, avg))
        report.append('== %s: %d bytes, %.1f%% of the slot' % (g, total, 100.0 * total / slot))
        log(report[-1])
    return report


def build_dc_audio(wiiu, out, langs, pool, work):
    report = []
    for p in DC_AUDIO + ['%s_%s' % (b, l) for b in DC_GENBANK for l in langs]:
        entries = load_bank(os.path.join(wiiu, p + '.pak'), work, dc=True)
        enc = [encode_fsb(e, DC_NOISE, pool) for e in entries]
        mn, avg = write_bank(os.path.join(wiiu, p + '.pak'), os.path.join(out, p + '.pak'), entries, enc, work, 10)
        report.append('%-36s %3d sounds  SNR min %.1f mean %.1f dB' % (p, len(entries), mn, avg))
        log(report[-1])
    return report


# ---------------------------------------------------------------- checks
def check_pak(dst, files, langs, base=None):
    """Re-read: same data, sorted FNV hashes, data chain, 360 IGZ, no Wii U paths or languages.
    base: starting 360 pak, whose entries stay even in languages other than the chosen ones."""
    keep = set(n for _, n in igawrite.iga_names(base)) if base else set()
    f2, d2 = iga.read_iga(dst)
    got = {x[0]: d2(x) for x in f2}
    if got != {f[0]: f[2] for f in files}:
        raise SystemExit('%s: re-read data differs' % dst)
    hs = igawrite.iga_hashes(dst); names = igawrite.iga_names(dst)
    if hs != sorted(hs) or not all(h == fnv1a(n[1].lower()) for h, n in zip(hs, names)):
        raise SystemExit('%s: hash' % dst)
    if not igawrite.check_chain(dst):
        raise SystemExit('%s: entry chain' % dst)
    for n, d in got.items():
        if d[:4] == b'IGZ\x01' and struct.unpack_from('>I', d, 0xC)[0] != 5:
            raise SystemExit('%s: %s is not for 360' % (dst, n))
        if not wanted_lang(n, langs) and n not in keep:
            raise SystemExit('%s: language not present on the 360 game (%s)' % (dst, n))
    if any('/cafe/' in n[0] for n in names):
        raise SystemExit('%s: Wii U paths left' % dst)
    return len(got)


def check_audio_pak(dst, src):
    if igawrite.iga_hashes(dst) != igawrite.iga_hashes(src) or igawrite.iga_names(dst) != igawrite.iga_names(src):
        raise SystemExit('%s: hashes or names differ from the Wii U pak' % dst)
    if not igawrite.check_chain(dst):
        raise SystemExit('%s: entry chain' % dst)
    f2, d2 = iga.read_iga(dst)
    if any(fsb.parse(d2(x))['codec'] != fsb.CODEC_XMA for x in f2):
        raise SystemExit('%s: audio is not XMA' % dst)


# ---------------------------------------------------------------- test copy
def link_or_copy(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def make_test_copy(root, out_arch, dest):
    if not os.path.exists(os.path.join(root, EXE_NAME)):
        raise SystemExit('--test-copy requires the recompiled installation folder in --x360')
    if os.path.exists(dest):
        raise SystemExit('%s already exists: give a new folder' % dest)
    for d, _, fs in os.walk(root):
        rel = os.path.relpath(d, root)
        os.makedirs(os.path.join(dest, rel), exist_ok=True)
        for f in fs:
            link_or_copy(os.path.join(d, f), os.path.join(dest, rel, f))
    arch = os.path.join(dest, 'rom', 'archives')
    for f in os.listdir(out_arch):
        t = os.path.join(arch, f)
        if os.path.exists(t):
            os.remove(t)                 # drop the link: the original does not change
        shutil.copyfile(os.path.join(out_arch, f), t)
    exe = os.path.join(dest, EXE_NAME)
    b = open(exe, 'rb').read()
    os.remove(exe)                       # drop the link before writing
    patched, msg = patch_exe(b)
    open(exe, 'wb').write(patched or b)
    vb = os.path.join(dest, 'rom', VRAM_NAME)
    if os.path.exists(vb):
        t = open(vb, 'rb').read()
        os.remove(vb)
        open(vb, 'wb').write(patch_vram(t))
    toml = os.path.join(dest, TOML_NAME)
    if os.path.exists(toml):
        t = open(toml, encoding='utf-8').read()
        os.remove(toml)
        if re.search(r'^\s*portal_menu_nintendo\s*=', t, re.M):
            t = re.sub(r'^(\s*portal_menu_nintendo\s*=\s*)\w+', r'\1true', t, flags=re.M)
        else:
            lines = t.split('\n')
            sec = [k for k, l in enumerate(lines) if re.fullmatch(r'\s*\[\s*"?Portal"?\s*\]\s*', l)]
            if sec:
                lines.insert(sec[0] + 1, 'portal_menu_nintendo = true')
            else:
                lines += ['', '["Portal"]', 'portal_menu_nintendo = true']
            t = '\n'.join(lines)
        open(toml, 'w', encoding='utf-8').write(t)
    open(os.path.join(dest, 'Launch SuperChargers superpatcher.bat'), 'w', newline='\r\n').write(
        '@echo off\ncd /d "%~dp0"\nstart "" ' + EXE_NAME +
        ' --user_data_root="%USERPROFILE%\\Documents\\Skylanders\\SuperChargers superpatcher"\n')
    return msg


def patch_exe_in_place(root):
    exe = os.path.join(root, EXE_NAME)
    if not os.path.exists(exe):
        raise SystemExit('%s not found in %s' % (EXE_NAME, root))
    b = open(exe, 'rb').read()
    patched, msg = patch_exe(b)
    if patched:
        backup = exe + '.orig'
        if os.path.exists(backup):
            os.remove(backup)
        os.replace(exe, backup)
        open(exe, 'wb').write(patched)
        msg += ' (original kept as %s)' % os.path.basename(backup)
    return msg


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description='Wii U characters and challenges in the recompiled SuperChargers 360')
    ap.add_argument('--wiiu', help='extracted SuperChargers Wii U folder')
    ap.add_argument('--x360', help='Core of Light folder with SuperChargers')
    ap.add_argument('--output', help='folder for the modified files')
    ap.add_argument('--test-copy', help='new folder for a separate installation')
    ap.add_argument('--languages', help='e.g. es,it (default: from the Xbox 360 game)')
    ap.add_argument('--jobs', type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument('--patch-exe', metavar='FOLDER',
                    help='only enlarge the memory slots of the executable in a Core of Light folder, '
                         'for example after a Core of Light update (a backup is kept)')
    a = ap.parse_args()
    if a.patch_exe:
        print(patch_exe_in_place(a.patch_exe))
        return
    if not (a.wiiu and a.x360 and a.output):
        ap.error('--wiiu, --x360 and --output are required')
    t0 = time.time()
    import subprocess
    try:
        dec = subprocess.run(['ffmpeg', '-hide_banner', '-decoders'], capture_output=True, text=True).stdout
    except OSError:
        dec = ''
    if ' xma2 ' not in dec:
        raise SystemExit('ffmpeg with the xma2 decoder is required (to verify the converted audio)')

    arch = find_archives(a.x360)
    langs = a.languages.split(',') if a.languages else langs_360(arch)
    bad = [l for l in langs if l not in ALL_LANGS]
    if bad:
        raise SystemExit('unknown languages: %s' % bad)
    conv.LANGS_360 = tuple(langs)
    log('Xbox 360 game: %s, languages %s' % (arch, ','.join(langs)))

    wiiu = find_wiiu(a.wiiu)
    out = os.path.join(a.output, 'archives')
    need = [p + '.pak' for p in TOY_PAKS + DC_PAKS + DC_AUDIO + ['permanent', 'APA4', 'APL1']]
    for g, (_, paks, lv) in AUDIO_GROUPS.items():
        need += [p + '.pak' for p in paks] + ['%s_%s.pak' % (lv, l) for l in langs]
    need += ['%s_%s.pak' % (p, l) for p in DC_GENBANK for l in langs]
    need += VIDEOS
    missing = [n for n in need if not os.path.exists(os.path.join(wiiu, n))]
    if missing:
        raise SystemExit('Wii U files missing in %s: %s' % (wiiu, ', '.join(missing)))
    log('Wii U files: %s, all %d required files present' % (wiiu, len(need)))
    os.makedirs(out, exist_ok=True)
    work = tempfile.mkdtemp(prefix='superpatcher_')

    report = ['languages: ' + ','.join(langs)]
    # 2. toys, vehicles and challenge levels
    for p in TOY_PAKS + DC_PAKS:
        files = convert_pak(os.path.join(wiiu, p + '.pak'), os.path.join(out, p + '.pak'), langs)
        report.append('%-36s %4d entries' % (p, check_pak(os.path.join(out, p + '.pak'), files, langs)))
        log(report[-1])
    for v in VIDEOS:
        shutil.copyfile(os.path.join(wiiu, v), os.path.join(out, v))
    # 3. figure registry and challenges visible in the menu
    files, added = build_permanent(arch, wiiu, os.path.join(out, 'permanent.pak'), langs)
    check_pak(os.path.join(out, 'permanent.pak'), files, langs, os.path.join(arch, 'permanent.pak'))
    report.append('permanent.pak: %d entries added, permanent_pkg from Wii U, added driver challenges visible' % len(added))
    log(report[-1])
    # 4. challenge maps in the track paks
    for name in MAPS:
        files, stubs, add = build_map(arch, wiiu, name, os.path.join(out, name + '.pak'), langs)
        check_pak(os.path.join(out, name + '.pak'), files, langs, os.path.join(arch, name + '.pak'))
        report.append('%s.pak: map replaced, %d entries added, _pkg from Wii U' % (name, len(add)))
        log(report[-1])
    # 5. audio to XMA
    audio_work = os.path.join(work, 'audio')
    os.makedirs(audio_work, exist_ok=True)
    with ProcessPoolExecutor(a.jobs) as pool:
        report += build_toy_audio(wiiu, out, langs, pool, audio_work)
        report += build_dc_audio(wiiu, out, langs, pool, audio_work)
    for f in os.listdir(out):
        if f.endswith('.pak') and (f.startswith(('CHAMP_', 'MAP_', 'MUSIC_')) or '_genbank_' in f):
            check_audio_pak(os.path.join(out, f), os.path.join(wiiu, f))
    first = len(report)
    report.append('checks passed on %d files' % len(os.listdir(out)))
    if a.test_copy:
        report.append(make_test_copy(os.path.abspath(a.x360), out, a.test_copy))
        report.append('test copy: ' + a.test_copy)
    report.append('time: %.0f s' % (time.time() - t0))
    open(os.path.join(a.output, 'summary.txt'), 'w', encoding='utf-8').write('\n'.join(report) + '\n')
    log('\n'.join(report[first:]))
    shutil.rmtree(work, ignore_errors=True)   # temporary folder created here


if __name__ == '__main__':
    main()
