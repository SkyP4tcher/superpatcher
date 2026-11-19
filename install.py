"""Installing the converted files into a Core of Light game folder, and putting everything back.

What install replaces is copied into a backup folder inside the game folder, next to a manifest
(manifest.json) that lists every change: the file's checksum before and after, or the previous value
of a toml setting. uninstall() puts back only what is still as the install left it: a file that a
Core of Light update or a file check has replaced since then is kept as it is.
"""
import hashlib, json, os, re, shutil

MANIFEST = 'manifest.json'


def progress(step, total, label):
    """A line the launcher can read to show how far the work is."""
    print('PROGRESS %d/%d %s' % (step, total, label), flush=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_file(path, data):
    """Writes next to the target and then replaces it, so a stop half way leaves no half file.
    The old file is removed first: a hard link to another installation stays untouched."""
    tmp = path + '.part'
    with open(tmp, 'wb') as f:
        f.write(data)
    if os.path.exists(path):
        os.remove(path)
    os.replace(tmp, path)


# ---------------------------------------------------------------- toml settings
def read_text(path):
    """Text of a settings file with \\n line ends, and whether the file used \\r\\n."""
    if not os.path.exists(path):
        return '', True
    b = open(path, 'rb').read().decode('utf-8')
    return b.replace('\r\n', '\n'), '\r\n' in b


def write_text(path, t, crlf):
    write_file(path, (t.replace('\n', '\r\n') if crlf else t).encode('utf-8'))


def _lines(t):
    """Lines of a text, and the final line end to put back."""
    if t.endswith('\n'):
        return t[:-1].split('\n'), '\n'
    return t.split('\n') if t else [], ''


def _section_range(lines, section):
    head = re.compile(r'\s*\[\s*"?%s"?\s*\]\s*$' % re.escape(section))
    start = next((k for k, l in enumerate(lines) if head.match(l)), None)
    if start is None:
        return None, None
    end = next((k for k in range(start + 1, len(lines)) if re.match(r'\s*\[', lines[k])), len(lines))
    return start, end


def toml_get(t, section, key):
    lines = t.split('\n')
    start, end = _section_range(lines, section)
    if start is None:
        return None
    for k in range(start + 1, end):
        m = re.match(r'\s*%s\s*=\s*([^#]*?)\s*(#.*)?$' % re.escape(key), lines[k])
        if m:
            return m.group(1)
    return None


def toml_set(t, section, key, value):
    """Sets key = value in [section], adding the key or the section if missing."""
    lines, tail = _lines(t)
    start, end = _section_range(lines, section)
    if start is None:
        if lines and lines[-1].strip():
            lines.append('')
        lines += ['["%s"]' % section, '%s = %s' % (key, value)]
        return '\n'.join(lines) + tail
    for k in range(start + 1, end):
        if re.match(r'\s*%s\s*=' % re.escape(key), lines[k]):
            lines[k] = re.sub(r'^(\s*%s\s*=\s*)[^#]*?(\s*(#.*)?)$' % re.escape(key),
                              lambda m: m.group(1) + value + m.group(2), lines[k])
            return '\n'.join(lines) + tail
    lines.insert(start + 1, '%s = %s' % (key, value))
    return '\n'.join(lines) + tail


def toml_del(t, section, key):
    """Removes key from [section], and the section header too if nothing else is left in it."""
    lines, tail = _lines(t)
    start, end = _section_range(lines, section)
    if start is None:
        return t
    lines = [l for k, l in enumerate(lines)
             if not (start < k < end and re.match(r'\s*%s\s*=' % re.escape(key), l))]
    start, end = _section_range(lines, section)
    if not any(l.strip() for l in lines[start + 1:end]):
        del lines[start:end]
        while start > 0 and not lines[start - 1].strip():
            del lines[start - 1]
            start -= 1
    return '\n'.join(lines) + tail


# ---------------------------------------------------------------- install and uninstall
class Installer:
    """Records every change in the manifest as it happens, so even a stopped install can be undone."""

    def __init__(self, root, backup_name, tool):
        self.root = root
        self.backup = os.path.join(root, backup_name)
        self.data = {'tool': tool, 'files': [], 'toml': []}
        os.makedirs(self.backup, exist_ok=True)
        self.save()

    def save(self):
        write_file(os.path.join(self.backup, MANIFEST),
                   json.dumps(self.data, indent=1).encode('utf-8'))

    def put(self, rel, data):
        """Writes a file of the game folder, keeping the one it replaces."""
        path = os.path.join(self.root, rel)
        entry = {'path': rel.replace(os.sep, '/')}
        if os.path.exists(path):
            entry['old'] = sha256(path)
            keep = os.path.join(self.backup, rel)
            os.makedirs(os.path.dirname(keep), exist_ok=True)
            shutil.copyfile(path, keep)
        self.data['files'].append(entry)
        self.save()                      # before writing: a stop now still knows the file
        write_file(path, data)
        entry['new'] = hashlib.sha256(data).hexdigest()
        self.save()

    def edit(self, rel, func):
        """func(old bytes) -> new bytes, or None to leave the file as it is."""
        path = os.path.join(self.root, rel)
        if not os.path.exists(path):
            return False
        old = open(path, 'rb').read()
        new = func(old)
        if new is None or new == old:
            return False
        self.put(rel, new)
        return True

    def setting(self, toml_rel, section, key, value):
        path = os.path.join(self.root, toml_rel)
        t, crlf = read_text(path)
        if toml_get(t, section, key) == value:
            return
        self.data['toml'].append({'file': toml_rel.replace(os.sep, '/'), 'section': section, 'key': key,
                                  'old': toml_get(t, section, key), 'new': value})
        self.save()
        write_text(path, toml_set(t, section, key, value), crlf)


def installed(root, backup_name):
    return os.path.exists(os.path.join(root, backup_name, MANIFEST))


def uninstall(root, backup_name):
    """Puts back what the install changed. Returns the report lines."""
    backup = os.path.join(root, backup_name)
    man = os.path.join(backup, MANIFEST)
    if not os.path.exists(man):
        raise SystemExit('nothing to uninstall: %s not found' % man)
    data = json.load(open(man, encoding='utf-8'))
    kept = []
    settings = 0
    for s in reversed(data.get('toml', [])):
        path = os.path.join(root, s['file'])
        if not os.path.exists(path):
            continue
        t, crlf = read_text(path)
        if toml_get(t, s['section'], s['key']) != s['new']:
            kept.append('%s: %s' % (s['file'], s['key']))
            continue
        t = toml_del(t, s['section'], s['key']) if s['old'] is None else \
            toml_set(t, s['section'], s['key'], s['old'])
        write_text(path, t, crlf)
        settings += 1
    restored = removed = 0
    for e in reversed(data.get('files', [])):
        path = os.path.join(root, e['path'])
        if 'new' in e and os.path.exists(path) and sha256(path) != e['new']:
            kept.append(e['path'])
            continue
        if 'old' in e:
            write_file(path, open(os.path.join(backup, e['path']), 'rb').read())
            restored += 1
        elif os.path.exists(path):
            os.remove(path)
            removed += 1
    shutil.rmtree(backup)
    report = ['uninstalled: %d files restored, %d added files removed, %d settings restored'
              % (restored, removed, settings)]
    for k in kept:
        report.append('changed by something else since the install, left as it is: ' + k)
    return report
