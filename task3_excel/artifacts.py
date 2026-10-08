"""Публикация Excel и отчётов без перезаписи и с откатом при обычной ошибке I/O."""
import errno
import shutil
import hashlib
import json
import os
from pathlib import Path
import tempfile
from datetime import datetime, timezone


def publish_exclusive(source, destination):
    """Prefer atomic hard-link publication; copy exclusively on unsupported filesystems."""
    try:
        os.link(source, destination)
    except OSError as exc:
        if exc.errno not in {errno.EPERM, errno.EXDEV, errno.ENOSYS, errno.EOPNOTSUPP}:
            raise
        # Never overwrite a file created by another process. A copy is not atomic
        # to readers; consumers should validate the sidecar hash before use.
        with open(destination, 'xb') as out:
            identity = os.fstat(out.fileno())
            try:
                with open(source, 'rb') as src: shutil.copyfileobj(src, out)
                out.flush(); os.fsync(out.fileno())
            except BaseException:
                current = destination.stat()
                if (current.st_dev,current.st_ino)==(identity.st_dev,identity.st_ino):
                    destination.unlink()
                raise
    st = destination.stat()
    return st.st_dev, st.st_ino


def save_bundle(workbook, output, report, quality, overwrite=False):
    output = Path(output)
    if output.suffix.lower() != '.xlsx':
        raise ValueError('Выходной файл должен иметь расширение .xlsx')
    output.parent.mkdir(parents=True, exist_ok=True)
    names = (output, output.with_suffix('.metadata.json'), output.with_suffix('.quality.json'))
    if any(path.exists() for path in names) and not overwrite:
        raise ValueError('Результат уже существует; выберите новый --output')
    if overwrite and any(path.exists() for path in names) and not all(path.exists() for path in names):
        raise ValueError('Неполный существующий набор файлов; разберите его вручную')
    published = []
    # Same filesystem, fully prepare payloads before publication. Hard links enforce
    # exclusive creation even if another process produces the same name meanwhile.
    with tempfile.TemporaryDirectory(prefix='.excel-stage-', dir=output.parent) as staging:
        staging = Path(staging)
        staged = [staging/path.name for path in names]
        workbook.save(staged[0])
        digest = hashlib.sha256(staged[0].read_bytes()).hexdigest()
        report['xlsx_sha256'] = quality['xlsx_sha256'] = digest
        for path, data in zip(staged[1:], (report, quality)):
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
        backup = None
        try:
            if overwrite and all(path.exists() for path in names):
                backup = output.parent / (output.stem + '.backup-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
                backup.mkdir()
                for path in names:
                    path.rename(backup / path.name)
            # Publish workbook LAST; visible workbook implies both sidecars were prepared.
            for i in (1, 2, 0):
                identity = publish_exclusive(staged[i], names[i])
                published.append((names[i], identity))
        except OSError:
            for destination, identity in reversed(published):
                if destination.exists():
                    st = destination.stat()
                    if (st.st_dev,st.st_ino) == identity: destination.unlink()
            if backup:
                for path in names:
                    previous = backup / path.name
                    if previous.exists(): previous.rename(path)
                backup.rmdir()
            raise
    # Abrupt process termination can still leave sidecars; never silently overwrite them.
