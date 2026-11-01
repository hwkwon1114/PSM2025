"""Solver-free export controls: data order, overwrite refusal and workbook structure."""
import csv
import json
import zipfile
import xml.etree.ElementTree as ET

import pytest

from calibration.ewcal.common import digest
from shape_design.export_illustrated import export, load_passes


@pytest.fixture
def csv_source(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    entries = []
    for number in range(1, 17):
        file = source / f'pass_{number:02d}.csv'
        with file.open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['point_index', 'x_mm', 'y_mm', 'z_mm'])
            writer.writerows([[0, -10, 10, 0], [1, 10, 114, 0]])
        entries.append(dict(k=0, **{'pass': number}, round=(number-1)//4+1,
                            direction='F' if number % 4 in (1, 2) else 'C',
                            face='front' if number % 4 in (1, 0) else 'back',
                            segments=1, wheel_band_width_mm=2, file=file.name, csv_sha256=digest(file)))
    (source / 'manifest.json').write_text(json.dumps(entries))
    return source


def test_export_preserves_points_and_refuses_overwrite(csv_source, tmp_path):
    out = tmp_path / 'illustrated'
    path = export(csv_source, out, 0)
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None
        ns = {'m': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
        sheets = ET.fromstring(archive.read('xl/workbook.xml')).findall('.//m:sheet', ns)
        assert len(sheets) == 18
        assert len([p for p in archive.namelist() if p.startswith('xl/media/')]) == 16
        # Read me, Pass order, Pass 01: the data row must retain the historical CSV axes.
        cells = ET.fromstring(archive.read('xl/worksheets/sheet3.xml'))
        assert [float(c.find('m:v', ns).text) for c in cells.findall('.//m:row[@r="4"]/m:c', ns)] == [0, -10, 10, 0]
    before = digest(path)
    with pytest.raises(FileExistsError):
        export(csv_source, out, 0)
    assert digest(path) == before


def test_modified_csv_fails_before_creating_output(csv_source, tmp_path):
    (csv_source / 'pass_01.csv').write_text('changed')
    out = tmp_path / 'illustrated'
    with pytest.raises(ValueError, match='Changed source CSV'):
        export(csv_source, out, 0)
    assert not out.exists()


def test_missing_pass_is_not_silently_skipped(csv_source):
    file = csv_source / 'manifest.json'
    entries = json.loads(file.read_text())
    file.write_text(json.dumps(entries[1:]))
    with pytest.raises(ValueError, match='16 ordered'):
        load_passes(csv_source, 0)
