import pytest
from calibration.case_labels import archive_case_id, public_case_label


def test_public_aliases_preserve_historical_identity():
    assert [archive_case_id(label) for label in ('Run 1', 'run_2', 'RUN3')] == ['run6', 'run9', 'run10']
    assert [public_case_label(label) for label in ('run6', 'run9', 'run10')] == ['Run 1', 'Run 2', 'Run 3']


def test_unknown_ids_are_not_silently_reassigned():
    with pytest.raises(ValueError):
        archive_case_id('run6')
    with pytest.raises(ValueError):
        public_case_label('run7')
