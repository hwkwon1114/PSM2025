"""Public case labels for the current joint calibration.

These are presentation aliases, not renamed datasets or historical IDs.
"""
PUBLIC_TO_ARCHIVE = {'run1': 'run6', 'run2': 'run9', 'run3': 'run10'}


def archive_case_id(public_id):
    """Resolve a public Run 1/2/3 label to its immutable historical ID."""
    key = str(public_id).strip().lower().replace(' ', '').replace('_', '')
    if key not in PUBLIC_TO_ARCHIVE:
        raise ValueError(f'Unknown public case {public_id!r}; use Run 1, Run 2 or Run 3')
    return PUBLIC_TO_ARCHIVE[key]


def public_case_label(archive_id):
    """Return the public Run 1/2/3 label for an exact historical ID."""
    for public, archived in PUBLIC_TO_ARCHIVE.items():
        if archive_id == archived:
            return f'Run {public[3:]}'
    raise ValueError(f'Historical case {archive_id!r} has no public alias')


if __name__ == '__main__':
    for public, archived in PUBLIC_TO_ARCHIVE.items():
        print(f'{public_case_label(archived)} -> {archived}')
