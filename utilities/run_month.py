#!/usr/bin/env python
r"""Reduce a month of MAROON-X data into per-night folders.

Groups raw GOA bundles by Hawaii observing night, symlinks each bundle into a
working folder, and drives :mod:`run_reduction` over those folders in
dependency order: master darks and flats first, then each night's etalons and
science.

Layout produced under the reduction directory::

    <out>/
    |-- MaroonX_masterframe/
    |   |-- 2025121x/          master darks + dark coefficients
    |   `-- 202512xx/          master flats
    |-- 2025-12-18/            etalons + science of that night
    |-- 2025-12-21/
    `-- ...

Master frame folder names follow the legacy wildcard convention (see
``doc/usermanuals/MAROONXDR_UserManual/legacy.rst``): trailing day digits are
replaced by ``x`` to cover the range of dates in the stack - one digit for
darks, two for flats, widened automatically if the dates need it. Filenames
are never touched, so ``caldb`` keeps exactly the paths DRAGONS wrote.

A night is delimited noon to noon in Hawaii time, so the evening and
morning calibrations that bracket a night stay with the science taken between
them. Calibrations are shared through the single ``caldb``: a night with no
darks or flats of its own resolves them from the master frame folders.

Examples
--------
Print the plan, creating and reducing nothing::

    python utilities/run_month.py --raw /data10/MaroonX_spectra/raw --dry-run

Stage and reduce a month into the current directory::

    python utilities/run_month.py --raw /data10/MaroonX_spectra/raw \
        --target HD3651

Re-run, skipping whatever already landed::

    python utilities/run_month.py --raw /data10/MaroonX_spectra/raw --resume

Stage the folders but stop before reducing, to inspect them first::

    python utilities/run_month.py --raw /data10/MaroonX_spectra/raw --stage-only
"""

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path

import astrodata

import maroonx_instruments  # noqa: F401 - registers the MAROON-X AstroData tags

RUN_REDUCTION = Path(__file__).with_name('run_reduction.py')

# Hawaii is UTC-10; adding the 12 hour noon boundary puts a whole observing
# night, evening through morning, under one label.
NIGHT_SHIFT = timedelta(hours=22)

MASTERFRAME_DIRNAME = 'MaroonX_masterframe'

# Wildcard digits per master frame kind, and the steps each folder runs.
DARK = 'dark'
FLAT = 'flat'
NIGHT = 'night'

WILDCARD_DIGITS = {DARK: 1, FLAT: 2}

FOLDER_STEPS = {
    DARK: ('debundle', 'darks', 'darkcoeffs'),
    FLAT: ('debundle', 'flats'),
    NIGHT: (
        'debundle',
        'wavecal',
        'syntheticdarks',
        'science',
        'barycor',
        'export',
    ),
}

# Master frames have to exist before any night's science can resolve them.
PASS_ORDER = (DARK, FLAT, NIGHT)


@dataclass
class Frame:
    """One raw GOA bundle and the grouping keys read from its header."""

    path: Path
    kind: str
    ut_date: str
    night: str


@dataclass
class Folder:
    """A working folder, the bundles staged into it, and the steps it runs."""

    name: str
    path: Path
    kind: str
    frames: list = field(default_factory=list)

    @property
    def steps(self):
        """Return the reduction steps this folder's kind runs."""
        return FOLDER_STEPS[self.kind]


@dataclass
class FolderResult:
    """Outcome of reducing one folder."""

    folder: str
    kind: str
    nframes: int
    status: str
    seconds: float = 0.0


def classify(tags):
    """
    Classify a raw bundle by its AstroData tags.

    Parameters
    ----------
    tags : set of str
        Tag set of the bundle.

    Returns
    -------
    str or None
        ``'dark'``, ``'flat'``, ``'etalon'``, ``'science'``, or None when the
        frame is not one the month workflow places.
    """
    if 'BUNDLE' not in tags:
        return None
    if 'FLAT' in tags:
        return FLAT
    if 'DARK' in tags and not {'DARK_COEFF', 'DARK_SYNTH'} & tags:
        return DARK
    if 'ETALON' in tags:
        return 'etalon'
    if 'SCI' in tags:
        return 'science'
    return None


def read_frames(raw_dir, pattern):
    """
    Read every raw bundle in a directory and derive its grouping keys.

    Parameters
    ----------
    raw_dir : Path
        Directory holding the raw GOA bundles.
    pattern : str
        Glob selecting candidate files.

    Returns
    -------
    tuple
        ``(frames, skipped)`` - the classified frames, and a list of
        ``(name, reason)`` for files the workflow does not place.
    """
    frames = []
    skipped = []
    for path in sorted(raw_dir.glob(pattern)):
        try:
            ad = astrodata.open(path)
            kind = classify(set(ad.tags))
            when = ad.ut_datetime()
        except Exception as err:  # noqa: BLE001 - report and keep scanning
            skipped.append((path.name, f'{type(err).__name__}: {err}'))
            continue
        if kind is None:
            skipped.append((path.name, 'not a placeable bundle'))
            continue
        night = (when - NIGHT_SHIFT).date().isoformat()
        frames.append(Frame(path, kind, when.strftime('%Y%m%d'), night))
    return frames, skipped


def wildcard_label(ut_dates, digits):
    """
    Collapse a set of UT dates into a legacy-style wildcard label.

    Trailing day digits become ``x`` until every date in the group shares the
    remaining prefix, so the label never claims a narrower range than the
    stack actually spans.

    Parameters
    ----------
    ut_dates : set of str
        ``YYYYMMDD`` strings of the frames in the group.
    digits : int
        Minimum number of digits to wildcard.

    Returns
    -------
    str
        Label such as ``'2025121x'`` or ``'202512xx'``.
    """
    for width in range(digits, 9):
        prefixes = {date[: 8 - width] for date in ut_dates}
        if len(prefixes) == 1:
            return prefixes.pop() + 'x' * width
    return 'x' * 8


def plan(frames, out_dir, masterframe_dir):
    """
    Group frames into the folders that will be created and reduced.

    Darks and flats are grouped into master frame folders labelled by the
    wildcard convention; etalons and science are grouped per observing night.

    Parameters
    ----------
    frames : list of Frame
        Classified raw bundles.
    out_dir : Path
        Reduction directory holding the night folders.
    masterframe_dir : Path
        Directory holding the master frame folders.

    Returns
    -------
    list of Folder
        Folders in dependency order: darks, then flats, then nights.
    """
    folders = []

    for kind in (DARK, FLAT):
        members = [f for f in frames if f.kind == kind]
        if not members:
            continue
        label = wildcard_label({f.ut_date for f in members}, WILDCARD_DIGITS[kind])
        folders.append(Folder(label, masterframe_dir / label, kind, members))

    nights = {}
    for frame in frames:
        if frame.kind in (DARK, FLAT):
            continue
        nights.setdefault(frame.night, []).append(frame)
    for night in sorted(nights):
        folders.append(Folder(night, out_dir / night, NIGHT, nights[night]))

    return folders


def stage(folder, *, dry_run=False):
    """
    Create a folder and symlink its bundles into it.

    Symlinks avoid duplicating the raw data while still letting DRAGONS write
    its outputs into the folder. Existing links are left alone, so staging is
    safe to repeat.

    Parameters
    ----------
    folder : Folder
        The folder to stage.
    dry_run : bool
        When True, report what would be linked and create nothing.

    Returns
    -------
    int
        Number of links created.
    """
    created = 0
    if not dry_run:
        folder.path.mkdir(parents=True, exist_ok=True)
    for frame in folder.frames:
        link = folder.path / frame.path.name
        if link.exists() or link.is_symlink():
            continue
        created += 1
        if not dry_run:
            link.symlink_to(frame.path.resolve())
    return created


def reduce_folder(folder, args):
    """
    Run :mod:`run_reduction` over one folder as a subprocess.

    A subprocess keeps each folder's DRAGONS logging and working directory
    independent, and keeps a crash in one night from taking the month down.

    Parameters
    ----------
    folder : Folder
        The folder to reduce.
    args : argparse.Namespace
        Parsed month-level options.

    Returns
    -------
    FolderResult
        Outcome of the run.
    """
    command = [
        sys.executable,
        str(RUN_REDUCTION),
        str(folder.path),
        '--steps',
        ','.join(folder.steps),
        '--logfile',
        str(folder.path / 'reduce.log'),
    ]
    if args.target:
        command += ['--target', args.target]
    if args.resume:
        command.append('--resume')
    if args.keep_going:
        command.append('--keep-going')
    if args.verbose:
        command.append('--verbose')

    print(
        f'\n{"=" * 78}\n== {folder.kind}: {folder.name} '
        f'({len(folder.frames)} bundles)\n{"=" * 78}'
    )
    print(' '.join(command))
    if args.dry_run:
        return FolderResult(folder.name, folder.kind, len(folder.frames), 'dry run')

    start = time.monotonic()
    completed = subprocess.run(command, check=False)  # noqa: S603 - fixed argv
    elapsed = time.monotonic() - start
    status = 'ok' if completed.returncode == 0 else f'exit {completed.returncode}'
    return FolderResult(folder.name, folder.kind, len(folder.frames), status, elapsed)


def print_plan(folders, skipped):
    """
    Print the folder plan before anything is created.

    Parameters
    ----------
    folders : list of Folder
        Planned folders.
    skipped : list of tuple
        ``(name, reason)`` for files that will not be placed.
    """
    print(f'{"folder":<26}{"kind":<9}{"bundles":>8}  steps')
    print('-' * 78)
    for folder in folders:
        print(
            f'{folder.name:<26}{folder.kind:<9}{len(folder.frames):>8}  '
            f'{",".join(folder.steps)}'
        )
    print('-' * 78)
    print(f'{len(folders)} folder(s), {sum(len(f.frames) for f in folders)} bundle(s)')
    if skipped:
        print(f'\nnot placed ({len(skipped)}):')
        for name, reason in skipped:
            print(f'  {name:<40} {reason}')


def print_summary(results):
    """
    Print the month-level summary of every folder that ran.

    Parameters
    ----------
    results : list of FolderResult
        Outcomes in the order they ran.
    """
    if not results:
        print('\nNothing ran.')
        return
    print(f'\n{"folder":<26}{"kind":<9}{"bundles":>8}{"status":>10}{"time":>10}')
    print('-' * 78)
    total = 0.0
    for result in results:
        total += result.seconds
        print(
            f'{result.folder:<26}{result.kind:<9}{result.nframes:>8}'
            f'{result.status:>10}{result.seconds / 60:>9.1f}m'
        )
    print('-' * 78)
    failed = [r for r in results if r.status not in ('ok', 'dry run')]
    print(
        f'{len(results)} folder(s), {len(failed)} failed, '
        f'total {total / 60:.1f} min'
    )
    for result in failed:
        print(f'FAILED  {result.kind} {result.folder}: {result.status}')


def stage_all(folders, *, dry_run=False):
    """
    Stage every folder and report how many links each one gained.

    Parameters
    ----------
    folders : list of Folder
        Folders to stage.
    dry_run : bool
        When True, create nothing.
    """
    for folder in folders:
        created = stage(folder, dry_run=dry_run)
        verb = 'would link' if dry_run else 'linked'
        print(f'{folder.name:<26} {verb} {created}/{len(folder.frames)} bundle(s)')


def build_parser():
    """
    Build the command line parser.

    Returns
    -------
    argparse.ArgumentParser
        The parser for this script.
    """
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n\n')[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--raw', required=True, help='directory holding the raw GOA bundles'
    )
    parser.add_argument(
        '--out',
        default='.',
        help='reduction directory for the night folders (default: cwd)',
    )
    parser.add_argument(
        '--masterframe-dir',
        default=None,
        help=f'directory for the master frame folders '
        f'(default: <out>/{MASTERFRAME_DIRNAME})',
    )
    parser.add_argument(
        '--glob',
        default='N*.fits',
        help='glob selecting raw bundles in --raw (default: N*.fits)',
    )
    parser.add_argument(
        '--nights',
        nargs='+',
        default=None,
        metavar='DATE',
        help='only place and reduce these observing nights, e.g. 2025-12-21',
    )
    parser.add_argument(
        '--target',
        default='',
        help='OBJECT substring passed to the barycentric correction',
    )
    parser.add_argument(
        '--plan-only', action='store_true', help='print the plan and exit'
    )
    parser.add_argument(
        '--stage-only',
        action='store_true',
        help='create the folders and symlinks, then stop before reducing',
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='pass --resume to every folder, skipping products already on disk',
    )
    parser.add_argument(
        '--keep-going',
        action='store_true',
        help='pass --keep-going to every folder, and carry on to the next '
        'folder after a failure',
    )
    parser.add_argument(
        '-n',
        '--dry-run',
        action='store_true',
        help='print the plan and the commands, create and reduce nothing',
    )
    parser.add_argument(
        '-v', '--verbose', action='store_true', help='pass --verbose to every folder'
    )
    return parser


def main(argv=None):
    """
    Plan, stage and reduce a month of data.

    Parameters
    ----------
    argv : list of str, optional
        Argument list. Defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        Exit status: 0 when every folder succeeded, 1 otherwise.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    raw_dir = Path(args.raw).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    masterframe_dir = (
        Path(args.masterframe_dir).expanduser().resolve()
        if args.masterframe_dir
        else out_dir / MASTERFRAME_DIRNAME
    )
    if not raw_dir.is_dir():
        parser.error(f'not a directory: {raw_dir}')

    print(f'raw         : {raw_dir}')
    print(f'reduction   : {out_dir}')
    print(f'master frame: {masterframe_dir}')
    print('night       : noon-to-noon HST (UTC-10)\n')

    frames, skipped = read_frames(raw_dir, args.glob)
    if not frames:
        print(f'no placeable bundles matching {args.glob!r} in {raw_dir}')
        return 1

    folders = plan(frames, out_dir, masterframe_dir)
    if args.nights:
        wanted = set(args.nights)
        folders = [f for f in folders if f.kind != NIGHT or f.name in wanted]
    print_plan(folders, skipped)
    if args.plan_only:
        return 0

    print()
    stage_all(folders, dry_run=args.dry_run)
    if args.stage_only:
        print('\nstaged only, nothing reduced')
        return 0

    results = []
    try:
        for kind in PASS_ORDER:
            for folder in folders:
                if folder.kind != kind:
                    continue
                result = reduce_folder(folder, args)
                results.append(result)
                if result.status not in ('ok', 'dry run') and not args.keep_going:
                    return 1
    finally:
        print_summary(results)

    return 1 if any(r.status not in ('ok', 'dry run') for r in results) else 0


if __name__ == '__main__':
    sys.exit(main())
