#!/usr/bin/env python
r"""Chain a full MAROON-X reduction into a single command.

This wraps the nine-step workflow of the reduction tutorial (see
``doc/tutorials/MAROONXDR_Tutorial/example_api.rst``) so a reduction can be
driven end to end, or resumed at any step, without retyping the
``dataselect`` / ``Reduce`` blocks by hand.

Steps, in order:

1. ``debundle``       - split raw GOA bundles into per-arm FITS files
2. ``darks``          - master dark per (exposure time, arm)
3. ``darkcoeffs``     - dark scaling coefficients per arm
4. ``flats``          - master flat per arm
5. ``wavecal``        - dynamic etalon wavelength solution
6. ``syntheticdarks`` - dark interpolated to each science exposure time
7. ``science``        - full echelle extraction of the science frames
8. ``barycor``        - barycentric correction of the reduced spectra
9. ``export``         - merge BLUE + RED into the final science bundle

Nothing is hard-coded to the tutorial dataset: the exposure times of step 2
and the flat recipe of step 4 are read off the files actually present, and a
step whose selection comes up empty is reported and skipped rather than
raising. That keeps the script usable both on a handful of test files and on
a full reduction.

Examples
--------
See what is in a directory before reducing anything::

    python utilities/run_reduction.py /data/mx_test --inventory

Dry run of the whole chain - prints the selection, recipe and parameters of
every step and runs nothing::

    python utilities/run_reduction.py /data/mx_test --dry-run

Reduce a small test set end to end, carrying on past failures::

    python utilities/run_reduction.py /data/mx_test --keep-going

Re-run the science step and everything after it::

    python utilities/run_reduction.py /data/mx_test --steps science-

Resume a long reduction, skipping steps whose products already exist::

    python utilities/run_reduction.py /data/mx_test --resume

Restrict the run to one arm, one night and one target::

    python utilities/run_reduction.py /data/mx_test --arms BLUE \
        --expr 'ut_date=="2025-07-17"' --target HD3651

Override any primitive parameter, scoped to a step::

    python utilities/run_reduction.py /data/mx_test --steps science \
        --param science:combineFibers:max_clips=30

An initialised calibration database is required; see the "Calibration
Database Setup" section of the tutorial.
"""

import argparse
import ast
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import astrodata
from gempy.adlibrary import dataselect
from gempy.utils import logutils
from recipe_system.reduction.coreReduce import Reduce

import maroonx_instruments  # noqa: F401 - registers the MAROON-X AstroData tags

DRPKG = 'maroonxdr'
ALL_ARMS = ('BLUE', 'RED')

# Exposure time tags look like '60s' or '1800s'.
EXPTIME_TAG = re.compile(r'^(\d+)s$')

# Five-letter flat fiber patterns and the recipe that consumes them. 'DFFFD'
# plus 'DDDDF' are combined by makeProcessedFlatDFFFF, 'DFFFD' plus 'FDDDF'
# by the default makeProcessedFlat.
FLAT_PATTERN_RECIPES = {
    'DDDDF': 'makeProcessedFlatDFFFF',
    'FDDDF': 'makeProcessedFlat',
}

# Suffixes the pipeline appends to its outputs. Used to strip a filename back
# to its root when checking whether a per-frame step has already run.
OUTPUT_SUFFIXES = (
    '_reduced',
    '_barycor',
    '_synth_dark',
    '_darkCoefficients',
    '_dark',
    '_flat',
    '_wavecal',
)

# (name, one-line description) in execution order.
STEPS = (
    ('debundle', 'Split raw GOA bundles into per-arm files'),
    ('darks', 'Master dark per exposure time and arm'),
    ('darkcoeffs', 'Dark scaling coefficients per arm'),
    ('flats', 'Master flat per arm'),
    ('wavecal', 'Dynamic etalon wavelength solution'),
    ('syntheticdarks', 'Dark interpolated to each science exposure time'),
    ('science', 'Full echelle extraction of the science frames'),
    ('barycor', 'Barycentric correction of the reduced spectra'),
    ('export', 'Merge BLUE + RED into the final science bundle'),
)
STEP_NAMES = tuple(name for name, _ in STEPS)

# Outcome of a single reduce call.
OK = 'ok'
FAILED = 'failed'
NO_INPUT = 'no input'
DONE = 'done'
DRY_RUN = 'dry run'
SKIPPED = 'skipped'


@dataclass
class StepResult:
    """Outcome of one attempted reduce call."""

    step: str
    unit: str
    status: str
    nfiles: int = 0
    seconds: float = 0.0
    message: str = ''


@dataclass
class RecipeCall:
    """How one step invokes a recipe."""

    name: str = ''
    defaults: dict = field(default_factory=dict)
    suffix: str = ''


@dataclass
class Options:
    """Everything the reduction needs to know, resolved from the CLI."""

    directory: Path
    pattern: str = '*.fits'
    arms: tuple = ALL_ARMS
    exptimes: tuple = ()
    expression: str = ''
    steps: tuple = STEP_NAMES
    flat_recipe: str = 'auto'
    target: str = ''
    simbad_target: str = ''
    multithreading: bool = True
    overrides: dict = field(default_factory=dict)
    dry_run: bool = False
    keep_going: bool = False
    resume: bool = False
    batch: bool = False
    verbose: bool = False


def parse_value(text):
    """
    Convert a command line parameter value to a Python object.

    Parameters
    ----------
    text : str
        Right hand side of a ``--param`` assignment, e.g. ``'20'``,
        ``'[5]'``, ``'True'`` or ``'HD3651'``.

    Returns
    -------
    object
        The literal the text describes, or the text itself when it is not a
        Python literal.
    """
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


def resolve_step(token):
    """
    Resolve one step token to its index in :data:`STEPS`.

    Parameters
    ----------
    token : str
        A step number (``'7'``), a step name (``'science'``) or an
        unambiguous prefix of one (``'sci'``).

    Returns
    -------
    int
        Index of the step in :data:`STEPS`.
    """
    token = token.strip().lower()
    if token.isdigit():
        index = int(token) - 1
        if not 0 <= index < len(STEP_NAMES):
            msg = f'step number out of range (1-{len(STEP_NAMES)}): {token}'
            raise argparse.ArgumentTypeError(msg)
        return index

    matches = [i for i, name in enumerate(STEP_NAMES) if name.startswith(token)]
    if not matches:
        msg = f'unknown step {token!r}; choose from {", ".join(STEP_NAMES)}'
        raise argparse.ArgumentTypeError(msg)
    if len(matches) > 1:
        names = ', '.join(STEP_NAMES[i] for i in matches)
        msg = f'ambiguous step {token!r}; matches {names}'
        raise argparse.ArgumentTypeError(msg)
    return matches[0]


def parse_steps(spec):
    """
    Expand a step specification into the step names it selects.

    Parameters
    ----------
    spec : str
        Comma separated list of steps and ranges, e.g. ``'1-5'``,
        ``'darks,flats'``, ``'science-'``, ``'-flats'`` or ``'all'``.

    Returns
    -------
    tuple of str
        Selected step names, in execution order.
    """
    if spec.strip().lower() in {'all', ''}:
        return STEP_NAMES

    selected = set()
    for token in spec.split(','):
        if not token.strip():
            continue
        if '-' in token:
            start, _, stop = token.partition('-')
            first = resolve_step(start) if start.strip() else 0
            last = resolve_step(stop) if stop.strip() else len(STEP_NAMES) - 1
            selected.update(range(min(first, last), max(first, last) + 1))
        else:
            selected.add(resolve_step(token))
    return tuple(STEP_NAMES[i] for i in sorted(selected))


def parse_overrides(assignments):
    """
    Turn ``--param`` assignments into per-step ``uparms`` dictionaries.

    Parameters
    ----------
    assignments : list of str
        Strings of the form ``STEP:PRIMITIVE:PARAM=VALUE``, where ``STEP``
        may be ``all`` to apply the override to every step.

    Returns
    -------
    dict
        Mapping of step name (or ``'all'``) to a ``{'primitive:param':
        value}`` dictionary suitable for ``Reduce.uparms``.
    """
    overrides = {}
    for assignment in assignments or []:
        key, sep, value = assignment.partition('=')
        parts = key.split(':')
        if not sep or len(parts) != 3:
            msg = f'expected STEP:PRIMITIVE:PARAM=VALUE, got {assignment!r}'
            raise argparse.ArgumentTypeError(msg)
        step, primitive, param = (part.strip() for part in parts)
        if step.lower() != 'all':
            step = STEP_NAMES[resolve_step(step)]
        overrides.setdefault(step, {})[f'{primitive}:{param}'] = parse_value(value)
    return overrides


def normalise_exptimes(values):
    """
    Normalise exposure time arguments to dataselect tags.

    Parameters
    ----------
    values : list of str
        Exposure times as given on the command line, with or without the
        trailing ``s`` (``'300'`` or ``'300s'``).

    Returns
    -------
    tuple of str
        Exposure time tags, ordered by duration.
    """
    tags = [value if value.endswith('s') else f'{value}s' for value in values or []]
    return tuple(sorted(tags, key=lambda tag: int(tag[:-1])))


class Reduction:
    """
    Run the MAROON-X reduction steps over one working directory.

    Each ``step_*`` method drives one step of the tutorial workflow. They all
    funnel their reduce calls through :meth:`run`, which owns the dry-run,
    resume, error and bookkeeping behaviour so the steps stay a readable
    transcription of the tutorial.

    Parameters
    ----------
    options : Options
        Resolved command line options.
    """

    def __init__(self, options):
        self.opts = options
        self.results = []
        self.failures = 0

    # ------------------------------------------------------------------
    # File selection
    # ------------------------------------------------------------------

    def files(self, pattern=None):
        """
        Return the sorted paths matching a glob in the working directory.

        Parameters
        ----------
        pattern : str, optional
            Glob to match. Defaults to the ``--glob`` pattern.

        Returns
        -------
        list of str
            Matching paths, sorted by name.
        """
        directory = self.opts.directory
        return sorted(str(p) for p in directory.glob(pattern or self.opts.pattern))

    def select(self, tags, xtags=(), pattern=None):
        """
        Select the files carrying a set of tags from the working directory.

        Parameters
        ----------
        tags : sequence of str
            Tags every returned file must carry.
        xtags : sequence of str, optional
            Tags that exclude a file from the selection.
        pattern : str, optional
            Glob a candidate must also match, on top of ``--glob``. Steps
            that know the shape of their input (``*_reduced.fits``,
            ``*_barycor.fits``) pass it to narrow the selection; both globs
            apply, so ``--glob`` still restricts every step.

        Returns
        -------
        list of str
            Selected paths.
        """
        candidates = self.files()
        if pattern:
            candidates = sorted(set(candidates) & set(self.files(pattern)))
        kwargs = {'tags': list(tags), 'xtags': list(xtags)}
        if self.opts.expression:
            kwargs['expression'] = dataselect.expr_parser(self.opts.expression)
        return dataselect.select_data(candidates, **kwargs)

    def has_output(self, path, suffix):
        """
        Report whether a per-frame step has already written its output.

        The check globs the input's root name instead of building the exact
        output name, so it holds whether or not the pipeline strips the
        preceding suffix when it writes the new one.

        Parameters
        ----------
        path : str
            Input file of the step.
        suffix : str
            Suffix the step's recipe appends, e.g. ``'_reduced'``.

        Returns
        -------
        bool
            True when a matching output file exists.
        """
        root = Path(path).stem
        for known in OUTPUT_SUFFIXES:
            if root.endswith(known):
                root = root[: -len(known)]
                break
        return bool(self.files(f'{root}*{suffix}.fits'))

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def record(self, result):
        """
        Store and print the outcome of one reduce call.

        Parameters
        ----------
        result : StepResult
            The outcome to record.
        """
        self.results.append(result)
        detail = f' - {result.message}' if result.message else ''
        print(f'  {result.status}{detail}')

    def uparms(self, step, defaults=None):
        """
        Build the ``uparms`` dictionary for one step.

        Parameters
        ----------
        step : str
            Step name, used to pick up its ``--param`` overrides.
        defaults : dict, optional
            The step's built-in parameter overrides.

        Returns
        -------
        dict
            Merged parameters, with ``--param`` taking precedence.
        """
        params = dict(defaults or {})
        params.update(self.opts.overrides.get('all', {}))
        params.update(self.opts.overrides.get(step, {}))
        return params

    def run(self, step, unit, files, recipe=None):
        """
        Run one reduce call over a list of files.

        Parameters
        ----------
        step : str
            Step name.
        unit : str
            Label for what is being reduced, e.g. an arm or a filename.
        files : list of str
            Input files. An empty list is reported and skipped.
        recipe : RecipeCall, optional
            Recipe to run. ``None`` leaves the default recipe for the tags.
        """
        recipe = recipe or RecipeCall()
        print(f'[{step}] {unit}: {len(files)} file(s)')
        self.print_files(files)
        if not files:
            self.record(StepResult(step, unit, NO_INPUT))
            return

        params = self.uparms(step, recipe.defaults)
        print(f'  recipe: {recipe.name or "<default>"}')
        if params:
            print(f'  params: {params}')
        if self.opts.dry_run:
            self.record(StepResult(step, unit, DRY_RUN, len(files)))
            return

        start = time.monotonic()
        try:
            reducer = Reduce()
            reducer.files.extend(files)
            reducer.drpkg = DRPKG
            if recipe.name:
                reducer.recipename = recipe.name
            if params:
                reducer.uparms = params
            reducer.runr()
        except Exception as err:
            self.failures += 1
            message = f'{type(err).__name__}: {err}'
            elapsed = time.monotonic() - start
            self.record(StepResult(step, unit, FAILED, len(files), elapsed, message))
            if not self.opts.keep_going:
                raise
        else:
            elapsed = time.monotonic() - start
            self.record(StepResult(step, unit, OK, len(files), elapsed))

    def run_each(self, step, unit, files, recipe=None):
        """
        Run one reduce call per file, or a single call with ``--batch``.

        The recipes behind these steps work frame by frame, so reducing one
        file per call keeps a single bad frame from taking the rest of the
        group down with it, and lets ``--resume`` restart mid-group.

        Parameters
        ----------
        step : str
            Step name.
        unit : str
            Label for the group, e.g. the arm.
        files : list of str
            Input files.
        recipe : RecipeCall, optional
            Recipe to run. Its ``suffix`` is what ``--resume`` looks for.
        """
        if self.opts.batch or not files:
            self.run(step, unit, files, recipe)
            return

        suffix = recipe.suffix if recipe else ''
        for path in files:
            name = Path(path).name
            if self.opts.resume and suffix and self.has_output(path, suffix):
                print(f'[{step}] {unit} {name}: 1 file(s)')
                self.record(StepResult(step, f'{unit} {name}', DONE, 1))
                continue
            self.run(step, f'{unit} {name}', [path], recipe)

    def print_files(self, files):
        """
        Print the selected filenames, truncated unless ``--verbose``.

        Parameters
        ----------
        files : list of str
            The selected files.
        """
        shown = files if self.opts.verbose else files[:6]
        for path in shown:
            print(f'      {Path(path).name}')
        if len(files) > len(shown):
            print(f'      ... and {len(files) - len(shown)} more')

    def already_done(self, step, unit, tags, xtags=()):
        """
        Report whether a batched step already has its product in place.

        Parameters
        ----------
        step : str
            Step name.
        unit : str
            Label for the group.
        tags : sequence of str
            Tags identifying the step's own output.
        xtags : sequence of str, optional
            Tags excluding files from that check.

        Returns
        -------
        bool
            True when the step is already done and was recorded as such.
        """
        if not self.opts.resume:
            return False
        existing = self.select(tags, xtags=xtags)
        if not existing:
            return False
        print(f'[{step}] {unit}: {len(existing)} existing product(s)')
        self.record(StepResult(step, unit, DONE, len(existing)))
        return True

    # ------------------------------------------------------------------
    # Steps
    # ------------------------------------------------------------------

    def step_debundle(self):
        """Split every raw GOA bundle into one file per arm (step 1)."""
        bundles = self.select(['BUNDLE'])
        if self.opts.resume and self.files('*Z_*.fits'):
            print('[debundle] all: debundled files already present')
            self.record(StepResult('debundle', 'all', DONE, len(bundles)))
            return
        self.run_each('debundle', 'bundle', bundles)

    def step_darks(self):
        """Stack the raw darks per exposure time and arm (step 2)."""
        for arm in self.opts.arms:
            exptimes = self.dark_exptimes(arm)
            if not exptimes:
                self.run('darks', arm, [])
                continue
            for exptime in exptimes:
                unit = f'{arm} {exptime}'
                if self.already_done(
                    'darks',
                    unit,
                    ['PROCESSED', 'DARK', arm, exptime],
                    xtags=['DARK_COEFF', 'DARK_SYNTH'],
                ):
                    continue
                self.run('darks', unit, self.select(['RAW', 'DARK', arm, exptime]))

    def step_darkcoeffs(self):
        """Fit the dark scaling coefficients per arm (step 3)."""
        recipe = RecipeCall(name='makeDarkCoefficients')
        for arm in self.opts.arms:
            if self.already_done('darkcoeffs', arm, ['DARK_COEFF', arm]):
                continue
            darks = self.select(
                ['PROCESSED', 'DARK', arm], xtags=['DARK_COEFF', 'DARK_SYNTH']
            )
            self.run('darkcoeffs', arm, darks, recipe)

    def step_flats(self):
        """Build the master flat per arm (step 4)."""
        for arm in self.opts.arms:
            if self.already_done('flats', arm, ['PROCESSED', 'FLAT', arm]):
                continue
            flats = self.select(['RAW', 'FLAT', arm])
            recipe = RecipeCall(name=self.flat_recipe(flats), suffix='_flat')
            self.run('flats', arm, flats, recipe)

    def step_wavecal(self):
        """Fit the dynamic etalon wavelength solution (step 5)."""
        recipe = RecipeCall(
            name='makeDynamicWavecal',
            defaults={
                'getPeaksAndPolynomials:multithreading': self.opts.multithreading
            },
            suffix='_wavecal',
        )
        for arm in self.opts.arms:
            self.run_each('wavecal', arm, self.select(['RAW', 'ETALON', arm]), recipe)

    def step_syntheticdarks(self):
        """Interpolate a dark for each science exposure time (step 6)."""
        recipe = RecipeCall(name='makeSyntheticDark', suffix='_synth_dark')
        for arm in self.opts.arms:
            self.run_each(
                'syntheticdarks', arm, self.select(['RAW', 'SCI', arm]), recipe
            )

    def step_science(self):
        """Run the full echelle reduction of the science frames (step 7)."""
        recipe = RecipeCall(
            defaults={
                'extractStripes:straylight_removal_fibers': [5],
                'getPeaksAndPolynomials:multithreading': self.opts.multithreading,
                'combineFibers:max_clips': 20,
            },
            suffix='_reduced',
        )
        for arm in self.opts.arms:
            self.run_each('science', arm, self.select(['RAW', 'SCI', arm]), recipe)

    def step_barycor(self):
        """Apply the barycentric correction to the reduced spectra (step 8)."""
        defaults = {}
        if self.opts.target:
            defaults['barycentricCorrection:target_name'] = self.opts.target
        if self.opts.simbad_target:
            key = 'barycentricCorrection:simbad_target_name'
            defaults[key] = self.opts.simbad_target
        recipe = RecipeCall(
            name='applyBarycentricCorrection', defaults=defaults, suffix='_barycor'
        )

        for arm in self.opts.arms:
            reduced = self.select(
                ['PROCESSED_SCIENCE', arm], xtags=['BUNDLE'], pattern='*_reduced.fits'
            )
            self.run_each('barycor', arm, reduced, recipe)

    def step_export(self):
        """
        Merge the BLUE and RED spectra of each observation (step 9).

        Both arms of an observation have to reach ``exportReducedBundle`` in
        the same call, so this step is never split per file.
        """
        if len(self.opts.arms) < len(ALL_ARMS):
            print('[export] all: needs both arms, but --arms restricts the run')
            self.record(
                StepResult('export', 'all', SKIPPED, message='single arm selected')
            )
            return
        if self.opts.resume and self.files('[!0-9]*_reduced.fits'):
            print('[export] all: exported bundles already present')
            self.record(StepResult('export', 'all', DONE))
            return
        barycor = self.select(['PROCESSED', 'BARYCOR'], pattern='*_barycor.fits')
        self.run('export', 'all', barycor, RecipeCall(name='exportReducedBundle'))

    # ------------------------------------------------------------------
    # Discovery helpers
    # ------------------------------------------------------------------

    def dark_exptimes(self, arm):
        """
        Return the exposure time tags of the raw darks present for one arm.

        Parameters
        ----------
        arm : str
            ``'BLUE'`` or ``'RED'``.

        Returns
        -------
        tuple of str
            Exposure time tags, ordered by duration. ``--exptimes``
            short-circuits the discovery when given.
        """
        if self.opts.exptimes:
            return self.opts.exptimes
        found = set()
        for path in self.select(['RAW', 'DARK', arm]):
            found.update(t for t in astrodata.open(path).tags if EXPTIME_TAG.match(t))
        exptimes = tuple(sorted(found, key=lambda tag: int(tag[:-1])))
        if exptimes:
            print(f'[darks] {arm}: exposure times found: {", ".join(exptimes)}')
        return exptimes

    def flat_recipe(self, flats):
        """
        Pick the flat recipe matching the fiber patterns of the raw flats.

        ``makeProcessedFlat`` combines ``DFFFD`` with ``FDDDF``, while
        ``makeProcessedFlatDFFFF`` combines ``DFFFD`` with ``DDDDF``. Which
        one applies is a property of the data, so it is read off the frames
        unless ``--flat-recipe`` says otherwise.

        Parameters
        ----------
        flats : list of str
            Raw flats selected for one arm.

        Returns
        -------
        str
            Recipe name, or an empty string to leave the default recipe in
            place.
        """
        if self.opts.flat_recipe != 'auto':
            return self.opts.flat_recipe
        if not flats:
            return ''

        patterns = {astrodata.open(path).fiber_setup(short=True) for path in flats}
        recipes = {
            FLAT_PATTERN_RECIPES[p] for p in patterns if p in FLAT_PATTERN_RECIPES
        }
        if len(recipes) > 1:
            print(
                f'  warning: flat patterns {sorted(patterns)} need different '
                f'recipes; narrow the selection with --expr or force one with '
                f'--flat-recipe'
            )
            return ''
        if recipes:
            return recipes.pop()
        print(f'  warning: unrecognised flat patterns {sorted(patterns)}')
        return ''


def print_inventory(reduction):
    """
    Print the AstroData tags of every file in the working directory.

    Parameters
    ----------
    reduction : Reduction
        The reduction whose working directory is inspected.
    """
    paths = reduction.files()
    print(f'{len(paths)} file(s) in {reduction.opts.directory}\n')
    counts = {}
    for path in paths:
        name = Path(path).name
        try:
            tags = sorted(astrodata.open(path).tags)
        except Exception as err:  # noqa: BLE001 - report unreadable files, keep going
            print(f'{name:<44} <unreadable: {type(err).__name__}: {err}>')
            continue
        print(f'{name:<44} {" ".join(tags)}')
        for tag in tags:
            counts[tag] = counts.get(tag, 0) + 1

    print('\ntag counts:')
    for tag, count in sorted(counts.items()):
        print(f'  {tag:<20} {count}')


def print_summary(reduction):
    """
    Print the per-call summary table of a finished run.

    Parameters
    ----------
    reduction : Reduction
        The reduction that was run.
    """
    if not reduction.results:
        print('\nNothing ran.')
        return

    print(f'\n{"step":<16}{"unit":<40}{"status":<10}{"files":>6}{"time":>10}')
    print('-' * 82)
    total = 0.0
    counts = {}
    for result in reduction.results:
        unit = result.unit if len(result.unit) <= 39 else f'...{result.unit[-36:]}'
        total += result.seconds
        counts[result.status] = counts.get(result.status, 0) + 1
        print(
            f'{result.step:<16}{unit:<40}{result.status:<10}'
            f'{result.nfiles:>6}{result.seconds:>9.1f}s'
        )
    print('-' * 82)
    tally = ', '.join(f'{status}: {count}' for status, count in sorted(counts.items()))
    print(f'{tally}   total time: {total / 60:.1f} min')

    for result in reduction.results:
        if result.status == FAILED:
            print(f'FAILED  {result.step} {result.unit}: {result.message}')


def preflight(directory):
    """
    Warn about setup problems that would otherwise surface as odd failures.

    Parameters
    ----------
    directory : Path
        The working directory of the reduction.
    """
    dragonsrc = Path.home() / '.dragons' / 'dragonsrc'
    if not dragonsrc.exists():
        print(f'warning: {dragonsrc} not found; calibrations will not be stored')
        return
    text = dragonsrc.read_text()
    if 'databases' not in text:
        print(f'warning: no "databases" entry in {dragonsrc}')
    elif 'store' not in text:
        print(f'warning: no "store" flag in {dragonsrc}; processed calibrations')
        print('         will not be registered and later steps will not find them')
    if not (directory / 'calibrations').exists():
        print(f'note: no calibrations/ in {directory} yet (first run here?)')


def build_parser():
    """
    Build the command line parser.

    Returns
    -------
    argparse.ArgumentParser
        The parser for this script.
    """
    step_help = '\n'.join(
        f'  {i + 1}. {name:<16}{desc}' for i, (name, desc) in enumerate(STEPS)
    )
    parser = argparse.ArgumentParser(
        description=f'Run a MAROON-X reduction.\n\nSteps:\n{step_help}',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('dir', help='working directory holding the raw files')
    parser.add_argument(
        '--steps',
        default='all',
        help='steps to run: names, numbers or ranges, e.g. "1-5", '
        '"darks,flats", "science-" (default: all)',
    )
    parser.add_argument(
        '--arms',
        nargs='+',
        default=list(ALL_ARMS),
        choices=list(ALL_ARMS),
        help='arms to reduce (default: both)',
    )
    parser.add_argument(
        '--exptimes',
        nargs='+',
        default=None,
        metavar='T',
        help='dark exposure times to process, e.g. 300 600 (default: every '
        'exposure time found among the raw darks)',
    )
    parser.add_argument(
        '--expr',
        default='',
        metavar='EXPRESSION',
        help='dataselect expression applied to every selection, e.g. '
        '\'ut_date=="2025-07-17"\'',
    )
    parser.add_argument(
        '--glob',
        default='*.fits',
        help='glob for the candidate files in the working directory '
        '(default: *.fits)',
    )
    parser.add_argument(
        '--target',
        default='',
        help='OBJECT substring selecting the frames to barycentric correct '
        '(default: correct every frame using its own OBJECT)',
    )
    parser.add_argument(
        '--simbad-target',
        default='',
        help='SIMBAD name to resolve, when OBJECT is not resolvable itself',
    )
    parser.add_argument(
        '--flat-recipe',
        default='auto',
        choices=['auto', 'makeProcessedFlat', 'makeProcessedFlatDFFFF'],
        help='flat recipe (default: auto, from the fiber patterns present)',
    )
    parser.add_argument(
        '-p',
        '--param',
        action='append',
        default=[],
        metavar='STEP:PRIMITIVE:PARAM=VALUE',
        help='override a primitive parameter for one step, or for every step '
        'with STEP=all; repeatable',
    )
    parser.add_argument(
        '--no-multithreading',
        action='store_true',
        help='run getPeaksAndPolynomials single threaded',
    )
    parser.add_argument(
        '--batch',
        action='store_true',
        help='pass every frame of a group to one reduce call instead of '
        'reducing frame by frame',
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='skip the steps and frames whose products are already on disk',
    )
    parser.add_argument(
        '--keep-going',
        action='store_true',
        help='carry on with the remaining steps after a failure',
    )
    parser.add_argument(
        '-n',
        '--dry-run',
        action='store_true',
        help='print the selection, recipe and parameters of every step '
        'without reducing anything',
    )
    parser.add_argument(
        '--inventory',
        action='store_true',
        help='print the AstroData tags of every file, then exit',
    )
    parser.add_argument(
        '-v', '--verbose', action='store_true', help='list every input file'
    )
    parser.add_argument(
        '--logfile',
        default='reduce.log',
        help='DRAGONS log file (default: reduce.log)',
    )
    parser.add_argument(
        '--logmode',
        default='standard',
        choices=['quiet', 'standard', 'debug'],
        help='DRAGONS log verbosity (default: standard)',
    )
    return parser


def build_options(args):
    """
    Resolve parsed arguments into an :class:`Options` instance.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line arguments.

    Returns
    -------
    Options
        The resolved options.
    """
    return Options(
        directory=Path(args.dir).expanduser().resolve(),
        pattern=args.glob,
        arms=tuple(args.arms),
        exptimes=normalise_exptimes(args.exptimes),
        expression=args.expr,
        steps=parse_steps(args.steps),
        flat_recipe=args.flat_recipe,
        target=args.target,
        simbad_target=args.simbad_target,
        multithreading=not args.no_multithreading,
        overrides=parse_overrides(args.param),
        dry_run=args.dry_run,
        keep_going=args.keep_going,
        resume=args.resume,
        batch=args.batch,
        verbose=args.verbose,
    )


def print_header(options, args):
    """
    Print what the run is about to do.

    Parameters
    ----------
    options : Options
        Resolved options.
    args : argparse.Namespace
        Parsed arguments, for the logging settings.
    """
    print(f'directory : {options.directory}')
    print(f'steps     : {", ".join(options.steps)}')
    print(f'arms      : {", ".join(options.arms)}')
    if options.expression:
        print(f'expression: {options.expression}')
    print(f'log       : {options.directory / args.logfile} ({args.logmode})')
    if options.dry_run:
        print('dry run   : nothing will be reduced')
    print()
    preflight(options.directory)
    print()


def main(argv=None):
    """
    Run the reduction described by the command line.

    Parameters
    ----------
    argv : list of str, optional
        Argument list. Defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        Exit status: 0 on success, 1 if any reduce call failed.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        options = build_options(args)
    except argparse.ArgumentTypeError as err:
        parser.error(str(err))

    if not options.directory.is_dir():
        parser.error(f'not a directory: {options.directory}')

    # Reduce writes its outputs to the current directory, so the working
    # directory has to be the data directory.
    os.chdir(options.directory)
    logutils.config(file_name=args.logfile, mode=args.logmode)

    reduction = Reduction(options)
    if args.inventory:
        print_inventory(reduction)
        return 0

    print_header(options, args)
    try:
        for step in options.steps:
            getattr(reduction, f'step_{step}')()
    finally:
        print_summary(reduction)

    return 1 if reduction.failures else 0


if __name__ == '__main__':
    sys.exit(main())
