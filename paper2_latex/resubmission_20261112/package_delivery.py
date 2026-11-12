"""Build the complete handoff and the journal-facing submission archive."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import zipfile

ROOT = Path(__file__).resolve().parent
TITLE = 'Separating coverage gains from error targeting in molecular prediction sets'
REPOSITORY = 'DCarchimonde/aidd-paper-factory'
BRANCH = 'paper2-tame-resubmission-20261112'
EXCLUDED_PARTS = {'qa', '__pycache__', '.git'}
EXCLUDED_SUFFIXES = {'.log', '.aux', '.out', '.toc', '.pyc'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def project_files():
    return sorted(p for p in ROOT.rglob('*') if p.is_file()
                  and not EXCLUDED_PARTS.intersection(p.relative_to(ROOT).parts)
                  and p.suffix not in EXCLUDED_SUFFIXES and p.name != '.gitignore')


def archive(destination, files, texts=None):
    """Files maps archive names to local paths; texts supplies generated readmes."""
    with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED) as z:
        for name, path in sorted(files.items()):
            z.write(path, name)
        for name, content in sorted((texts or {}).items()):
            z.writestr(name, content)
    with zipfile.ZipFile(destination) as z:
        assert z.testzip() is None
        assert len(z.namelist()) == len(set(z.namelist()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--commit', required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    assert out != ROOT and ROOT not in out.parents, 'Use an output directory outside the project.'
    out.mkdir(parents=True, exist_ok=True)
    assert (ROOT / 'evidence/all_cell_metrics.csv.gz').is_file()
    assert json.loads((ROOT / 'RUNTIME_SOURCE_VERIFICATION.json').read_text())['status'] == 'PASS'
    paths = project_files()

    main_sources = ['manuscript.tex', 'preamble.tex', 'results_discussion.tex', 'references.tex',
                    'tables/historical.tex', 'tables/activation.tex',
                    'figures/assay_coverage.pdf', 'figures/workload_tradeoff.pdf']
    archive(out / 'Manuscript-Source.zip', {n: ROOT / n for n in main_sources}, {
        'README.txt': TITLE + '\n\nMain compilation entry: manuscript.tex\n'
        'Compile twice with pdflatex manuscript.tex. All referenced source files and figures are included.\n'
    })

    data_files = {}
    for p in paths:
        rel = p.relative_to(ROOT)
        if rel.parts[0] in {'inputs', 'evidence', 'tables', 'figures', 'runtime_sources'}:
            if rel.as_posix() != 'runtime_sources/README.md':
                data_files[rel.as_posix()] = p
    for n in ['build_results.py', 'check_activation_identity.py', 'requirements.txt',
              'NUMERICAL_VERIFICATION.json', 'ACTIVATION_IDENTITY_CHECK.json',
              'RUNTIME_SOURCE_VERIFICATION.json']:
        data_files[n] = ROOT / n
    runtime_readme = (ROOT / 'runtime_sources/README.md').read_text()
    runtime_readme = runtime_readme.split('## Verification')[0] + '''## Verification

The root RUNTIME_SOURCE_VERIFICATION.json records the source matches and 21 passing checks.
verification_tests.txt preserves their output. The checks cover KMM and weighted-set behavior,
auxiliary-pool chemistry, a constrained quadratic optimum and objective units, the scaffold
retry, and CUDA state conversion with mock states.

The full experiment suite and verification test modules are in the repository linked by the
root README.txt. Fixed-input table and figure reconstruction is available directly in this
archive through build_results.py and check_activation_identity.py.
'''
    data_readme = f'''{TITLE}
Supporting data and reproducibility materials

Contents
- inputs/: frozen aggregates and provenance used to build the paper.
- evidence/: complete returned cell metrics, panel metrics and paired contrasts.
- tables/ and figures/: the paper's numerical tables and figures.
- runtime_sources/: executed CoDrug/KMM-CP sources, references and identified patches.

Rebuild the numerical tables and figures with Python 3.10 or newer:
    python -m pip install -r requirements.txt
    python build_results.py
    python check_activation_identity.py

The scripts reconstruct the reported summaries from the frozen aggregate inputs.
Model fitting and molecule-cluster resampling use the full experiment suite and its
row-level exports. Source hashes and transformations are in inputs/SOURCE_MANIFEST.json.

Versioned project and experiment suite:
https://github.com/{REPOSITORY}/tree/{args.commit}
Manuscript project: paper2_latex/resubmission_20261112/
Experiment suite: paper2_admet_benchmark/revision_2026/
'''
    archive(out / 'Supporting-Data.zip', data_files,
            {'README.txt': data_readme, 'runtime_sources/README.md': runtime_readme})

    upload_map = {
        'Manuscript.pdf': ROOT / 'manuscript.pdf',
        'Supporting-Information.pdf': ROOT / 'supplementary.pdf',
        'Cover-Letter.pdf': ROOT / 'cover_letter.pdf',
        'Response-to-Reviewer.pdf': ROOT / 'response.pdf',
        'Highlights.docx': ROOT / 'submission_assets/Highlights.docx',
        'Declaration-of-Interest.docx': ROOT / 'submission_assets/Declaration-of-Interest.docx',
        'Figure-1.pdf': ROOT / 'figures/assay_coverage.pdf',
        'Figure-2.pdf': ROOT / 'figures/workload_tradeoff.pdf',
        'Manuscript-Source.zip': out / 'Manuscript-Source.zip',
        'Supporting-Data.zip': out / 'Supporting-Data.zip',
    }
    for name, source in list(upload_map.items()):
        if source != out / name:
            shutil.copyfile(source, out / name)
        upload_map[name] = out / name
    archive(out / 'TAME-Submission.zip', upload_map)

    archive(out / 'Manuscript-Project.zip', {p.relative_to(ROOT).as_posix(): p for p in paths})
    for name in ['Submission-Instructions.txt', 'Decision-Letter.txt']:
        shutil.copyfile(ROOT / name, out / name)

    verification_names = ['NUMERICAL_VERIFICATION.json', 'ACTIVATION_IDENTITY_CHECK.json',
                          'BUILD_VERIFICATION.json', 'VISUAL_QA.json',
                          'RUNTIME_SOURCE_VERIFICATION.json', 'FROZEN_CONTENT_VERIFICATION.json',
                          'EDITORIAL_VERIFICATION.json', 'RESUBMISSION_VERIFICATION.json']
    receipt = {
        'title': TITLE, 'journal': 'Chemometrics and Intelligent Laboratory Systems',
        'original_manuscript_number': 'CHEMOLAB-D-26-01118',
        'submission_route': 'Substantially revised paper submitted as a new submission',
        'repository': REPOSITORY, 'branch': BRANCH, 'commit': args.commit,
        'status': 'READY_FOR_SUBMISSION_HANDOFF', 'submitted_to_journal': False,
        'authors': [
            {'name': 'Siyuan Tong', 'order': 1, 'equal_first_author': True, 'corresponding_author': True},
            {'name': 'Yuechen Wang', 'order': 2, 'equal_first_author': True, 'corresponding_author': False}],
        'corresponding_email': '25064241@siswa.um.edu.my',
        'files': {n: {'sha256': sha(p), 'bytes': p.stat().st_size} for n, p in upload_map.items()},
        'archives': {n: {'sha256': sha(out / n), 'bytes': (out / n).stat().st_size}
                     for n in ['TAME-Submission.zip', 'Manuscript-Project.zip']},
        'verification': {n: json.loads((ROOT / n).read_text()) for n in verification_names},
    }
    (out / 'Delivery-Receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    complete_files = ['TAME-Submission.zip', 'Manuscript-Project.zip',
                      'Submission-Instructions.txt', 'Decision-Letter.txt', 'Delivery-Receipt.json']
    archive(out / 'TAME-Complete.zip', {n: out / n for n in complete_files})
    print(json.dumps([{'file': str(out / n), 'bytes': (out / n).stat().st_size,
                       'sha256': sha(out / n)} for n in ['TAME-Complete.zip', 'TAME-Submission.zip']]))


if __name__ == '__main__':
    main()
