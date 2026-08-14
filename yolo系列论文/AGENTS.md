# Repository Guidelines

This repository is a curated library of YOLO (You Only Look Once) object-detection papers in PDF form, collected as reference material for the 2026 China Mathematical Contest in Modeling (选题 D). It contains no application code, test suite, or build tooling; contributions consist of adding, correcting, and organizing PDF documents.

## Project Structure & Module Organization

All papers currently live flat at the repository root. If the collection grows, group files under `papers/` (official YOLO series), `papers/variants/` (PP-, DAMO-, Gold-, Scaled-, YOLOX), and `papers/reviews/`.

## Naming Conventions

File names follow the pattern `<Model>_<ShortDescription>.pdf`:

- Official series: `YOLOv7_Trainable_Bag_of_Freebies.pdf`
- Vendor variants: `PP_YOLOE_Plus_Practical_Method_Accelerate.pdf`
- Reviews/evolution summaries: `Ultralytics_YOLO_Evolution_YOLO26_YOLO11_YOLOv8_YOLOv5.pdf`
- arXiv preprints: rename to `arXiv_<ID>_<ShortDescription>.pdf` (e.g., `arXiv_2602.14582_...pdf`)

Use underscores instead of spaces, keep descriptions short, and never rename a paper's version.

## Document Management Workflow

To add a paper:

1. Download the official version (arXiv or publisher PDF).
2. Name it per the conventions above.
3. Check for duplicates by title and version before adding.
4. If a `README.md` index exists, add a one-line entry describing the paper.

## Coding Style

No formatter or linting configuration exists. When adding Markdown or scripts, use two-space indentation, keep filenames ASCII, and prefer plain English in documentation.

## Commit & Pull Request Guidelines

The directory is not yet under Git version control. When a repository is initialized, follow these conventions:

- Use Conventional Commits, e.g., `docs: add YOLOv13 hypergraph paper` or `chore: rename arXiv preprint`.
- Make one logical change per commit; state which paper was added, renamed, or corrected.
- In pull request descriptions, list the papers changed, note duplicate checks performed, and confirm each PDF opens. Screenshots are unnecessary.

## Review Checklist

Before merging a contribution, verify that:

- The PDF opens and is complete (not truncated or a low-quality scan).
- The filename follows the conventions and contains no duplicate title/version.
- The `README.md` index is updated, if present.
- The file size stays reasonable (under ~50 MB) to keep the repository lightweight.
