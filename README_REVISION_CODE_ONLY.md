# Reviewer Revision Code-Only Upload

This package is intentionally **code only**.

It contains:
- reviewer-requested robustness and sensitivity analysis scripts;
- bulk-carrier ERA5 harmonisation code;
- tanker ERA5 download/interpolation/harmonisation and C4 audit code;
- lightweight diagnostic tools;
- the revision-specific Python requirements file.

It intentionally excludes:
- raw or processed AIS/fuel data;
- ERA5 NetCDF files;
- derived row-level datasets;
- CSV/JSON result tables and manifests;
- model binaries, NumPy prediction arrays, and checkpoints;
- figures and other generated outputs.

## Suggested repository placement

Merge the package contents into the existing repository root:

    src/
      paper_analysis/
        reviewer_revision/
      environment/
        tanker/
    tools/

Do not delete or replace the repository's existing canonical preprocessing and core-analysis scripts.

## Inputs

Several analyses require restricted study inputs or outputs from the canonical pipeline.
Those inputs are supplied through CLI arguments and/or environment variables and are
not included in this public package.

## Release policy

Create the final GitHub release/tag only after these code files have been merged into
the public repository and the repository-level README / requirements have been checked.

The manuscript Code Availability statement should cite the exact published release tag.
