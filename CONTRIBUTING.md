<!--contributing-start-->

# Contributing

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given.

Here are some ways you can contribute:

## Types of Contributions

### Report Bugs

Report bugs at [https://github.com/antarctica/canari-ml/issues](https://github.com/antarctica/canari-ml/issues).

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

### Fix Bugs

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help
wanted" is open to whoever wants to implement it.

### Implement Features

Look through the GitHub issues for features. Anything tagged with "enhancement"
and "help wanted" is open to whoever wants to implement it.

### Write Documentation

`canari-ml` could always use more documentation, whether as part of the
official `canari-ml` docs, in docstrings, or even on the web in blog posts,
articles, and such.

### Submit Feedback

The best way to send feedback is to file an issue at https://github.com/antarctica/canari-ml/issues.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions
  are welcome :)

## Developer Install

To install all dependencies including testing and documentation, clone the code as normal,
change directory into it, then run:

``` bash
pip install -e .[dev,docs]
```

<!--contributing-end-->

<!--releases-start-->

# Releases

We use the [PEP440](https://peps.python.org/pep-0440/) standard for release versions. This section guides you through how to release new versions as a maintainer.

## Generating releases

To aid in managing automated changelogs and package version control, this codebase uses the [Commitizen](https://commitizen-tools.github.io/commitizen/) package (installed as dev dependency). Please note that Commitizen has an automated approach of inferring what semantic versioning level the next bump should be (See last sub-section of this page).

To bump to the next stable version:

```bash
cz bump
```

Examples:

```
v0.0.1 → v0.0.2
```

## Create a Pre-release

Commitizen supports alpha, beta, and release candidate (RC) pre-releases using the --prerelease flag.

### Alpha (a)

```bash
cz bump --prerelease alpha
```

Examples:

```
v1.0.4 → v1.0.5a0
```

### Beta (b)

```bash
cz bump --prerelease beta
```

Examples:

```
v1.0.5a2 → v1.0.5b0
```

### Release Candidate (rc)

```bash
cz bump --prerelease rc
```

Examples:

```
v1.0.5b2 → v1.0.5rc0
```

### Explicit patch-level prerelease

Commitizen will automatically infer the SemVer level based on the commit history.
While this can be nice, you might want to define your own release definition like
I have been doing till hitting production stage.

To start a new patch-level prerelease explicitly rather than letting commitizen
infer it:

```bash
cz bump --increment patch --prerelease alpha
```

Examples:

```
v1.2.4a3 → v1.2.5a0
```

In this case, without specifying we want to increment the patch number, it will increment the alpha version instead!

<div class="result">
  <details class="admonition note" open="true">
    <summary class="admonition-title"><strong>Note</strong></summary>
    <p>This project follows PEP 440 versioning. Pre-releases use a (alpha), b (beta), or rc (release candidate)
    without hyphens or dots (e.g., 1.2.5a1 not 1.2.5-alpha.1) unlike SemVer.</p>
    <p>Examples: 1.2.5a1, 1.2.5b2, 1.2.5rc1, 1.2.5.dev0</p>
  </details>
</div>

For full documentation on its usage, please peruse the [Commitizen docs](https://commitizen-tools.github.io/commitizen/commands/bump/).

<!--releases-end-->

