---
type: howto
category: how_to
tags: [deployment, github-pages, ci]
---

# GitHub Pages Deployment Workflow

Documentation is built and published to GitHub Pages automatically by the
repository's own workflow file,
[`.github/workflows/docs.yml`](https://github.com/noaa-oar-arl/monet-stats/blob/main/.github/workflows/docs.yml).
You do not need to create it — this page describes what it does.

## Triggers

- A `push` to `main` — builds **and** deploys the site.
- A `pull_request` targeting `main` — builds the site to validate it, without
  deploying.

## Workflow Configuration

```yaml
name: Deploy Documentation to GitHub Pages

on:
  push:
    branches:
      - main
  pull_request:
    branches:
      - main

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: "pages"
  cancel-in-progress: false

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
    - name: Checkout
      uses: actions/checkout@v6
    - name: Set up Python
      uses: actions/setup-python@v6
      with:
        python-version: "3.11"
    - name: Install uv
      uses: astral-sh/setup-uv@v7
      with:
        enable-cache: true
    - name: Install dependencies
      run: |
        uv pip install --system -e .[test,dev,docs]
    - name: Build documentation
      run: |
        mkdocs build
    - name: Upload documentation artifact
      uses: actions/upload-pages-artifact@v5
      with:
        path: './site'

  deploy:
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    runs-on: ubuntu-latest
    needs: build
    steps:
    - name: Deploy to GitHub Pages
      id: deployment
      uses: actions/deploy-pages@v5
```

## Workflow Explanation

- **Trigger**: The workflow runs on pushes and pull requests to the `main` branch.
- **Permissions**: It requires write permissions for `pages` and `id-token` to deploy to GitHub Pages.
- **Concurrency**: A shared `pages` concurrency group serialises deployments so two runs never publish at once.
- **Build Job**:
  - Checks out the repository and sets up Python 3.11.
  - Installs [uv](https://docs.astral.sh/uv/) with caching enabled.
  - Installs the package with the `test`, `dev`, and `docs` extras — the `docs`
    extra provides `mkdocs`, `mkdocstrings`, and `mkdocs-material`.
  - Builds the documentation using `mkdocs build` (output in `./site`).
  - Uploads the built site as the Pages artifact.
- **Deploy Job**:
  - Runs only on pushes to `main` (pull requests stop after the build).
  - Deploys the artifact to GitHub Pages using the official `actions/deploy-pages` action.

## Enabling GitHub Pages

The workflow assumes the repository's Pages **source** is set to *GitHub Actions*:

1. Go to your repository on GitHub.
2. Click on the **Settings** tab.
3. In the left sidebar, click on **Pages**.
4. Under "Build and deployment", select **GitHub Actions** as the source.
5. Click **Save**.

Your documentation will be automatically built and deployed to GitHub Pages every time you push to the `main` branch.

## Testing a Documentation Build Locally

Before pushing, build the site locally to catch broken nav entries or mkdocstrings errors:

```bash
pip install -e ".[docs]"
mkdocs serve      # live-reload preview at http://127.0.0.1:8000
mkdocs build      # the same non-strict build CI performs
```
