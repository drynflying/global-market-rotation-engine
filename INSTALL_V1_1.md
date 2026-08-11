# Install Version 1.1

Using GitHub Desktop:

1. Choose **Repository -> Show in Finder**.
2. Unzip this patch.
3. Copy the contents of the patch into the repository root.
4. When macOS asks, choose **Replace** / **Merge** for the existing files/folders.
5. In GitHub Desktop, confirm the changed files.
6. Commit to `main` with:
   `Upgrade rotation engine to v1.1`
7. Click **Push origin**.
8. On GitHub, run:
   **Actions -> Manual Market Rotation Test -> Run workflow**
9. Do not add Gemini yet.

Expected after the rerun:
- `scored_tickers` should be 111 if all four pair signals have sufficient history.
- `cross_sectional_scored` should be 107.
- `pair_scored` should be 4.
- `score_formula_version` in generated results should be `v1.1`.
