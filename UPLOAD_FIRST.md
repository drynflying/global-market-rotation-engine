# What to upload now

You already have `config/rotation_universe.csv` in your private repository.

Unzip this package on your computer. Then upload the following into the repository's
**main** branch, preserving the folders:

- `.github/`
- `src/`
- `data/`
- `results/`
- `docs/`
- `requirements.txt`
- `.gitignore`
- `README.md`

The package also contains `config/rotation_universe.csv`. If GitHub asks whether you want
to replace your existing configuration file, replacing it with the packaged copy is fine
because it is the same file used to build this starter package.

After upload, the top level of the repository should look like:

    .github/
    config/
    data/
    docs/
    results/
    src/
    .gitignore
    README.md
    requirements.txt

Then open the repository's **Actions** tab. You should see:

    Manual Market Rotation Test

Do not add a Gemini key yet. Run the deterministic pipeline first.
