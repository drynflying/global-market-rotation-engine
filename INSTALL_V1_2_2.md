# v1.2.2 workflow environment fix

This patch fixes the GitHub Actions environment wiring for the provider-agnostic AI layer.

Your repository variable `AI_PROVIDERS=gemini` is already correct. The prior workflow
only passed the legacy singular variable `AI_PROVIDER`, so the Python process saw
no enabled providers.

Apply this patch to the repository root, merge `.github`, commit, push, and rerun
the manual workflow.

Commit message:
`Fix AI provider workflow environment`
