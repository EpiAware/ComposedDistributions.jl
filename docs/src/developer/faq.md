# [Developer FAQ](@id developer-faq)

Short answers to environment questions that come up when developing this package.

## Why is there no `test/Manifest.toml`?

This package uses a Julia 1.12 workspace: `Project.toml` declares `[workspace] projects = ["test", "docs"]`, so the `test` and `docs` sub-projects resolve against the shared root `Manifest.toml` rather than a manifest of their own.
Pkg never writes or reads a per-member manifest under a workspace, so a `test/Manifest.toml` on disk is a stale leftover from before the workspace migration (or from a manual `Pkg.instantiate()` run inside `test/`) and can be deleted.
It is already covered by the repository's `.gitignore`, so it never gets committed either way.

## I pulled `main` after a cross-repo dependency change and something looks stale

Check the `[sources]` block in `Project.toml`.
Entries pinned to `rev = "main"` (currently `ConvolvedDistributions` and `EpiAwareADTools`, kept there only until they register — see the comment above `[sources]`) are resolved once and then frozen in the local `Manifest.toml`.
A plain `git pull` updates this repository's own source but does not touch that frozen resolution, so an upstream `rev = "main"` package can move on and your local environment will keep using the old git-tree-sha with no warning or error.

Run `Pkg.update()` (not just `Pkg.instantiate()`) after any change that touches a `rev = "main"` dependency upstream, and whenever a bug looks like it should have been fixed by someone else's merge.
Skipping this step produced a real phantom-bug investigation once (#191); the underlying pin hazard is tracked at #193 and only goes away once the pinned packages register.
