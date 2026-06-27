# Contributing to Leopard 44 KB — Shared Layer Documentation

This file describes how to contribute model-generic Leopard 44 and Sunsail 444 knowledge
to the shared layer of Leopard 44 KB. Content merged here ships with the package and is
updated on every user's machine via `git pull`, so the quality bar is deliberate: useful
to any L44 owner, free of personal or vessel-identifying information.


## What qualifies as shared-layer content

Contribute content that any Leopard 44 or Sunsail 444 owner would find useful, regardless
of which vessel they own:

- Factory specifications, service manuals, and technical documentation
- Model-wide known issues, workarounds, and service bulletins
- Yanmar engine documentation (4JH45, 4JH57, 3JH5 series)
- Common system references: electrical, plumbing, rigging, sails, refrigeration
- Owner-distilled upgrade reports that apply to the model class, not one specific vessel
- Community knowledge sourced from the Leopard 44 owners group (anonymised — see below)

**Never contribute:**

- Personal maintenance logs, repair records, or costs
- WhatsApp chat exports in raw form
- Photos of a specific boat
- Equipment serial numbers, purchase history, or receipts
- Anything that identifies a specific vessel, owner, or location

**The test:** Would another L44 owner find this useful regardless of which boat they own?
If yes, it belongs in `shared/`. If it is only useful to one vessel, it belongs in `data/`
(gitignored, never committed).


## Anonymisation rules

When contributing content sourced from the Leopard 44 owners group, a forum thread, or
other community sources:

- Strip all member names, boat names, and contact details before committing.
- Credit the source as "Leopard 44 owners group" or the forum thread URL — never an
  individual's name or handle.
- Remove or generalise any references to a specific marina, berth, or location unless
  the information is model-generic (e.g. a watermaker specification is fine; "we were
  anchored in Fiji when this happened" is not).
- Keep the shared layer PII-clean so the public repository does not expose personal
  information that contributors did not intend to make permanently public.

If you are unsure whether a piece of information is personal, remove it. A future
contributor can always add context; removing PII after a public push is much harder.


## Recommended header block

Adding a short header block at the top of your contributed document helps reviewers
assess provenance and scope. The header is **recommended, not required** — the existing
seeded files in `shared/` carry none, and that is intentional. Do not feel blocked by
the absence of a header.

When you do include one, a simple Markdown block comment works well:

```markdown
<!--
title: <Short descriptive title>
topic: <leopard44 | yanmar | systems | upgrades>
source: <Manual page / forum thread URL / "Leopard 44 owners group">
source-date: <approximate date of the original information, if known>
license: MIT
-->
```

Use whichever fields are meaningful. Omit the block entirely if you prefer.


## Fork, branch, and pull request flow

Leopard 44 KB is hosted at [primesoftnz/leopard44-kb](https://github.com/primesoftnz/leopard44-kb).

1. **Fork** the repository on GitHub: click "Fork" on the `primesoftnz/leopard44-kb` page.
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/<your-username>/leopard44-kb.git
   cd leopard44-kb
   ```
3. **Create a topic branch** from `main`:
   ```bash
   git checkout -b contrib/yanmar-4jh45-service-intervals
   ```
   Name the branch after the content you are adding.
4. **Place the document** under the right `shared/` subdirectory:
   - `shared/leopard44/` — model-wide reference: specs, known issues, performance
   - `shared/yanmar/` — engine documentation: 4JH45 / 4JH57 service manuals and intervals
   - `shared/systems/` — electrical, plumbing, rigging, sails, refrigeration, navigation
   - `shared/upgrades/` — model-generic upgrades distilled from owner reports
5. **Commit** your document:
   ```bash
   git add shared/yanmar/4jh45-service-intervals.md
   git commit -m "docs: add Yanmar 4JH45 service interval reference"
   ```
6. **Push** to your fork:
   ```bash
   git push origin contrib/yanmar-4jh45-service-intervals
   ```
7. **Open a pull request** against `primesoftnz/leopard44-kb` on GitHub. In the PR description,
   note the source of the information (manual page, forum thread URL, or personal
   observation confirmed across multiple owners).

A maintainer will review the content for scope (model-generic only), anonymisation, and
formatting before merging.


## The data/ guard

The `data/` directory is gitignored vessel-layer storage. A pre-commit hook defined in
`.pre-commit-config.yaml` (hook id: `no-data-commits`) blocks any commit that attempts to
add files under `data/`. If you accidentally stage a file there, the hook exits non-zero
with the message:

```
ERROR: data/ is gitignored vessel-layer storage. Move file or stash before commit.
```

This is a mechanical safeguard against vessel-specific or personal content leaking into
the public repository. It applies to all contributors, including maintainers. The hook
runs automatically once you have installed the pre-commit framework:

```bash
uv run pre-commit install
```


## License and consent

All content in `shared/` is MIT-licensed and distributed with the Leopard 44 KB package.
By opening a pull request against `primesoftnz/leopard44-kb`, you confirm that you have
the right to share the content under those terms and you agree that your contribution may
be distributed under the MIT licence alongside the rest of the shared layer.

Community content contributed from the Leopard 44 owners group is credited to that group,
not to individuals, and is shared with the understanding that it benefits all L44 owners.
