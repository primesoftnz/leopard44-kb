# Leopard 44 KB — Shared Knowledge Layer

This directory holds model-generic Leopard 44 / Sunsail 444 knowledge that ships with
Leopard 44 KB and is updated via `git pull`. It is version-controlled and open for
community contribution via pull request.

## What belongs here

Content that would be useful to any Leopard 44 or Sunsail 444 owner:

- Factory specifications, service manuals, and technical documentation
- Model-wide known issues, workarounds, and service bulletins
- Yanmar engine documentation (4JH45, 4JH57, 3JH5 series)
- Common system references: electrical, plumbing, rigging, sails, refrigeration
- Owner-distilled upgrade reports that apply to the model class, not one specific vessel
- Community knowledge from the Leopard 44 owners group (anonymised with consent)

## What does NOT belong here

Anything specific to one owner's vessel goes in `data/` (gitignored, never committed):

- Personal maintenance logs, repair records, costs
- WhatsApp chat exports
- Photos of your specific boat
- Equipment serial numbers and purchase history
- Anything that identifies a specific vessel, owner, or location

**If in doubt:** Would another L44 owner find this useful? If yes, it belongs in `shared/`.
Only useful to your boat? It belongs in `data/`.

## Topic subdirectories

| Directory | Contents |
|-----------|----------|
| `leopard44/` | Model-wide reference: specs, known issues, buyer's guide, sailing performance |
| `yanmar/` | Engine documentation: 4JH45 / 4JH57 service manuals, service intervals |
| `systems/` | Electrical, plumbing, rigging, sails, refrigeration, navigation |
| `upgrades/` | Model-generic upgrades distilled from owner reports |

## Attribution

When contributing, note the source of the information (manual page, forum thread URL,
personal observation confirmed by multiple owners). Attribution rules and the full
contribution process are documented in `CONTRIBUTING.md`.

All content in `shared/` is MIT-licensed and intended for distribution with the Leopard 44
KB package. By submitting a pull request, you agree that your contribution may be
distributed under the same terms.
