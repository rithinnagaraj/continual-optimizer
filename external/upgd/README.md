# UPGD dependency subset

This directory bundles the optimizer and the two published experiment files
needed by the CAP experiment provenance checks. The upstream repository and
revision are recorded in the root THIRD_PARTY_NOTICES.md. Required copyright
and MIT terms are preserved in LICENSE.

Generic standalone README defaults (not the task-specific benchmark settings):
`lr=1e-5, weight_decay=0.001, beta_utility=0.999, sigma=0.001`.

The task-specific settings are taken from the pinned upstream experiment files.
Only the dependency files and protocol references are bundled; standalone
upstream experiments require the full upstream repository.
