# Third-party code

`external/upgd/` contains the unmodified files needed for the UPGD optimizer and
its protocol provenance. Source: https://github.com/mohmdelsayed/upgd, revision
`b75e90ad4b09c28971ac9dbb902a8fd86709b28c`.

Its MIT license and required copyright notice are retained in
`external/upgd/LICENSE`. The three Python files
are retained byte for byte because the experiment checks their pinned hashes.
The upstream experiment files are reference material, not executable entry
points in this dependency subset. Use this repository's launcher for experiments.

Other dependencies are installed from `requirements.txt` under their respective
licenses. Datasets and pretrained backbone weights are obtained separately and
are not redistributed here. First-party source has no assigned license.
The upstream MIT notice applies only to upstream code.
