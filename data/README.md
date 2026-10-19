# ReLIS foil evaluation annotations and source references

Human annotations and source-video references for the adjudicated foil
evaluation in *Stable-Regime Segmentation of Competitive Dyadic Fencing
Interactions* (WACV 2027).

**No video is redistributed.** The source material is a public YouTube
playlist of broadcast recordings from the 2025 Shanghai Foil Grand Prix. Users
who wish to reproduce the evaluation must obtain the videos from YouTube
themselves, subject to YouTube's Terms of Service and the rights of the
original broadcaster. This folder contains only video identifiers, temporal
positions, and the authors' own labels. It contains no athlete names, video
titles, pose tracks, or other derived visual data.

## Files

| File | Contents |
| --- | --- |
| `sources.csv` | `source_index`, `youtube_id`, `url` for each source broadcast used |
| `clips.csv` | One row per evaluated exchange (234 records; 223 usable) |
| `segments_consensus.csv` | Adjudicated consensus stable/active partition of each exchange |
| `segments_individual.csv` | Independent labels of the two annotators (`A`, `B`) before adjudication |

### `clips.csv`

- `clip_id`: `PREFIX-SOURCE-EXCHANGE`; the middle field equals `source_index`.
- `cohort`: `reference` or `confirmation` (annotation-blind) cohort.
- `usable`, `clean`: inclusion flags used in the paper.
- `source_index`, `youtube_id`, `exchange_index`: source broadcast and exchange number within it.
- `fps`: 25. Clip frame `f` corresponds to source time `source_start_sec + f / fps`.
- `clip_num_frames`, `source_start_frame`, `source_start_sec`, `source_end_sec`:
  position of the clip within the 25-fps source video, recovered by frame matching.
- `eval_start_frame`, `eval_end_frame`: evaluated window in clip frames
  (inclusive), ending at the scoring light (`endpoint`).
- `outcome_label`: scoring-light side (`Left`, `Right`, `None`).
- `valid`, `pose_issue`, `ambiguous`, `not_exchange`: annotation quality flags.

### `segments_*.csv`

One row per labeled interval: `clip_id`, (`annotator`), `segment`,
`start_frame`, `end_frame` (clip frames, inclusive), `start_sec`, `end_sec`,
`activity` (`stable` or `active`), `uncertain`.

Timing note: downloaded YouTube streams can differ by a frame or so from the
copies used here; allow a small tolerance when aligning.

## License

The annotations and metadata in this folder are released under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The source videos
are not covered by this license.

## Removal requests

If you appear in or own a source video and want it removed from this list,
open an issue or email ziwenc3@illinois.edu.
