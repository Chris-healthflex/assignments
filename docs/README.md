# Demo material

Supporting media for the README. Nothing here is imported by the application —
it exists so a reviewer can see the system working without installing it.

## What goes where

```
docs/
├── screenshots/          PNG, referenced inline from the main README
│   ├── 01-intake.png             upload screen, file chosen
│   ├── 02-processing.png         pipeline mid-run, stages advancing
│   ├── 03-review.png             the finished record
│   ├── 04-confidence.png         "How is this calculated?" expanded
│   ├── 05-flagged-fields.png     the review rail, flags visible
│   ├── 06-timings.png            "How long did this take?" expanded
│   └── 07-swagger.png            /docs, for the API side
├── demo.mp4              screen recording of a full run
└── sample-report.pdf     a PDF exported from the app
```

Keep the filenames exactly as above — the main README links to them by name.

## Practical notes

**Screenshots.** PNG at roughly 1440 px wide renders well on GitHub without
being enormous. These are the only media GitHub displays *inline* in a README,
so they carry most of the value: a reviewer sees the interface before deciding
whether to run anything.

**Video.** MP4 (H.264) is the safest container. Two things worth knowing:

- GitHub **warns above 50 MB** per file and **rejects above 100 MB**. A
  screen recording of a full two-and-a-half minute run can exceed that easily
  — trim to the interesting parts, or drop the frame rate, before committing.
- A repo-relative `.mp4` link in a README renders as a **download link, not a
  player**. To get inline playback, drag the file into a GitHub issue or the
  web editor: GitHub uploads it to its own CDN and rewrites the link. Commit
  the file here as well so the repository is self-contained.

**PDF.** Export it from the running app via the *Download PDF* button rather
than printing the page by hand, so it carries the real print stylesheet —
including the appendix listing every flagged field and the signature block.
