# Mature DeEcho Routing Plan

## Goal
Use the same strategy family as mature RVC cover projects:
- prefer learned DeEcho / DeReverb models
- otherwise pass separated lead vocals directly into RVC
- avoid stacking hand-crafted dereverb, tail gating, and aggressive source reconstruction by default

## Steps
1. Add configurable VC preprocess strategy: `auto`, `direct`, `uvr_deecho`, `legacy`
2. Add configurable source-constraint strategy: `auto`, `off`, `on`
3. Default to `auto` + `auto` in config for mature-project behavior
4. Expose both options in the Gradio cover UI
5. Route pipeline behavior from UI/config into `CoverPipeline.process`
6. Keep legacy chain available for comparison, but not as the default path
7. Validate syntax and confirm logs show the new routing branch

## Expected Result
- Cleaner conversion on echo-tail sections
- Less false vocal/noise generation in gaps and reverberant endings
- Behavior closer to AICoverGen / official UVR+RVC workflows
