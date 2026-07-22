# Reference Annotations

The final file `chatgpt_5_6_sol_reference.jsonl` contains one GPT-5.6 Sol silver-reference annotation for each of the 300 benchmark articles.

The reference is deliberately called a silver label: it measures FLAN-T5 agreement with a stronger annotator under a frozen taxonomy, not objective ground-truth accuracy. Each row records the protocol version and target metadata. News text is excluded from this directory so the public repository does not redistribute article summaries; labels are joined to the private benchmark by `article_id`.

Temporary batch files and model predictions belong under the Git-ignored `outputs/` directory.
