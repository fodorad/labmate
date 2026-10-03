Write a LinkedIn post that tells the story of a research paper. The reader is an ML engineer
scrolling their feed: after reading it they should know what the paper did, how, and what it
achieved, and want to look at the attached data-flow diagram.

Paper: {title}

- "hook": the first line, at most 15 words, specific to this paper and curiosity-raising, not
  clickbait. It says what the paper itself does, not what its applications are.
- "takeaways": 3 to 5 sentences (at most 30 words each) that read as one connected story, not a
  list. In this order: the problem, the idea, how it works, the key result with its numbers
  and what they are measured on, and why it matters or its limitation. Link them with
  plain connecting words ("So", "Instead", "The result:") and do not start two sentences the
  same way. Each sentence cites the claim ids it uses. Use only the facts below; keep numbers
  exactly.
- "question": one question about a specific thing in this paper (a component, a trade-off,
  a dataset or a number) that invites readers to share their own experience with it. It
  must not fit any other paper: no "How has X impacted your workflow?". No facts in it.
- Plain text, no emoji, no hashtags, no "In this paper".

Example of the style, for a made-up paper (do not reuse its words or facts):
  hook: Sorting 1M items without ever comparing two of them.
  takeaways:
  1. Comparison sorts hit an n log n wall, and GPUs sit idle while they wait on branches.
  2. So the authors map each key to a bucket with a learned function and skip comparing.
  3. Items land in buckets in one parallel pass, and a small fix-up pass orders each bucket.
  4. The result: 3.1x faster than the best GPU radix sort on 1M random integers.
  5. The catch: skewed keys make the buckets uneven and erase most of the gain.
  question: How skewed are the keys you sort in production?

The paper's overview (facts you may use, with their claim ids):
{cards}
