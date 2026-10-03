You read a job posting and list what it asks of the candidate.

Job posting:
{job}

- "position": the job title as written.
- "company": the employer's name as written; "" if the posting does not name it.
- "requirements": the 4 to 12 things the candidate must bring: skills, tools, experience,
  education, languages. For each:
  - "text": the requirement in a few words (for example "PyTorch experience").
  - "quote": the sentence or phrase of the posting it comes from, copied character for
    character (at least 3 words).
  - "must": true for a must-have ("required", "you have"), false for a nice-to-have
    ("plus", "preferred").
Do not list perks, salary, location or what the company does.
