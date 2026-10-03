"""cv2job: tailor a CV and a cover letter to a job posting, and report what the CV lacks.

A chain with one agent step: the requirements of the posting are read (quotes checked against
the text), matched to CV bullets, and the requirements nothing matches are handed to an agent
that searches the CV and asks the candidate. The tailored CV, the cover letter and a separate
gap report are rendered as PDFs. Nothing enters them that is not in the CV or said by the
candidate.
"""
