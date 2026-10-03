// cv2job: the cover letter (A4). Data arrives as JSON via sys.inputs.
#let data = json(bytes(sys.inputs.data))
#let theme = data.theme
#set document(title: data.name + " - cover letter", date: none)
#set page(paper: "a4", margin: (x: 2.4cm, y: 2.4cm), fill: rgb(theme.background))
#set text(font: theme.font, size: 11pt, fill: rgb(theme.text), fallback: true)
#set par(leading: 0.7em, spacing: 1.1em)

Dear Hiring Manager,

#data.opening

#if data.intro != "" [
  #data.intro
  #for h in data.highlights [
    - *#h.skill*: #h.reason
  ]
]

#data.closing

Sincerely, \
#data.name
