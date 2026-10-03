// cv2job: the tailored CV (A4, one column). Data arrives as JSON via sys.inputs; strings are
// inserted as plain text, so nothing in the CV can inject Typst markup.
#let data = json(bytes(sys.inputs.data))
#let theme = data.theme
#let muted = rgb(theme.muted)
#let accent = rgb(theme.accent)
#set document(title: data.name + " - CV", date: none)
#set page(paper: "a4", margin: (x: 2cm, y: 1.8cm), fill: rgb(theme.background))
#set text(font: theme.font, size: 10pt, fill: rgb(theme.text), fallback: true)
#set par(leading: 0.55em)

#let heading-line(label) = {
  v(10pt)
  text(size: 9pt, weight: "bold", fill: accent, tracking: 1pt, upper(label))
  v(-4pt)
  line(length: 100%, stroke: 0.5pt + rgb(theme.border))
}

#text(size: 22pt, weight: "bold", data.name)
#if data.headline != "" [
  #v(2pt)
  #text(size: 11pt, fill: muted, data.headline)
]
#if data.contact != "" [
  #v(2pt)
  #text(size: 9pt, fill: muted, data.contact)
]

#if data.skills.len() > 0 {
  heading-line("Skills")
  v(2pt)
  data.skills.join(" · ")
}

#heading-line("Experience")
#for role in data.roles {
  v(5pt)
  grid(columns: (1fr, auto),
    text(weight: "bold", role.title + ", " + role.company),
    text(fill: muted, role.period))
  for b in role.bullets {
    grid(columns: (8pt, 1fr), column-gutter: 3pt,
      move(dy: 3pt, circle(radius: 1.8pt, fill: rgb(theme.text), stroke: none)), b)
    v(1pt)
  }
}

#if data.education.len() > 0 {
  heading-line("Education")
  for e in data.education {
    v(3pt)
    grid(columns: (1fr, auto),
      [#text(weight: "bold", e.degree), #e.school], text(fill: muted, e.year))
  }
}
