// cv2job: the gap report, for the candidate only (A4). Data arrives as JSON via sys.inputs.
#let data = json(bytes(sys.inputs.data))
#let theme = data.theme
#let muted = rgb(theme.muted)
#let accent = rgb(theme.accent)
#set document(title: "Gap report - " + data.position, date: none)
#set page(paper: "a4", margin: (x: 2cm, y: 1.8cm), fill: rgb(theme.background))
#set text(font: theme.font, size: 10pt, fill: rgb(theme.text), fallback: true)
#set par(leading: 0.55em)

#let section(label) = {
  v(12pt)
  text(size: 9pt, weight: "bold", fill: accent, tracking: 1pt, upper(label))
  v(-4pt)
  line(length: 100%, stroke: 0.5pt + rgb(theme.border))
  v(4pt)
}
#let tag(must) = if must { text(fill: accent, weight: "bold", "must") } else { text(fill: muted, "nice") }

#text(size: 20pt, weight: "bold", "Gap report")
#v(2pt)
#text(fill: muted, data.position + (if data.company != "" { ", " + data.company } else { "" }))
#v(2pt)
#text(size: 9pt, fill: muted, "For you only: what the CV shows for this posting, and what it does not.")

#section("Not covered: do not claim these")
#if data.gaps.len() == 0 [ Every requirement is covered. ]
#for g in data.gaps {
  block(breakable: false, [#tag(g.must) #h(4pt) *#g.requirement* \ #text(size: 9pt, fill: muted, "\"" + g.quote + "\"")])
  v(3pt)
}

#section("Covered")
#for c in data.covered {
  block(breakable: false, [
    #tag(c.must) #h(4pt) *#c.requirement* #text(fill: muted, "(" + c.source + ")")
    #for e in c.evidence [ \ #text(size: 9pt, fill: muted, "• " + e) ]
  ])
  v(3pt)
}

#if data.told.len() > 0 {
  section("What you told me (not in your CV yet)")
  for t in data.told [ - #t ]
}
