// paper2carousel LinkedIn post image (4:5): hook, paper, the proposed method as a graph.
// Data arrives as JSON via sys.inputs; strings are inserted as plain text.
#let data = json(bytes(sys.inputs.post))
#let theme = data.theme
#let accent = rgb(theme.accent)
#let muted = rgb(theme.muted)

#set page(width: 540pt, height: 675pt, margin: (x: 36pt, top: 38pt, bottom: 30pt),
  fill: rgb(theme.background))
#set text(font: theme.font, size: 12pt, fill: rgb(theme.text), fallback: true)

#block(height: 100%, {
  rect(width: 36pt, height: 4pt, fill: accent, stroke: none)
  v(10pt)
  text(size: 10pt, fill: accent, weight: "bold", tracking: 1.2pt, upper(data.at("kicker", default: "Proposed method")))
  v(8pt)
  text(size: 22pt, weight: "bold", data.hook)
  v(8pt)
  text(size: 10.5pt, fill: muted, data.title)
  if data.authors != "" {
    v(2pt)
    text(size: 9pt, fill: muted, data.authors)
  }
  v(12pt)
  block(height: 1fr, width: 100%, radius: 10pt, fill: rgb(theme.surface),
    stroke: 0.6pt + rgb(theme.border), inset: 10pt,
    align(center + horizon, image(data.graph, width: 100%, height: 100%, fit: "contain")))
  v(8pt)
  align(center, text(size: 10pt, fill: muted, data.caption))
  v(8pt)
  align(center, {
    for item in data.legend {
      box(inset: (x: 5pt), {
        box(width: 10pt, height: 10pt, radius: 2pt, fill: rgb(item.fill),
          stroke: 1pt + rgb(item.border), baseline: 1.5pt)
        h(4pt)
        text(size: 9pt, fill: muted, item.name)
      })
    }
  })
  v(10pt)
  line(length: 100%, stroke: 0.6pt + rgb(theme.border))
  v(6pt)
  text(size: 9pt, fill: muted, weight: "semibold", theme.handle)
  h(1fr)
  text(size: 9pt, fill: muted, "Paper link in the post")
})
