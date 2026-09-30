// paper2flow overview (A4 portrait): the paper, its four blocks, its flow diagrams.
// Data arrives as JSON via sys.inputs; strings are inserted as plain text, so nothing in
// the paper can inject Typst markup.
#let data = json(bytes(sys.inputs.data))
#let theme = data.theme
#let muted = rgb(theme.muted)
#let accent = rgb(theme.accent)
#let border = rgb(theme.border)

// No creation date: the same run renders the same bytes.
#set document(title: data.title, date: none)
#set page(paper: "a4", margin: (x: 1.8cm, top: 1.7cm, bottom: 1.9cm),
  fill: rgb(theme.background),
  footer: context {
    set text(size: 8pt, fill: muted)
    line(length: 100%, stroke: 0.5pt + border)
    v(-4pt)
    [paper2flow · every statement is traced to a quote in the paper]
    h(1fr)
    counter(page).display("1 / 1", both: true)
  })
#set text(font: theme.font, size: 10.5pt, fill: rgb(theme.text), fallback: true)
#set par(leading: 0.6em)

// A diagram at most at its natural size: small ones are not blown up to fill the page.
#let diagram(d) = layout(size => {
  let (w, h) = (d.width * 1pt, d.height * 1pt)
  image(d.image, width: w * calc.min(1.0, size.width / w, size.height / h))
})

#let kicker(label) = {
  rect(width: 32pt, height: 3.5pt, fill: accent, stroke: none)
  v(6pt)
  text(size: 9pt, fill: accent, weight: "bold", tracking: 1.1pt, upper(label))
}

#let legend-row(items) = align(center, {
  for item in items {
    box(inset: (x: 6pt), {
      box(width: 10pt, height: 10pt, radius: 2pt, fill: rgb(item.fill),
        stroke: 1pt + rgb(item.border), baseline: 1.5pt)
      h(4pt)
      text(size: 9pt, fill: muted, item.name)
    })
  }
})

// --- page 1: the paper ------------------------------------------------------------------
#block(height: 100%, {
  kicker("Paper overview")
  v(12pt)
  text(size: 24pt, weight: "bold", data.title)
  if data.authors != "" {
    v(8pt)
    text(size: 12pt, fill: muted, data.authors)
  }
  if data.publication != "" {
    v(6pt)
    text(size: 11pt, weight: "semibold", data.publication)
  }
  if data.url != "" {
    v(4pt)
    text(size: 9pt, fill: accent, data.url)
  }
  v(18pt)
  if data.at("figure", default: none) != none {
    block(height: 1fr, width: 100%, radius: 8pt, fill: rgb(theme.surface),
      stroke: 0.6pt + border, inset: 12pt,
      align(center + horizon, image(data.figure, width: 100%, height: 100%, fit: "contain")))
    let cap = data.at("figure_caption", default: "")
    if cap != "" {
      v(6pt)
      align(center, text(size: 9pt, fill: muted, cap))
    }
  } else if data.abstract != "" {
    text(size: 13pt, weight: "bold", "Abstract")
    v(6pt)
    set par(justify: true)
    text(size: 10.5pt, data.abstract)
  }
})

// --- page 2: the four cards, two by two, like a project page ---------------------------
#let card(c) = block(width: 100%, height: 100%, fill: rgb(c.color), radius: 12pt,
  inset: (x: 12pt, top: 20pt, bottom: 10pt), {
    place(top + left, dy: -30pt, box(fill: white, stroke: 0.8pt + rgb("#d9d9d9"),
      radius: 12pt, inset: (x: 12pt, y: 5pt), text(size: 10pt, weight: "bold", c.label)))
    text(size: 11pt, weight: "bold", c.title)
    v(4pt)
    for b in c.bullets {
      grid(columns: (8pt, 1fr), column-gutter: 3pt,
        move(dy: 3.5pt, circle(radius: 2pt, fill: rgb(theme.text), stroke: none)),
        text(size: 9.5pt, b))
      v(1pt)
    }
  })

#pagebreak()
#kicker("At a glance")
#v(24pt)
#grid(columns: (1fr, 1fr), rows: (1fr, 1fr), column-gutter: 12pt, row-gutter: 32pt,
  ..data.cards.map(card))

// --- pages 3+: the flow diagrams --------------------------------------------------------
#for d in data.diagrams {
  pagebreak()
  block(height: 100%, {
    kicker(d.kicker)
    v(10pt)
    text(size: 18pt, weight: "bold", d.title)
    v(6pt)
    text(size: 10.5pt, fill: muted, d.caption)
    v(12pt)
    block(height: 1fr, width: 100%, radius: 8pt, fill: rgb(theme.surface),
      stroke: 0.6pt + border, inset: 12pt,
      align(center + horizon, diagram(d)))
    v(8pt)
    legend-row(d.legend)
  })
}
