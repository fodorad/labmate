// paper2carousel one-page summary, laid out like a project page of adamfodor.com:
// title, authors, main image, abstract, then Task / Challenges / Proposed method /
// Main results cards. Data arrives as JSON via sys.inputs (plain text, no markup).
#let data = json(bytes(sys.inputs.summary))
#let theme = data.theme
#let muted = rgb(theme.muted)
#let accent = rgb(theme.accent)
#let gap = 16pt

#set page(width: 21cm, height: auto, margin: (x: 1.7cm, top: 1.6cm, bottom: 1.4cm),
  fill: rgb(theme.surface))
#set text(font: theme.font, size: 10pt, fill: rgb(theme.text), fallback: true)
#set par(leading: 0.58em)

#align(center, {
  text(size: 22pt, weight: "bold", data.title)
  if data.authors != "" {
    v(6pt)
    text(size: 11pt, fill: muted, data.authors)
  }
  if data.url != "" {
    v(4pt)
    text(size: 9pt, fill: accent, data.url)
  }
})

#if data.at("image", default: none) != none {
  v(14pt)
  align(center, block(radius: 8pt, clip: true, stroke: 0.6pt + rgb(theme.border),
    image(data.image, width: 100%, height: 6.4cm, fit: "contain")))
  let cap = data.at("image_caption", default: "")
  if cap != "" {
    v(4pt)
    align(center, text(size: 8.5pt, fill: muted, cap))
  }
}

#if data.abstract != "" {
  v(16pt)
  align(center, block(width: 86%, {
    text(size: 14pt, weight: "bold", "Abstract")
    v(6pt)
    set par(justify: true)
    align(left, text(size: 9.5pt, data.abstract))
  }))
}

#v(28pt)

// A card: pastel background, rounded, white pill title overlapping the top edge.
#let card-body(c) = {
  if c.title != "" {
    align(center, text(size: 10.5pt, weight: "bold", c.title))
    v(4pt)
  }
  for b in c.bullets {
    grid(columns: (9pt, 1fr), column-gutter: 4pt,
      move(dy: 3.5pt, circle(radius: 2pt, fill: rgb(theme.text), stroke: none)),
      text(size: 9.5pt, b))
    v(3pt)
  }
}
#let card(c, width, height) = block(width: width, height: height, fill: rgb(c.color),
  radius: 12pt, inset: (x: 14pt, top: 20pt, bottom: 12pt), {
    place(top + center, dy: -31pt, box(fill: white, stroke: 0.8pt + rgb("#d9d9d9"),
      radius: 12pt, inset: (x: 14pt, y: 5pt),
      text(size: 11pt, weight: "bold", c.label)))
    card-body(c)
  })

// Two cards per row, equal heights within a row.
#layout(size => {
  let w = (size.width - gap) / 2
  let cards = data.cards
  for i in range(0, cards.len(), step: 2) {
    let row = cards.slice(i, calc.min(i + 2, cards.len()))
    let h = calc.max(..row.map(c => measure(block(width: w,
      inset: (x: 14pt, top: 20pt, bottom: 12pt), card-body(c))).height))
    grid(columns: (w, w), column-gutter: gap, ..row.map(c => card(c, w, h)))
    v(26pt)
  }
})

#align(center, text(size: 8pt, fill: muted,
  "Made with paper2carousel: every bullet is traced to a quote in the paper."))
