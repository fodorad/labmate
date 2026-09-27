// paper2flow LinkedIn post (4:5 pages): the post text to copy, then the images to attach.
// Data arrives as JSON via sys.inputs; strings are inserted as plain text.
#let data = json(bytes(sys.inputs.data))
#let theme = data.theme
#let accent = rgb(theme.accent)
#let muted = rgb(theme.muted)
#let border = rgb(theme.border)

// No creation date: the same run renders the same bytes.
#set document(title: data.title, date: none)
#set page(width: 540pt, height: 675pt, margin: (x: 36pt, top: 36pt, bottom: 44pt),
  fill: rgb(theme.background),
  footer: context {
    set text(size: 8.5pt, fill: muted)
    line(length: 100%, stroke: 0.5pt + border)
    v(-4pt)
    text(weight: "semibold", theme.handle)
    h(1fr)
    counter(page).display("1 / 1", both: true)
  })
#set text(font: theme.font, size: 11pt, fill: rgb(theme.text), fallback: true)
#set par(leading: 0.62em)

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

// --- page 1: the post text --------------------------------------------------------------
#kicker("LinkedIn post · text")
#v(10pt)
#block(width: 100%, radius: 10pt, fill: rgb(theme.surface), stroke: 0.6pt + border,
  inset: 16pt, {
    text(size: 13pt, weight: "bold", data.hook)
    v(8pt)
    for t in data.takeaways {
      grid(columns: (12pt, 1fr), column-gutter: 4pt,
        move(dy: 3pt, rect(width: 7pt, height: 7pt, radius: 1.5pt, fill: accent, stroke: none)),
        t)
      v(4pt)
    }
    v(4pt)
    data.question
    v(8pt)
    text(size: 10pt, fill: muted, "Paper: " + data.title)
    if data.url != "" {
      linebreak()
      text(size: 10pt, fill: accent, data.url)
    }
  })
#v(1fr)
#text(size: 9pt, fill: muted)[
  Every sentence above passed the same fact-check as the overview. The next pages are
  the images for the post: attach one, or upload pages 2–#(data.diagrams.len() + 1) as a
  document to post them as a carousel.
]

// --- pages 2+: one image per diagram ----------------------------------------------------
#for (i, d) in data.diagrams.enumerate() {
  pagebreak()
  block(height: 100%, {
    kicker(d.kicker)
    v(8pt)
    if i == 0 {
      text(size: 20pt, weight: "bold", data.hook)
      v(6pt)
      text(size: 10pt, fill: muted, data.title)
      if data.publication != "" {
        text(size: 10pt, fill: muted, " · " + data.publication)
      }
    } else {
      text(size: 18pt, weight: "bold", d.title)
    }
    v(10pt)
    block(height: 1fr, width: 100%, radius: 10pt, fill: rgb(theme.surface),
      stroke: 0.6pt + border, inset: 10pt,
      align(center + horizon, diagram(d)))
    v(6pt)
    align(center, text(size: 9.5pt, fill: muted, d.caption))
    v(6pt)
    align(center, {
      for item in d.legend {
        box(inset: (x: 5pt), {
          box(width: 9pt, height: 9pt, radius: 2pt, fill: rgb(item.fill),
            stroke: 1pt + rgb(item.border), baseline: 1.5pt)
          h(3pt)
          text(size: 8.5pt, fill: muted, item.name)
        })
      }
    })
  })
}
