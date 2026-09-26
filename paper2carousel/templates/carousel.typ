// paper2carousel slide template. Data arrives as JSON via sys.inputs; strings are
// inserted as plain text, so nothing in the paper can inject Typst markup.
#let data = json(bytes(sys.inputs.deck))
#let theme = data.theme

#set page(width: 540pt, height: 675pt, margin: 42pt, fill: rgb(theme.background))
#set text(font: theme.font, size: 17pt, fill: rgb(theme.text))
#set par(leading: 0.6em)

#let footer(i, n) = place(bottom + left, dy: 18pt, text(size: 10pt, fill: rgb(theme.muted))[
  #data.source #h(1fr) #str(i) / #str(n)
])

// Cover
#block(height: 100%)[
  #v(1fr)
  #text(size: 12pt, fill: rgb(theme.accent), weight: "bold", upper("paper explained"))
  #v(10pt)
  #text(size: 34pt, weight: "bold", data.title)
  #v(18pt)
  #text(size: 13pt, fill: rgb(theme.muted), data.paper_title)
  #v(1fr)
]
#footer(1, data.slides.len() + 1)

#for (i, slide) in data.slides.enumerate() {
  pagebreak()
  block(height: 100%)[
    #text(size: 12pt, fill: rgb(theme.accent), weight: "bold", str(i + 1))
    #v(6pt)
    #text(size: 26pt, weight: "bold", slide.title)
    #v(22pt)
    #for b in slide.bullets {
      grid(columns: (14pt, 1fr), column-gutter: 6pt,
        text(fill: rgb(theme.accent), "•"), text(b))
      v(12pt)
    }
  ]
  footer(i + 2, data.slides.len() + 1)
}
