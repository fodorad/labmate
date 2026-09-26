// paper2carousel slide template. Data arrives as JSON via sys.inputs; strings are
// inserted as plain text, so nothing in the paper can inject Typst markup.
#let data = json(bytes(sys.inputs.deck))
#let theme = data.theme
#let accent = rgb(theme.accent)
#let muted = rgb(theme.muted)
#let n = data.slides.len() + 1

#set page(width: 540pt, height: 675pt, margin: (x: 44pt, top: 46pt, bottom: 58pt),
  fill: rgb(theme.background))
#set text(font: theme.font, size: 16pt, fill: rgb(theme.text), fallback: true)
#set par(leading: 0.62em)

#let footer(i) = place(bottom + left, dy: 34pt, block(width: 100%, {
  line(length: 100%, stroke: 0.6pt + rgb(theme.border))
  v(6pt)
  text(size: 9pt, fill: muted, weight: "semibold", theme.handle)
  h(1fr)
  text(size: 9pt, fill: muted, str(i) + " / " + str(n))
}))

// ── Cover ──────────────────────────────────────────────────────────────────────────
#block(height: 100%, {
  let cover = data.at("cover_image", default: none)
  if cover != none {
    block(width: 100%, height: 42%, radius: 10pt, clip: true,
      image(cover, width: 100%, height: 100%, fit: "cover"))
    v(1fr)
  } else {
    v(1fr)
  }
  rect(width: 36pt, height: 4pt, fill: accent, stroke: none)
  v(12pt)
  text(size: 11pt, fill: accent, weight: "bold", tracking: 1.2pt, upper("Paper explained"))
  v(10pt)
  text(size: if cover != none { 30pt } else { 36pt }, weight: "bold", data.title)
  v(16pt)
  text(size: 12pt, fill: muted, data.paper_title)
  v(4pt)
  text(size: 10pt, fill: muted, data.source)
  v(if cover != none { 12pt } else { 1fr })
})
#footer(1)

// ── Content slides ─────────────────────────────────────────────────────────────────
#for (i, slide) in data.slides.enumerate() {
  let img = slide.at("image", default: none)
  pagebreak()
  block(height: 100%, {
    box(inset: (x: 7pt, y: 3pt), radius: 4pt, fill: rgb(theme.accent_muted),
      text(size: 10pt, fill: accent, weight: "bold", str(i + 1).clusters().join()))
    v(10pt)
    text(size: 25pt, weight: "bold", slide.title)
    v(if img == none { 22pt } else { 14pt })
    for b in slide.bullets {
      grid(columns: (12pt, 1fr), column-gutter: 6pt,
        move(dy: 2pt, circle(radius: 3pt, fill: accent, stroke: none)),
        text(size: if img == none { 17pt } else { 14.5pt }, b))
      v(if img == none { 14pt } else { 8pt })
    }
    if img != none {
      v(4pt)
      block(height: 1fr, width: 100%, clip: true, radius: 8pt,
        fill: rgb(theme.surface), inset: 8pt,
        align(center + horizon, image(img, fit: "contain", width: 100%, height: 100%)))
      let cap = slide.at("image_caption", default: none)
      if cap != none and cap != "" {
        v(5pt)
        align(center, text(size: 9.5pt, fill: muted, cap))
      }
      v(12pt)  // keep the caption clear of the footer rule
    }
  })
  footer(i + 2)
}
