#set page(width: 820pt, height: auto, margin: 12pt)
#set text(font: "New Computer Modern", size: 9pt, fill: rgb("#2d3436"))

#import "@preview/cetz:0.4.0": canvas, draw

// ====== Color Palette ======
#let c-frozen = (fill: rgb("#d9dee3"), stroke: rgb("#7e8a97"))
#let c-mpo60 = (fill: rgb("#cfe5ff"), stroke: rgb("#3a78c2"))
#let c-mpo30 = (fill: rgb("#d7f0ee"), stroke: rgb("#2e8b7d"))
#let c-skip = (fill: rgb("#f8e3b2"), stroke: rgb("#b7791f"))
#let c-embed = (fill: rgb("#e8d9f6"), stroke: rgb("#7a4fa3"))
#let c-core = (fill: rgb("#e3f0ff"), stroke: rgb("#3a78c2"))
#let c-title = rgb("#1e3a5f")

// ====== Helpers ======
#let section-label(body) = text(weight: "bold", size: 10.5pt, fill: c-title, body)
#let meta(body) = text(size: 7.5pt, fill: rgb("#636e72"), body)
#let mono(body) = text(font: "DejaVu Sans Mono", size: 7pt, body)
#let core-pill(label) = box(
  fill: c-core.fill, stroke: c-core.stroke + 0.8pt, radius: 10pt,
  inset: (x: 5pt, y: 2.5pt),
  text(size: 7pt, weight: "bold", fill: rgb("#2c5282"), label)
)
#let bond = text(size: 10pt, fill: rgb("#3a78c2"), weight: "bold")[$dash.em$]
#let sec-box(colors, body) = rect(
  fill: colors.fill, stroke: colors.stroke + 1.2pt, radius: 5pt,
  inset: 8pt, width: 100%, body
)

// ====== TITLE ======
#rect(fill: c-title, stroke: none, radius: 5pt, inset: (x: 16pt, y: 10pt), width: 100%)[
  #grid(columns: (1fr, auto), align: (left + horizon, right + horizon),
    text(fill: white, weight: "bold", size: 13pt)[MPO-Compressed Model Architecture],
    text(fill: rgb("#a8c8e8"), size: 9pt)[
      TinyLlama-1.1B #sym.dot.c 22 blocks #sym.dot.c
      #sym.chi = 60 / 30 two-stage schedule #sym.dot.c
      1100M #sym.arrow 475M (43.2%)
    ],
  )
]

#v(10pt)

// ====== MAIN 2-COLUMN LAYOUT ======
#grid(columns: (52%, 48%), gutter: 12pt,
  // ========= LEFT COLUMN: Model Overview =========
  [
    // --- Embedding ---
    #sec-box(c-embed)[
      #section-label[embed\_tokens] #h(6pt) #meta[Embedding(32000, 2048) · 65.5M params · not compressed]
    ]

    #v(6pt)
    #align(center, text(fill: rgb("#b2bec3"), size: 9pt)[$arrow.b$])
    #v(4pt)

    // --- Frozen ---
    #sec-box(c-frozen)[
      #section-label[Blocks 0--1: Frozen] #h(6pt) #meta[Original nn.Linear, no compression]
      #v(5pt)
      #text(size: 8pt)[All 7 projections kept dense:]
      #v(3pt)
      #grid(columns: (1fr,) * 4, gutter: 3pt,
        ..("q_proj 2048×2048", "k_proj 2048×256", "v_proj 2048×256", "o_proj 2048×2048").map(t =>
          box(fill: white, stroke: c-frozen.stroke + 0.5pt, radius: 2pt, inset: 3pt, mono(t))
        ),
      )
      #v(2pt)
      #grid(columns: (1fr,) * 3, gutter: 3pt,
        ..("gate_proj 2048×5632", "up_proj 2048×5632", "down_proj 5632×2048").map(t =>
          box(fill: white, stroke: c-frozen.stroke + 0.5pt, radius: 2pt, inset: 3pt, mono(t))
        ),
      )
      #align(right, meta[~44M params/block × 2 = 88M])
    ]

    #v(6pt)
    #align(center, text(fill: rgb("#b2bec3"), size: 9pt)[$arrow.b$ × 2 blocks])
    #v(4pt)

    // --- MPO χ=60 ---
    #sec-box(c-mpo60)[
      #section-label[Blocks 2--13: MPO, #sym.chi = 60]
      #h(6pt) #meta[12 blocks · 6 MPO layers + 1 dense skip per block]
      #v(6pt)

      #grid(columns: (auto, 1fr), gutter: 6pt, align: horizon,
        text(weight: "bold", size: 8pt)[Attention:],
        [
          #text(size: 8pt)[q, k, v, o #sym.arrow MPOLinear (3 cores)]
          #v(2pt)
          q/o: #core-pill[\[1,16,16,60\]] #bond #core-pill[\[60,8,8,60\]] #bond #core-pill[\[60,16,16,1\]] #h(4pt) #meta[261K (6.2%)]
          #v(2pt)
          k/v: #core-pill[\[1,8,16,60\]] #bond #core-pill[\[60,4,8,60\]] #bond #core-pill[\[60,8,16,1\]] #h(4pt) #meta[131K (25%)]
        ],
      )

      #v(6pt)

      #grid(columns: (auto, 1fr), gutter: 6pt, align: horizon,
        text(weight: "bold", size: 8pt)[MLP:],
        [
          #text(size: 8pt)[gate, up #sym.arrow MPOLinear (3 cores)]
          #v(2pt)
          #core-pill[\[1,22,16,60\]] #bond #core-pill[\[60,16,8,60\]] #bond #core-pill[\[60,16,16,1\]] #h(4pt) #meta[497K (4.3%)]
          #v(4pt)
          #box(fill: c-skip.fill, stroke: c-skip.stroke + 0.6pt, radius: 3pt, inset: 3pt,
            text(size: 7.5pt, fill: rgb("#7d6608"), weight: "bold")[down\_proj: kept dense (11.5M) — sensitivity]
          )
        ],
      )
    ]

    #v(6pt)
    #align(center, text(fill: rgb("#b2bec3"), size: 9pt)[$arrow.b$ × 12 blocks])
    #v(4pt)

    // --- MPO χ=30 ---
    #sec-box(c-mpo30)[
      #section-label[Blocks 14--21: MPO, #sym.chi = 30]
      #h(6pt) #meta[8 blocks · more aggressive compression]
      #v(6pt)

      #grid(columns: (auto, 1fr), gutter: 6pt, align: horizon,
        text(weight: "bold", size: 8pt)[Attention:],
        [
          q/o: #core-pill[\[1,16,16,30\]] #bond #core-pill[\[30,8,8,30\]] #bond #core-pill[\[30,16,16,1\]] #h(4pt) #meta[73K (1.7%)]
          #v(2pt)
          k/v: #core-pill[\[1,8,16,30\]] #bond #core-pill[\[30,4,8,30\]] #bond #core-pill[\[30,8,16,1\]] #h(4pt) #meta[37K (7.0%)]
        ],
      )

      #v(4pt)

      #grid(columns: (auto, 1fr), gutter: 6pt, align: horizon,
        text(weight: "bold", size: 8pt)[MLP:],
        [
          gate/up: #core-pill[\[1,22,16,30\]] #bond #core-pill[\[30,16,8,30\]] #bond #core-pill[\[30,16,16,1\]] #h(4pt) #meta[133K (1.2%)]
          #v(3pt)
          #box(fill: c-skip.fill, stroke: c-skip.stroke + 0.6pt, radius: 3pt, inset: 3pt,
            text(size: 7.5pt, fill: rgb("#7d6608"), weight: "bold")[down\_proj: kept dense (11.5M)]
          )
        ],
      )
    ]

    #v(6pt)
    #align(center, text(fill: rgb("#b2bec3"), size: 9pt)[$arrow.b$ × 8 blocks])
    #v(4pt)

    // --- Head ---
    #sec-box(c-embed)[
      #grid(columns: (1fr, 1fr), gutter: 10pt,
        [#section-label[RMSNorm] #h(4pt) #meta[dim=2048]],
        [#section-label[lm\_head] #h(4pt) #meta[Linear(2048 #sym.arrow 32000) · 65.5M · not compressed]],
      )
    ]
  ],

  // ========= RIGHT COLUMN: Detail + Legend =========
  [
    // --- Tensor Network Inset ---
    #rect(fill: rgb("#f8f9fa"), stroke: c-title + 1.5pt, radius: 6pt, inset: 10pt, width: 100%)[
      #align(center, text(weight: "bold", size: 10pt, fill: c-title)[
        MPO Factorization of q\_proj
      ])
      #v(2pt)
      #align(center, meta[Dense weight reshaped and decomposed into 3 tensor cores])
      #v(10pt)

      // Tensor network diagram via CeTZ
      #align(center, canvas(length: 1pt, {
        import draw: *

        // Dense matrix
        rect((-140, -28), (-40, 28), fill: rgb("#fce4e4"), stroke: rgb("#c0392b") + 1.2pt, radius: 4pt, name: "dense")
        content("dense", [
          #align(center)[
            #text(weight: "bold", size: 9pt)[$W_q$]
            #v(1pt)
            #text(size: 7pt, fill: gray)[2048 × 2048]
            #v(1pt)
            #text(size: 6.5pt, fill: gray)[4.19M params]
          ]
        ])

        // Arrow
        line((-32, 0), (-8, 0), mark: (end: "straight"), stroke: rgb("#555") + 1.2pt)
        content((-20, 8), text(size: 6.5pt, fill: rgb("#555"))[reshape +\ TT-SVD], anchor: "south")

        // G1
        circle((40, 0), radius: 22, fill: c-mpo60.fill, stroke: c-mpo60.stroke + 1.5pt, name: "g1")
        content("g1", text(weight: "bold", size: 10pt, fill: rgb("#2c5282"))[$G_1$])
        // G1 legs
        line((40, 22), (40, 48), stroke: rgb("#8e44ad") + 1pt)
        content((40, 54), text(size: 6pt, fill: rgb("#8e44ad"))[o#sub[1]=16], anchor: "south")
        line((40, -22), (40, -48), stroke: rgb("#27ae60") + 1pt)
        content((40, -54), text(size: 6pt, fill: rgb("#27ae60"))[i#sub[1]=16], anchor: "north")

        // Bond 1
        line((62, 0), (108, 0), stroke: c-mpo60.stroke + 2pt)
        content((85, 8), text(size: 7pt, fill: c-mpo60.stroke, weight: "bold")[$chi$], anchor: "south")

        // G2
        circle((130, 0), radius: 22, fill: c-mpo60.fill, stroke: c-mpo60.stroke + 1.5pt, name: "g2")
        content("g2", text(weight: "bold", size: 10pt, fill: rgb("#2c5282"))[$G_2$])
        // G2 legs
        line((130, 22), (130, 48), stroke: rgb("#8e44ad") + 1pt)
        content((130, 54), text(size: 6pt, fill: rgb("#8e44ad"))[o#sub[2]=8], anchor: "south")
        line((130, -22), (130, -48), stroke: rgb("#27ae60") + 1pt)
        content((130, -54), text(size: 6pt, fill: rgb("#27ae60"))[i#sub[2]=8], anchor: "north")

        // Bond 2
        line((152, 0), (198, 0), stroke: c-mpo60.stroke + 2pt)
        content((175, 8), text(size: 7pt, fill: c-mpo60.stroke, weight: "bold")[$chi$], anchor: "south")

        // G3
        circle((220, 0), radius: 22, fill: c-mpo60.fill, stroke: c-mpo60.stroke + 1.5pt, name: "g3")
        content("g3", text(weight: "bold", size: 10pt, fill: rgb("#2c5282"))[$G_3$])
        // G3 legs
        line((220, 22), (220, 48), stroke: rgb("#8e44ad") + 1pt)
        content((220, 54), text(size: 6pt, fill: rgb("#8e44ad"))[o#sub[3]=16], anchor: "south")
        line((220, -22), (220, -48), stroke: rgb("#27ae60") + 1pt)
        content((220, -54), text(size: 6pt, fill: rgb("#27ae60"))[i#sub[3]=16], anchor: "north")
      }))

      #v(8pt)

      // Annotations
      #align(center)[
        #text(size: 8pt, fill: rgb("#555"))[
          $d_"out" = underbrace(16, o_1) times underbrace(8, o_2) times underbrace(16, o_3) = 2048$
          #h(12pt)
          $d_"in" = underbrace(16, i_1) times underbrace(8, i_2) times underbrace(16, i_3) = 2048$
        ]
      ]

      #v(8pt)

      // Parameter breakdown
      #rect(fill: rgb("#fff8e1"), stroke: rgb("#f0c040") + 0.8pt, radius: 3pt, inset: 6pt, width: 100%)[
        #align(center)[
          #text(size: 8pt)[
            #text(weight: "bold")[Parameters (#sym.chi = 60):]
            #h(4pt)
            $underbrace(1 dot 16 dot 16 dot 60, "G"_1 ": 15K")
            + underbrace(60 dot 8 dot 8 dot 60, "G"_2 ": 230K")
            + underbrace(60 dot 16 dot 16 dot 1, "G"_3 ": 15K")
            =$ #text(weight: "bold", fill: rgb("#2e86c1"))[261,120]
          ]
          #v(2pt)
          #text(size: 8pt, fill: rgb("#555"))[
            Compression: 4,194,304 #sym.arrow 261,120 #h(4pt)
            #text(weight: "bold", fill: rgb("#27ae60"))[(6.2% of original, 16× smaller)]
          ]
        ]
      ]
    ]

    #v(12pt)

    // --- Core Shape Table ---
    #rect(fill: rgb("#f8f9fa"), stroke: rgb("#bdc3c7") + 0.8pt, radius: 5pt, inset: 8pt, width: 100%)[
      #text(weight: "bold", size: 9pt, fill: c-title)[Per-Layer Compression Summary]
      #v(6pt)
      #table(
        columns: (auto, auto, auto, auto, auto),
        stroke: rgb("#dee2e6") + 0.5pt,
        inset: 5pt,
        align: (left, center, center, center, center),
        table.header(
          text(weight: "bold", size: 7.5pt)[Projection],
          text(weight: "bold", size: 7.5pt)[Dense],
          text(weight: "bold", size: 7.5pt)[#sym.chi=60],
          text(weight: "bold", size: 7.5pt)[#sym.chi=30],
          text(weight: "bold", size: 7.5pt)[Retain],
        ),
        text(size: 7.5pt)[q\_proj / o\_proj], meta[4.19M], meta[261K], meta[73K], meta[6.2% / 1.7%],
        text(size: 7.5pt)[k\_proj / v\_proj], meta[0.52M], meta[131K], meta[37K], meta[25% / 7.0%],
        text(size: 7.5pt)[gate / up\_proj], meta[11.5M], meta[497K], meta[133K], meta[4.3% / 1.2%],
        text(size: 7.5pt)[down\_proj], meta[11.5M], meta[11.5M], meta[11.5M], text(size: 7.5pt, fill: rgb("#b7791f"))[kept],
      )
    ]

    #v(12pt)

    // --- Compression Schedule Summary ---
    #rect(fill: rgb("#f0f4f8"), stroke: c-title + 0.8pt, radius: 5pt, inset: 8pt, width: 100%)[
      #text(weight: "bold", size: 9pt, fill: c-title)[Compression Schedule]
      #v(5pt)
      #grid(columns: (auto, 1fr), gutter: (6pt, 4pt),
        text(weight: "bold", size: 8pt)[Blocks 0--1:],
        meta[Frozen — early blocks critical for quality],
        text(weight: "bold", size: 8pt)[Blocks 2--13:],
        meta[#sym.chi = 60 — moderate compression, 12 blocks],
        text(weight: "bold", size: 8pt)[Blocks 14--21:],
        meta[#sym.chi = 30 — aggressive compression, 8 deeper blocks],
        text(weight: "bold", size: 8pt)[down\_proj:],
        meta[Kept dense in all blocks — sensitive to factorization],
        text(weight: "bold", size: 8pt)[Embed / Head:],
        meta[Not compressed — embed\_tokens (65.5M) + lm\_head (65.5M)],
      )
    ]

    #v(12pt)

    // --- Legend ---
    #rect(fill: white, stroke: rgb("#dee2e6") + 0.5pt, radius: 4pt, inset: 8pt, width: 100%)[
      #text(weight: "bold", size: 8.5pt, fill: c-title)[Legend]
      #v(4pt)
      #grid(columns: (1fr, 1fr), gutter: 4pt,
        [#box(fill: c-frozen.fill, stroke: c-frozen.stroke, width: 10pt, height: 10pt, radius: 2pt, baseline: 2pt) #text(size: 7.5pt)[ Frozen (dense)]],
        [#box(fill: c-mpo60.fill, stroke: c-mpo60.stroke, width: 10pt, height: 10pt, radius: 2pt, baseline: 2pt) #text(size: 7.5pt)[ MPO #sym.chi = 60]],
        [#box(fill: c-mpo30.fill, stroke: c-mpo30.stroke, width: 10pt, height: 10pt, radius: 2pt, baseline: 2pt) #text(size: 7.5pt)[ MPO #sym.chi = 30]],
        [#box(fill: c-skip.fill, stroke: c-skip.stroke, width: 10pt, height: 10pt, radius: 2pt, baseline: 2pt) #text(size: 7.5pt)[ Skipped (sensitive)]],
        [#box(fill: c-embed.fill, stroke: c-embed.stroke, width: 10pt, height: 10pt, radius: 2pt, baseline: 2pt) #text(size: 7.5pt)[ Embedding / Head]],
        [#line(length: 10pt, stroke: rgb("#3a78c2") + 2pt) #text(size: 7.5pt)[ Bond dimension #sym.chi]],
      )
    ]
  ],
)
