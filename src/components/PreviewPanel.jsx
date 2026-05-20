function PreviewPanel({ eyebrow, title, children }) {
  return (
    <section className="panel">
      <p className="panel-eyebrow">{eyebrow}</p>
      <h2>{title}</h2>
      {children}
    </section>
  )
}

export default PreviewPanel
