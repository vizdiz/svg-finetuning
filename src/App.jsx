import './App.css'
import PreviewPanel from './components/PreviewPanel'
import { DRAFT_SVG, REVISED_SVG } from './content/sampleSvg'

const PROMPT_EXAMPLE =
  'a simple process diagram with three steps, clear labels, and straight lines...'
const FEEDBACK_EXAMPLE = 'make the connectors cleaner and keep the layout balanced.'

function App() {
  return (
    <main className="page">
      <section className="shell">
        <header className="hero">
          <h1>svgen</h1>
          <p className="lede">a preview of the product flow. coming soon.</p>
        </header>

        <div className="preview-grid">
          <PreviewPanel eyebrow="01 - PROMPT" title="svgen">
            <p className="panel-desc">start with a prompt.</p>
            <div className="input-box prompt-input static-copy" aria-hidden="true">
              {PROMPT_EXAMPLE}
            </div>
            <div className="button-row" aria-hidden="true">
              <div className="gold-button static-chip">generate draft →</div>
              <div className="link-button static-link">new prompt</div>
            </div>
            <p className="hint">- revise after the first draft, then download each version.</p>
          </PreviewPanel>

          <PreviewPanel eyebrow="02 - DRAFT & FEEDBACK" title="svgen">
            <p className="panel-desc">share what should change.</p>
            <div className="svg-frame">
              <div className="svg-render" dangerouslySetInnerHTML={{ __html: DRAFT_SVG }} />
            </div>
            <p className="field-label">what should change?</p>
            <div className="input-box feedback-input static-copy" aria-hidden="true">
              {FEEDBACK_EXAMPLE}
            </div>
            <div className="rating-row" aria-label="Rating preview" aria-hidden="true">
              {[1, 2, 3, 4, 5].map((value) => (
                <span key={value} className={`star-button ${value <= 4 ? 'active' : ''}`}>
                  ★
                </span>
              ))}
            </div>
            <div className="button-row" aria-hidden="true">
              <div className="gold-button static-chip">revise with feedback →</div>
              <div className="link-button static-link">download draft</div>
            </div>
          </PreviewPanel>

          <PreviewPanel eyebrow="03 - REVISE & DOWNLOAD" title="svgen">
            <p className="panel-desc">download your current result.</p>
            <div className="svg-frame">
              <div className="svg-render" dangerouslySetInnerHTML={{ __html: REVISED_SVG }} />
            </div>
            <p className="unlock-label">download unlocked.</p>
            <div className="button-row" aria-hidden="true">
              <div className="gold-button static-chip">download revised</div>
              <div className="link-button static-link">new prompt</div>
            </div>
            <p className="hint">your changes stay with this version.</p>
          </PreviewPanel>
        </div>
      </section>
    </main>
  )
}

export default App
