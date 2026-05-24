import { useState } from 'react'
import './App.css'

const PLACEHOLDER_SVG = `
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 640 480" role="img" aria-label="placeholder svg output">
  <rect width="640" height="480" fill="#E8E7E2"/>
  <text x="320" y="240" text-anchor="middle" dominant-baseline="middle" fill="#A0A090" font-family="IBM Plex Mono" font-size="11" font-style="italic">svg output renders here</text>
</svg>`
const GENERATE_URL = 'https://rhb6mf70r2.execute-api.us-east-1.amazonaws.com/api/generate'

function App() {
  const [view, setView] = useState('prompt')
  const [prompt, setPrompt] = useState('')
  const [feedback, setFeedback] = useState('')
  const [rating, setRating] = useState(0)
  const [hoverRating, setHoverRating] = useState(0)
  const [media, setMedia] = useState([])
  const [svg, setSvg] = useState('')
  const [error, setError] = useState('')
  const [status, setStatus] = useState('')

  const desc = {
    prompt: 'describe a technical diagram. receive an svg.',
    loading: 'describe a technical diagram. receive an svg.',
    feedback: 'how did we do?',
    done: 'your svg is ready.',
  }[view]

  async function generate({ revision = false } = {}) {
    if (!prompt.trim()) {
      setError('enter a prompt first.')
      return
    }
    if (revision && (!rating || !feedback.trim())) {
      setError('rating and feedback are required.')
      return
    }
    setError('')
    setStatus('')
    setView('loading')
    try {
      const response = await fetch(GENERATE_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt,
          feedback: revision ? feedback : '',
          feedback_rating: revision ? rating : null,
          max_tokens: 512,
          cache_namespace: 'preview-valid-svg-v1',
          reference_images: media.map((item) => item.uri),
          media_metadata: media.reduce((acc, item) => {
            acc[item.uri] = {
              filename: item.name,
              size: item.size,
              mime_type: item.type,
            }
            return acc
          }, {}),
        }),
      })
      const body = await response.json()
      if (!response.ok || !body.svg) {
        throw new Error(body.message || 'generation failed.')
      }
      if (!String(body.svg).includes('<svg')) {
        throw new Error('generation did not return svg.')
      }
      setSvg(body.svg)
      setStatus(body.cached ? 'loaded from cache.' : 'generated.')
      setFeedback('')
      setRating(0)
      setHoverRating(0)
      setView('feedback')
    } catch (err) {
      setError(err.message || 'generation failed.')
      setView(svg ? 'feedback' : 'prompt')
    }
  }

  function unlock() {
    if (!rating || !feedback.trim()) {
      setError('rating and feedback are required.')
      return
    }
    setError('')
    setView('done')
  }

  function reset() {
    setView('prompt')
    setPrompt('')
    setFeedback('')
    setRating(0)
    setHoverRating(0)
    setMedia([])
    setSvg('')
    setError('')
    setStatus('')
  }

  function addMedia(event) {
    const files = Array.from(event.target.files || [])
    setMedia((current) => {
      const next = [...current]
      files.forEach((file) => {
        const id = `${file.name}-${file.size}-${file.lastModified}`
        if (!next.some((item) => item.id === id)) {
          next.push({
            id,
            name: file.name,
            size: file.size,
            type: file.type || 'application/octet-stream',
            uri: `local://${id}`,
          })
        }
      })
      return next
    })
    event.target.value = ''
  }

  function downloadSvg() {
    const blob = new Blob([svg], { type: 'image/svg+xml;charset=utf-8' })
    const url = URL.createObjectURL(blob)
    const anchor = document.createElement('a')
    anchor.href = url
    anchor.download = 'svgen.svg'
    document.body.appendChild(anchor)
    anchor.click()
    anchor.remove()
    URL.revokeObjectURL(url)
  }

  return (
    <main className="page">
      <h1>svgen</h1>
      <p className="desc">{desc}</p>

      {view === 'loading' && (
        <div className="loading-bar" aria-label="generating">
          <span />
        </div>
      )}

      {view === 'prompt' && (
        <>
          <div className="prompt-row">
            <textarea
              value={prompt}
              onChange={(event) => setPrompt(event.target.value)}
              placeholder={'a sequence diagram showing\njwt auth flow between client, api gateway,\nand auth service...'}
              aria-label="diagram prompt"
            />
            <button type="button" onClick={() => generate()}>generate →</button>
          </div>
          <span className="hint">— be specific about layout, connections, and style.</span>
          {error && <span className="error-text">{error}</span>}

          <div className="media-upload">
            <span className="field-label">reference media (optional)</span>
            <div className="upload-actions">
              <label className="upload-button" htmlFor="media-input">attach media</label>
              <span className="upload-hint">images, video, or pdfs.</span>
            </div>
            <input
              id="media-input"
              className="file-input"
              type="file"
              accept="image/*,video/*,application/pdf"
              multiple
              onChange={addMedia}
            />
            <div className="upload-list">
              {media.length ? (
                media.map((item) => (
                  <div className="upload-item" key={item.id}>
                    <span>{item.name}</span>
                    <button
                      type="button"
                      onClick={() => setMedia((current) => current.filter((entry) => entry.id !== item.id))}
                    >
                      remove
                    </button>
                  </div>
                ))
              ) : (
                'no media attached.'
              )}
            </div>
          </div>
        </>
      )}

      {(view === 'feedback' || view === 'done') && (
        <>
          <div className="svg-frame" dangerouslySetInnerHTML={{ __html: svg || PLACEHOLDER_SVG }} />
          {status && <span className="status-text">{status}</span>}
        </>
      )}

      {view === 'feedback' && (
        <>
          <hr />
          <span className="field-label">rating</span>
          <div className="stars" onMouseLeave={() => setHoverRating(0)}>
            {[1, 2, 3, 4, 5].map((value) => {
              const active = value <= (hoverRating || rating)
              return (
                <button
                  key={value}
                  type="button"
                  className={active ? 'active' : ''}
                  onMouseEnter={() => setHoverRating(value)}
                  onClick={() => setRating(value)}
                  aria-label={`rate ${value}`}
                >
                  ★
                </button>
              )
            })}
          </div>

          <span className="field-label">what should change?</span>
          <textarea
            className="feedback-ta"
            value={feedback}
            onChange={(event) => setFeedback(event.target.value)}
            placeholder="incorrect connector routing, missing labels, wrong layout..."
            aria-label="feedback"
          />
          {error && <span className="error-text">{error}</span>}

          <div className="btn-row">
            <button type="button" onClick={() => generate({ revision: true })}>iterate →</button>
            <button className="secondary dark" type="button" onClick={unlock}>unlock download →</button>
          </div>
        </>
      )}

      {view === 'done' && (
        <>
          <span className="unlock-label">download unlocked.</span>
          <div className="result-actions">
            <button type="button" onClick={downloadSvg}>↓ download .svg</button>
            <button className="secondary dark" type="button" onClick={reset}>← new prompt</button>
          </div>
        </>
      )}
    </main>
  )
}

export default App
