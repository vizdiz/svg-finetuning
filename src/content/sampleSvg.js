function buildMockSvg({ title, connectorColor, note, prompt, feedback, revisionIndex, branchId }) {
  const promptLine = String(prompt || '').slice(0, 32)
  const feedbackLine = String(feedback || '').slice(0, 40)
  const branchLine = String(branchId || '').slice(0, 12)
  const safeTitle = String(title || 'svgen preview')
  const safeNote = String(note || 'output')

  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 640 360" role="img" aria-label="Technical diagram">
  <rect width="640" height="360" fill="#E8E7E2" />
  <text x="40" y="42" fill="#1A1A18" font-size="20" font-family="IBM Plex Mono">svgen preview</text>
  <text x="40" y="68" fill="#2F2F2A" font-size="14" font-family="IBM Plex Mono" font-style="italic">${safeTitle}</text>
  <rect x="36" y="108" width="126" height="76" fill="#F0EFEB" stroke="#1A1A18" stroke-width="2" />
  <rect x="257" y="108" width="126" height="76" fill="#F0EFEB" stroke="#1A1A18" stroke-width="2" />
  <rect x="478" y="108" width="126" height="76" fill="#F0EFEB" stroke="#1A1A18" stroke-width="2" />
  <text x="99" y="152" text-anchor="middle" fill="#1A1A18" font-size="18" font-family="IBM Plex Mono">start</text>
  <text x="320" y="152" text-anchor="middle" fill="#1A1A18" font-size="18" font-family="IBM Plex Mono">middle</text>
  <text x="541" y="152" text-anchor="middle" fill="#1A1A18" font-size="18" font-family="IBM Plex Mono">end</text>
  <path d="M162 146H257" stroke="${connectorColor}" stroke-width="3" />
  <path d="M383 146H478" stroke="${connectorColor}" stroke-width="3" />
  <rect x="188" y="232" width="264" height="66" fill="#F0EFEB" stroke="#D0CFC8" stroke-width="2" />
  <text x="320" y="258" text-anchor="middle" fill="#2F2F2A" font-size="16" font-family="IBM Plex Mono">${safeNote}</text>
  <text x="320" y="281" text-anchor="middle" fill="#2F2F2A" font-size="12" font-family="IBM Plex Mono">step:${revisionIndex} set:${branchLine}</text>
  <text x="320" y="302" text-anchor="middle" fill="#2F2F2A" font-size="12" font-family="IBM Plex Mono">${feedbackLine || promptLine}</text>
</svg>`.trim()
}

export function buildDraftSvg(payload = {}) {
  return buildMockSvg({
    title: 'draft preview',
    connectorColor: '#1A1A18',
    note: 'draft output',
    ...payload,
  })
}

export function buildRevisedSvg(payload = {}) {
  return buildMockSvg({
    title: 'revised preview',
    connectorColor: '#1A1A18',
    note: 'feedback applied',
    ...payload,
  })
}

export const DRAFT_SVG = buildDraftSvg()
export const REVISED_SVG = buildRevisedSvg()
