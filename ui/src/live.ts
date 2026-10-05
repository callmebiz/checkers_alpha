// Live analysis: one WebSocket to the server, a new request whenever the
// line, model or budget changes, and the latest snapshot for that request
// (older requests' snapshots are ignored). Reconnects if the server
// restarts.
import { useEffect, useRef, useState } from 'react'
import { type Analysis, type Line, liveUrl } from './api'

export interface LiveRequest {
  line: Line
  model: string
  sims: number // 0: keep searching (up to the server's cap)
}

export interface LiveResult {
  analysis: Analysis | null
  done: boolean
  error: string | null
}

/** Analyse `request` live; null stops. */
export function useLiveAnalysis(request: LiveRequest | null): LiveResult {
  const [result, setResult] = useState<LiveResult>({
    analysis: null,
    done: false,
    error: null,
  })
  const socket = useRef<WebSocket | null>(null)
  const id = useRef(0)
  const pending = useRef<string | null>(null) // sent once connected
  const [connected, setConnected] = useState(false)

  // keep a socket open, reconnecting after a second if it drops
  useEffect(() => {
    let closed = false
    let timer: ReturnType<typeof setTimeout> | undefined
    /** Open the socket and wire up its handlers. */
    const connect = () => {
      const ws = new WebSocket(liveUrl())
      socket.current = ws
      ws.onopen = () => {
        setConnected(true)
        if (pending.current) ws.send(pending.current)
      }
      ws.onmessage = (event) => {
        const msg = JSON.parse(event.data)
        if (msg.id !== id.current) return
        if (msg.error) {
          setResult({ analysis: null, done: true, error: msg.error })
        } else {
          const { id: _, done, ...analysis } = msg
          setResult({ analysis: analysis as Analysis, done, error: null })
        }
      }
      ws.onclose = () => {
        setConnected(false)
        if (!closed) timer = setTimeout(connect, 1000)
      }
    }
    connect()
    return () => {
      closed = true
      clearTimeout(timer)
      socket.current?.close()
    }
  }, [])

  // send a request (or a stop) whenever what to analyse changes
  const key = request ? JSON.stringify(request) : null
  useEffect(() => {
    const message = key
      ? JSON.stringify({ id: ++id.current, ...JSON.parse(key) })
      : JSON.stringify({ id: ++id.current, stop: true })
    pending.current = key ? message : null
    if (!key) setResult({ analysis: null, done: false, error: null })
    if (socket.current?.readyState === WebSocket.OPEN) {
      socket.current.send(message)
    }
  }, [key, connected])

  return result
}
