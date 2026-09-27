import React, { useState, useEffect, useRef } from 'react'
import { MapContainer, TileLayer, GeoJSON } from 'react-leaflet'
import 'leaflet/dist/leaflet.css'

import L from 'leaflet'
delete L.Icon.Default.prototype._getIconUrl
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png',
  iconUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png',
  shadowUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png',
})

function SplitPane({ position, children, onDrag }) {
  const ref = useRef(null)
  const [isDragging, setIsDragging] = useState(false)

  const handleMouseMove = (e) => {
    if (!isDragging || !ref.current) return
    const rect = ref.current.getBoundingClientRect()
    const x = e.clientX - rect.left
    const percent = (x / rect.width) * 100
    onDrag(Math.max(10, Math.min(90, percent)))
  }

  const handleMouseUp = () => {
    setIsDragging(false)
    document.removeEventListener('mousemove', handleMouseMove)
    document.removeEventListener('mouseup', handleMouseUp)
  }

  useEffect(() => {
    return () => {
      document.removeEventListener('mousemove', handleMouseMove)
      document.removeEventListener('mouseup', handleMouseUp)
    }
  }, [isDragging])

  return (
    <div
      ref={ref}
      className={`split-pane ${position}`}
      style={{ width: `${position}%` }}
      onMouseDown={(e) => {
        setIsDragging(true)
        document.addEventListener('mousemove', handleMouseMove)
        document.addEventListener('mouseup', handleMouseUp)
        e.preventDefault()
      }}
    >
      {children}
    </div>
  )
}

function SwipeHandle({ position, onDrag }) {
  const handleMove = (e) => {
    const mapContainer = document.querySelector('.viewer-area')
    if (!mapContainer) return
    const rect = mapContainer.getBoundingClientRect()
    const x = e.clientX - rect.left
    const percent = (x / rect.width) * 100
    onDrag(Math.max(10, Math.min(90, percent)))
  }

  const handleUp = () => {
    document.removeEventListener('mousemove', handleMove)
    document.removeEventListener('mouseup', handleUp)
  }

  return (
    <div
      className="overlay-swipe"
      style={{ left: `${position}%` }}
      onMouseDown={(e) => {
        e.stopPropagation()
        document.addEventListener('mousemove', handleMove)
        document.addEventListener('mouseup', handleUp)
        e.preventDefault()
      }}
    />
  )
}

function MatchPointLayer({ data, showUnrefined = true }) {
  if (!data || !data.features) return null

  const filtered = showUnrefined ? data.features : data.features.filter((f) => f.properties.refined)

  return (
    <GeoJSON
      data={{ type: 'FeatureCollection', features: filtered }}
      className="match-points-layer"
      pointToLayer={(feature, latlng) => {
        const refined = feature.properties.refined
        return L.circleMarker(latlng, {
          radius: 3,
          fillColor: refined ? '#4ade80' : '#f87171',
          color: refined ? '#22c55e' : '#ef4444',
          weight: 1.5,
          opacity: 0.9,
          fillOpacity: 0.8,
          className: refined ? 'refined' : 'unrefined',
        })
      }}
      onEachFeature={(feature, layer) => {
        const props = feature.properties
        layer.bindTooltip(
          `x: ${props.x_ref?.toFixed(2) || '?'}, y: ${props.y_ref?.toFixed(2) || '?'}${props.refined ? ' (refined)' : ''}`,
          { permanent: false, direction: 'top' }
        )
      }}
    />
  )
}

function ResidualHeatmap({ metrics }) {
  if (!metrics || !metrics.residual_map || !metrics.residual_map.per_cell_mean_px) return null

  const cellData = metrics.residual_map.per_cell_mean_px

  return (
    <div style={{ position: 'absolute', bottom: '12px', left: '12px', zIndex: 200, pointerEvents: 'none' }}>
      <div style={{ background: 'rgba(26,26,26,0.9)', padding: '8px 12px', borderRadius: '6px', border: '1px solid #333', fontSize: '0.7rem' }}>
        Residual Map (mean per cell):<br />
        {Object.entries(cellData).slice(0, 8).map(([cell, val]) => (
          <div key={cell}>{cell}: {parseFloat(val).toFixed(2)} px</div>
        ))}
      </div>
    </div>
  )
}

function App() {
  const [runs, setRuns] = useState([])
  const [selectedRun, setSelectedRun] = useState(null)
  const [splitPosition, setSplitPosition] = useState(50)
  const [viewMode, setViewMode] = useState('split')
  const [showUnrefined, setShowUnrefined] = useState(true)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    fetchRuns()
  }, [])

  const fetchRuns = async () => {
    try {
      const res = await fetch('/api/runs')
      if (res.ok) {
        const data = await res.json()
        setRuns(data)
      }
    } catch (e) {
      console.error('Failed to fetch runs:', e)
    }
  }

  const loadRun = async (runId) => {
    setLoading(true)
    setError(null)
    try {
      const [metricsRes, geojsonRes] = await Promise.all([
        fetch(`/runs/${runId}/metrics.json`),
        fetch(`/runs/${runId}/match_points.geojson`),
      ])

      if (!metricsRes.ok || !geojsonRes.ok) {
        throw new Error('Failed to load run data')
      }

      const [metrics, geojson] = await Promise.all([metricsRes.json(), geojsonRes.json()])
      setSelectedRun({ runId, metrics, geojson })
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  const trustFlag = selectedRun?.metrics?.trust_flag || 'UNKNOWN'
  const calibrationStatus = selectedRun?.metrics?.calibration_status || 'UNKNOWN'

  return (
    <div className="app">
      <header className="header">
        <h1>Chandra-Align Viewer</h1>
        <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
          <select
            value={selectedRun?.runId || ''}
            onChange={(e) => e.target.value && loadRun(e.target.value)}
            disabled={loading || runs.length === 0}
            style={{ padding: '6px 10px', background: '#222', border: '1px solid #333', borderRadius: '4px', color: '#eee' }}
          >
            <option value="">Select a run...</option>
            {runs.map((r) => (
              <option key={r.run_id} value={r.run_id}>
                {r.run_id} ({r.trust_flag})
              </option>
            ))}
          </select>

          {selectedRun && (
            <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
              <span
                className={`trust-badge ${
                  trustFlag.toLowerCase() === 'trusted'
                    ? 'trusted'
                    : trustFlag.toLowerCase() === 'not trusted'
                      ? 'not-trusted'
                      : 'uncalibrated'
                }`}
              >
                {trustFlag} {calibrationStatus === 'UNCALIBRATED' && '(UNCALIBRATED)'}
              </span>
            </div>
          )}
        </div>
      </header>

      <div className="main">
        <aside className="panel">
          <h2>Controls</h2>
          <div className="control-group">
            <label>View Mode</label>
            <select value={viewMode} onChange={(e) => setViewMode(e.target.value)}>
              <option value="split">Split Swipe</option>
              <option value="overlay">Overlay (50% opacity)</option>
              <option value="side-by-side">Side by Side</option>
            </select>
          </div>

          <div className="control-group">
            <label>
              <input type="checkbox" checked={showUnrefined} onChange={(e) => setShowUnrefined(e.target.checked)} />
              Show unrefined matches
            </label>
          </div>

          {viewMode === 'split' && (
            <div className="control-group">
              <label>Split Position: {splitPosition}%</label>
              <input
                type="range"
                min="10"
                max="90"
                value={splitPosition}
                onChange={(e) => setSplitPosition(Number(e.target.value))}
              />
            </div>
          )}
        </aside>

        <aside className="panel" style={{ borderLeft: '1px solid #333', borderRight: 'none', minWidth: '280px' }}>
          <h2>Run Info</h2>
          {loading ? (
            <div className="loading"><div className="spinner" />Loading run data...</div>
          ) : error ? (
            <div className="error-banner">{error}</div>
          ) : selectedRun ? (
            <div className="run-info">
              <div><strong>Run ID:</strong> {selectedRun.runId}</div>
              <div><strong>Trust Flag:</strong> {selectedRun.metrics.trust_flag}</div>
              <div><strong>Calibration:</strong> {selectedRun.metrics.calibration_status}</div>
              <div><strong>Matcher:</strong> {selectedRun.metrics.matcher}</div>
              <div><strong>Runtime:</strong> {selectedRun.metrics.runtime_s?.toFixed?.(2) || selectedRun.metrics.runtime_s}s</div>
              <div><strong>Raw Matches:</strong> {selectedRun.metrics.raw_matches ?? selectedRun.metrics.inliers?.raw_matches ?? 'UNMEASURED'}</div>
              <div><strong>Inliers:</strong> {selectedRun.metrics.inliers?.inliers ?? 'UNMEASURED'}</div>
              <div><strong>Inlier Ratio:</strong> {selectedRun.metrics.inliers?.inlier_ratio ?? 'UNMEASURED'}</div>
              <div><strong>Uniformity:</strong> {selectedRun.metrics.uniformity_score?.toFixed?.(3) ?? 'UNMEASURED'}</div>
              <div><strong>RMSE (held-out):</strong> {selectedRun.metrics.rmse?.rmse_px ?? 'UNMEASURED'} px</div>
              <div><strong>Approximation Flag:</strong> {selectedRun.metrics.approximation_flag ? 'Yes (geometry from metadata)' : 'No'}</div>
            </div>
          ) : (
            <div style={{ color: '#888', fontSize: '0.85rem' }}>Select a run from the dropdown to view details.</div>
          )}
        </aside>

        <div className="viewer-area" style={{ flex: 1 }}>
          {selectedRun ? (
            <MapViewer
              run={selectedRun}
              viewMode={viewMode}
              splitPosition={splitPosition}
              onSplitChange={setSplitPosition}
              onViewModeChange={setViewMode}
              showUnrefined={showUnrefined}
            />
          ) : (
            <div className="loading">
              <div className="spinner" />
              <span>Select a run to view</span>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

function MapViewer({ run, viewMode, splitPosition, onSplitChange, onViewModeChange, showUnrefined }) {
  const { metrics, geojson } = run
  const initialCenter = geojson?.features?.[0]
    ? [geojson.features[0].geometry.coordinates[1], geojson.features[0].geometry.coordinates[0]]
    : [0, 0]

  if (!geojson || !geojson.features || geojson.features.length === 0) {
    return (
      <div className="viewer-area" style={{ position: 'relative', display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%' }}>
        <div style={{ background: 'rgba(26,26,26,0.9)', padding: '20px', borderRadius: '8px', border: '1px solid #333', textAlign: 'center', color: '#aaa' }}>
          No match points available for this run
        </div>
      </div>
    )
  }

  return (
    <div className="viewer-area" style={{ position: 'relative' }}>
      <MapContainer
        center={initialCenter}
        zoom={14}
        style={{ width: '100%', height: '100%' }}
        attributionControl={false}
        zoomControl={true}
        scrollWheelZoom={true}
      >
        <TileLayer
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          attribution='&copy; OpenStreetMap contributors'
        />
        <MatchPointLayer data={geojson} showUnrefined={showUnrefined} />
        <ResidualHeatmap metrics={metrics} />

        {viewMode === 'split' && (
          <>
            <SplitPane position={splitPosition} onDrag={onSplitChange} />
            <SwipeHandle position={splitPosition} onDrag={onSplitChange} />
          </>
        )}
      </MapContainer>

      <div className="toolbar">
        <button className={viewMode === 'split' ? 'active' : ''} onClick={() => onViewModeChange('split')} title="Split swipe view">
          Split
        </button>
        <button className={viewMode === 'overlay' ? 'active' : ''} onClick={() => onViewModeChange('overlay')} title="Overlay view">
          Overlay
        </button>
        <button className={viewMode === 'side-by-side' ? 'active' : ''} onClick={() => onViewModeChange('side-by-side')} title="Side by side">
          Side-by-Side
        </button>
      </div>
    </div>
  )
}

export default App
