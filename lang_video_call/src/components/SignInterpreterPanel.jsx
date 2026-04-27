import { useEffect, useRef, useState } from "react";
import "./SignInterpreterPanel.css";

const API_BASE_URL = import.meta.env.VITE_SIGN_API_URL || "http://127.0.0.1:8000";
const CAPTURE_INTERVAL_MS = 900;

export default function SignInterpreterPanel({ userName }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const intervalRef = useRef(null);
  const requestInFlightRef = useRef(false);
  const sessionIdRef = useRef("");

  const [isRunning, setIsRunning] = useState(false);
  const [status, setStatus] = useState("Ready to start");
  const [error, setError] = useState("");
  const [transcript, setTranscript] = useState("");
  const [currentLabel, setCurrentLabel] = useState("");
  const [candidateLabel, setCandidateLabel] = useState("");
  const [confidence, setConfidence] = useState(0);
  const [topPredictions, setTopPredictions] = useState([]);
  const [hasHand, setHasHand] = useState(false);

  function releaseResources() {
    if (intervalRef.current) {
      window.clearInterval(intervalRef.current);
      intervalRef.current = null;
    }

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    requestInFlightRef.current = false;
    sessionIdRef.current = "";
  }

  useEffect(() => {
    return () => {
      releaseResources();
    };
  }, []);

  async function checkBackend() {
    const response = await fetch(`${API_BASE_URL}/api/health`);
    if (!response.ok) {
      throw new Error("Backend health check failed.");
    }
  }

  async function startInterpreter() {
    setError("");
    setStatus("Checking Python backend...");

    try {
      await checkBackend();
    } catch {
      setError("Start the Python API first: python sign_language_api.py");
      setStatus("Backend offline");
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: "user",
          width: { ideal: 640 },
          height: { ideal: 480 },
        },
        audio: false,
      });

      streamRef.current = stream;
      sessionIdRef.current = "";

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }

      setIsRunning(true);
      setStatus("Listening for signs...");
      intervalRef.current = window.setInterval(captureFrame, CAPTURE_INTERVAL_MS);
      captureFrame();
    } catch {
      setError("Unable to open a local camera stream for interpretation.");
      setStatus("Camera unavailable");
    }
  }

  function stopInterpreter() {
    releaseResources();
    setIsRunning(false);
    setStatus("Interpreter stopped");
  }

  async function clearTranscript() {
    setTranscript("");
    setCurrentLabel("");
    setCandidateLabel("");
    setTopPredictions([]);
    setConfidence(0);
    setHasHand(false);

    if (!sessionIdRef.current) {
      return;
    }

    try {
      await fetch(`${API_BASE_URL}/api/reset-session`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ sessionId: sessionIdRef.current }),
      });
    } catch {
      setError("Unable to reset the interpreter session.");
    }
  }

  async function captureFrame() {
    if (
      requestInFlightRef.current ||
      !videoRef.current ||
      !canvasRef.current ||
      videoRef.current.readyState < 2
    ) {
      return;
    }

    requestInFlightRef.current = true;

    try {
      const canvas = canvasRef.current;
      const context = canvas.getContext("2d");
      canvas.width = 320;
      canvas.height = 240;
      context.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);

      const response = await fetch(`${API_BASE_URL}/api/predict-frame`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          image: canvas.toDataURL("image/jpeg", 0.82),
          sessionId: sessionIdRef.current || undefined,
        }),
      });

      if (!response.ok) {
        throw new Error("Prediction request failed.");
      }

      const data = await response.json();
      sessionIdRef.current = data.sessionId;
      setTranscript(data.transcript || "");
      setCurrentLabel(data.label || "");
      setCandidateLabel(data.candidateLabel || "");
      setConfidence(data.confidence || 0);
      setTopPredictions(data.topPredictions || []);
      setHasHand(Boolean(data.hasHand));

      if (!data.hasHand) {
        setStatus("Show your hand inside the preview");
      } else if (data.label) {
        setStatus(`Recognized: ${data.label}`);
      } else if (data.candidateLabel) {
        setStatus(`Stabilizing: ${data.candidateLabel}`);
      } else {
        setStatus("Hand detected");
      }
    } catch {
      setError("The interpreter request failed. Check whether the Python API is still running.");
      setStatus("Connection issue");
    } finally {
      requestInFlightRef.current = false;
    }
  }

  const activeLabel = currentLabel || candidateLabel || "Waiting...";

  return (
    <aside className="interpreter-panel">
      <div className="panel-header">
        <div>
          <p className="panel-eyebrow">Live Sign Interpreter</p>
          <h2>{userName ? `${userName}'s local captions` : "Local captions"}</h2>
        </div>
        <span className={`status-pill ${isRunning ? "active" : ""}`}>{status}</span>
      </div>

      <div className="preview-shell">
        <video ref={videoRef} autoPlay muted playsInline />
        {!isRunning && <div className="preview-placeholder">Interpreter preview</div>}
        <canvas ref={canvasRef} hidden />
      </div>

      <div className="signal-grid">
        <div className="signal-card">
          <span className="signal-label">Current sign</span>
          <strong>{activeLabel}</strong>
        </div>
        <div className="signal-card">
          <span className="signal-label">Confidence</span>
          <strong>{hasHand ? `${Math.round(confidence * 100)}%` : "--"}</strong>
        </div>
      </div>

      <div className="transcript-block">
        <span className="signal-label">Transcript</span>
        <p>{transcript || "Recognized signs will appear here."}</p>
      </div>

      <div className="prediction-strip">
        {topPredictions.map((item) => (
          <span key={item.label} className="prediction-chip">
            {item.label} {Math.round(item.confidence * 100)}%
          </span>
        ))}
      </div>

      <div className="panel-actions">
        {!isRunning ? (
          <button type="button" className="primary-action" onClick={startInterpreter}>
            Start Interpreter
          </button>
        ) : (
          <button type="button" className="primary-action danger" onClick={stopInterpreter}>
            Stop Interpreter
          </button>
        )}

        <button type="button" className="ghost-action" onClick={clearTranscript}>
          Clear Transcript
        </button>
      </div>

      {error && <p className="panel-error">{error}</p>}
      <p className="panel-note">
        Uses your local camera feed and the Python backend at {API_BASE_URL}.
      </p>
    </aside>
  );
}