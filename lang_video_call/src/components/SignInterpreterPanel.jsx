import { useEffect, useRef, useState } from "react";
import "./SignInterpreterPanel.css";

const _rawApiBase =
  import.meta.env.VITE_SIGN_API_URL?.replace(/\/+$/, "") || "http://127.0.0.1:8000";
const API_BASE_URL = _rawApiBase;
const API_MIN_CONFIDENCE = 0.15;
const CAPTURE_INTERVAL_MS = 250;
const MODEL_SEQUENCE_LENGTH = 30;

function pickPreferredVoice(voices) {
  if (!voices.length) {
    return null;
  }

  const priorities = [
    "natural",
    "aria",
    "jenny",
    "guy",
    "sonia",
    "samantha",
    "zira",
    "david",
    "google us english",
    "microsoft",
  ];

  for (const token of priorities) {
    const match = voices.find((voice) => voice.name.toLowerCase().includes(token));
    if (match) {
      return match;
    }
  }

  return (
    voices.find((voice) => voice.lang?.toLowerCase().startsWith("en")) ||
    voices[0] ||
    null
  );
}

export default function SignInterpreterPanel({
  userName,
  onInterpreterStateChange,
  onCommittedCaption,
}) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const intervalRef = useRef(null);
  const requestInFlightRef = useRef(false);
  const sessionIdRef = useRef("");
  const runningRef = useRef(false);
  const onCommittedCaptionRef = useRef(onCommittedCaption);

  const [isRunning, setIsRunning] = useState(false);
  const [status, setStatus] = useState("Ready to start");
  const [error, setError] = useState("");
  const [transcript, setTranscript] = useState("");
  const [currentLabel, setCurrentLabel] = useState("");
  const [candidateLabel, setCandidateLabel] = useState("");
  const [lastCommittedLabel, setLastCommittedLabel] = useState("");
  const [confidence, setConfidence] = useState(0);
  const [topPredictions, setTopPredictions] = useState([]);
  const [hasHand, setHasHand] = useState(false);
  const [ready, setReady] = useState(false);
  const [framesBuffered, setFramesBuffered] = useState(0);
  const [detectionRatio, setDetectionRatio] = useState(0);
  const [speechSupported] = useState(
    () => typeof window !== "undefined" && "speechSynthesis" in window && "SpeechSynthesisUtterance" in window
  );
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [preferredVoice, setPreferredVoice] = useState(() => {
    if (typeof window === "undefined" || !("speechSynthesis" in window)) {
      return null;
    }

    return pickPreferredVoice(window.speechSynthesis.getVoices());
  });

  function resetSignalState() {
    setTranscript("");
    setCurrentLabel("");
    setCandidateLabel("");
    setLastCommittedLabel("");
    setConfidence(0);
    setTopPredictions([]);
    setHasHand(false);
    setReady(false);
    setFramesBuffered(0);
    setDetectionRatio(0);
  }

  function resetLiveSignalState() {
    setCurrentLabel("");
    setCandidateLabel("");
    setConfidence(0);
    setTopPredictions([]);
    setHasHand(false);
    setReady(false);
    setFramesBuffered(0);
    setDetectionRatio(0);
  }

  function stopSpeech() {
    if (typeof window === "undefined" || !("speechSynthesis" in window)) {
      return;
    }

    window.speechSynthesis.cancel();
    setIsSpeaking(false);
  }

  function releaseResources() {
    runningRef.current = false;

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
    onCommittedCaptionRef.current = onCommittedCaption;
  }, [onCommittedCaption]);

  useEffect(() => {
    if (typeof window === "undefined" || !speechSupported) {
      return undefined;
    }

    const synth = window.speechSynthesis;
    const syncVoices = () => {
      setPreferredVoice(pickPreferredVoice(synth.getVoices()));
    };

    if (typeof synth.addEventListener === "function") {
      synth.addEventListener("voiceschanged", syncVoices);
    } else {
      synth.onvoiceschanged = syncVoices;
    }

    return () => {
      stopSpeech();
      releaseResources();
      if (typeof synth.removeEventListener === "function") {
        synth.removeEventListener("voiceschanged", syncVoices);
      } else if (synth.onvoiceschanged === syncVoices) {
        synth.onvoiceschanged = null;
      }
    };
  }, [speechSupported]);

  useEffect(() => {
    onInterpreterStateChange?.({
      isRunning,
      status,
      transcript,
      currentLabel,
      candidateLabel,
      lastCommittedLabel,
      confidence,
      hasHand,
      ready,
      framesBuffered,
      detectionRatio,
    });
  }, [
    candidateLabel,
    confidence,
    currentLabel,
    detectionRatio,
    framesBuffered,
    hasHand,
    isRunning,
    lastCommittedLabel,
    onInterpreterStateChange,
    ready,
    status,
    transcript,
  ]);

  async function checkBackend() {
    const response = await fetch(`${API_BASE_URL}/api/health`);
    if (!response.ok) {
      throw new Error("Backend health check failed.");
    }
  }

  async function resetRemoteSession(sessionId) {
    if (!sessionId) {
      return;
    }

    try {
      await fetch(`${API_BASE_URL}/api/reset-session`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ sessionId }),
      });
    } catch {
      // Ignore cleanup failures and keep the local UI responsive.
    }
  }

  async function startInterpreter() {
    resetSignalState();
    setError("");
    setStatus("Checking Python backend...");

    try {
      await checkBackend();
    } catch {
      setError("Start the Python API first with python sign_language_api.py");
      setStatus("Backend offline");
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: "user",
          width: { ideal: 960 },
          height: { ideal: 720 },
        },
        audio: false,
      });

      streamRef.current = stream;
      sessionIdRef.current = "";
      runningRef.current = true;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }

      setIsRunning(true);
      setStatus("Interpreter live");
      intervalRef.current = window.setInterval(captureFrame, CAPTURE_INTERVAL_MS);
      captureFrame();
    } catch {
      runningRef.current = false;
      setError("Unable to open a local camera stream for interpretation.");
      setStatus("Camera unavailable");
    }
  }

  async function stopInterpreter() {
    const sessionId = sessionIdRef.current;
    releaseResources();
    stopSpeech();
    resetLiveSignalState();
    setIsRunning(false);
    setStatus("Interpreter stopped");
    await resetRemoteSession(sessionId);
  }

  async function clearTranscript() {
    stopSpeech();
    resetSignalState();
    setError("");
    setStatus(isRunning ? "Transcript cleared" : "Ready to start");
    await resetRemoteSession(sessionIdRef.current);
  }

  function speakTranscript() {
    if (!speechSupported) {
      setError("This browser does not support speech synthesis.");
      return;
    }

    if (isSpeaking) {
      stopSpeech();
      return;
    }

    const textToSpeak = (transcript || lastCommittedLabel || currentLabel || candidateLabel).trim();
    if (!textToSpeak) {
      setError("Add a sign to the transcript before using speech.");
      return;
    }

    const voice = preferredVoice || pickPreferredVoice(window.speechSynthesis.getVoices());
    const utterance = new SpeechSynthesisUtterance(textToSpeak);

    if (voice) {
      utterance.voice = voice;
      utterance.lang = voice.lang || "en-US";
    } else {
      utterance.lang = "en-US";
    }

    utterance.rate = voice?.name.toLowerCase().includes("natural") ? 1 : 0.94;
    utterance.pitch = 1.04;
    utterance.volume = 1;
    utterance.onend = () => setIsSpeaking(false);
    utterance.onerror = () => {
      setIsSpeaking(false);
      setError("Unable to speak the transcript in this browser.");
    };

    setError("");
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(utterance);
    setIsSpeaking(true);
  }

  async function captureFrame() {
    if (
      requestInFlightRef.current ||
      !runningRef.current ||
      !videoRef.current ||
      !canvasRef.current ||
      videoRef.current.readyState < 2
    ) {
      return;
    }

    const canvas = canvasRef.current;
    const context = canvas.getContext("2d");
    if (!context) {
      return;
    }

    requestInFlightRef.current = true;

    try {
      canvas.width = 360;
      canvas.height = 270;
      context.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);

      const response = await fetch(`${API_BASE_URL}/api/predict-frame`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          image: canvas.toDataURL("image/jpeg", 0.84),
          sessionId: sessionIdRef.current || undefined,
          minConfidence: API_MIN_CONFIDENCE,
        }),
      });

      if (!response.ok) {
        throw new Error("Prediction request failed.");
      }

      const data = await response.json();
      if (!runningRef.current) {
        return;
      }

      sessionIdRef.current = data.sessionId;
      setTranscript(data.transcript || "");
      setCurrentLabel(data.label || "");
      setCandidateLabel(data.candidateLabel || "");
      setLastCommittedLabel(data.committedLabel || "");
      setConfidence(data.confidence || 0);
      setTopPredictions(data.topPredictions || []);
      setHasHand(Boolean(data.hasHand));
      setReady(Boolean(data.ready));
      setFramesBuffered(data.framesBuffered || 0);
      setDetectionRatio(data.detectionRatio || 0);

      if (data.updated && data.committedLabel) {
        onCommittedCaptionRef.current?.(
          data.transcript || "",
          data.committedLabel,
          data.confidence ?? 0,
        );
      }

      if (!data.hasHand) {
        setStatus("Show your signing hand in the frame");
      } else if (!data.ready) {
        setStatus(`Buffering motion ${data.framesBuffered}/${MODEL_SEQUENCE_LENGTH}`);
      } else if (data.updated && data.committedLabel) {
        setStatus(`Added "${data.committedLabel}" to transcript`);
      } else if (data.label) {
        setStatus(`Tracking ${data.label}`);
      } else if (data.candidateLabel) {
        setStatus(`Hold ${data.candidateLabel} steady`);
      } else {
        setStatus("Hand detected");
      }
    } catch {
      if (runningRef.current) {
        setError("Interpreter request failed. Check whether the Python API is still running.");
        setStatus("Connection issue");
      }
    } finally {
      requestInFlightRef.current = false;
    }
  }

  const activeLabel = currentLabel || candidateLabel || (hasHand ? "Reading..." : "Waiting...");
  const transcriptFallback = isRunning
    ? "Hold one sign steady for a moment to build the transcript."
    : "Recognized signs stay here after stopping, until you clear them.";
  const canSpeak = speechSupported && (transcript || lastCommittedLabel || currentLabel || candidateLabel);

  return (
    <aside className="interpreter-panel">
      <div className="panel-header">
        <div>
          <p className="panel-eyebrow">SignBridge Interpreter</p>
          <h2>{userName ? `${userName}'s live captions` : "Live captions"}</h2>
        </div>
        <span className={`status-pill ${isRunning ? "active" : ""}`}>{status}</span>
      </div>

      <div className="preview-shell">
        <video ref={videoRef} autoPlay muted playsInline />
        {!isRunning && <div className="preview-placeholder">Camera preview appears here</div>}
        <div className="preview-overlay">
          <div className="preview-badge-row">
            <span className="preview-badge">Local camera</span>
            <span className={`preview-badge ${ready ? "accent" : ""}`}>
              {ready ? "Sequence ready" : "Warming up"}
            </span>
          </div>
          <div className="preview-footer">
            <span>Buffer {framesBuffered}/{MODEL_SEQUENCE_LENGTH}</span>
            <span>Tracked {Math.round(detectionRatio * 100)}%</span>
          </div>
        </div>
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
        <div className="signal-card">
          <span className="signal-label">Buffer</span>
          <strong>{framesBuffered}/{MODEL_SEQUENCE_LENGTH}</strong>
        </div>
        <div className="signal-card">
          <span className="signal-label">Hand tracking</span>
          <strong>{Math.round(detectionRatio * 100)}%</strong>
        </div>
      </div>

      <div className="transcript-block">
        <div className="transcript-header">
          <span className="signal-label">Transcript</span>
          {lastCommittedLabel && <span className="transcript-tag">Latest: {lastCommittedLabel}</span>}
        </div>
        <p className="transcript-value">{transcript || transcriptFallback}</p>
      </div>

      <div className="prediction-list">
        {topPredictions.length ? (
          topPredictions.map((item) => {
            const width = `${Math.max(8, Math.round(item.confidence * 100))}%`;
            return (
              <div key={item.label} className="prediction-item">
                <div className="prediction-copy">
                  <span>{item.label}</span>
                  <strong>{Math.round(item.confidence * 100)}%</strong>
                </div>
                <div className="prediction-track">
                  <span style={{ width }} />
                </div>
              </div>
            );
          })
        ) : (
          <p className="prediction-empty">Top matches will appear here while the model is reading your signs.</p>
        )}
      </div>

      <div className="panel-actions">
        {!isRunning ? (
          <button type="button" className="primary-action" onClick={startInterpreter}>
            Start Captions
          </button>
        ) : (
          <button type="button" className="primary-action danger" onClick={stopInterpreter}>
            Stop Captions
          </button>
        )}

        <button
          type="button"
          className="secondary-action"
          onClick={speakTranscript}
          disabled={!canSpeak && !isSpeaking}
        >
          {isSpeaking ? "Stop Voice" : "Speak Transcript"}
        </button>

        <button type="button" className="ghost-action" onClick={clearTranscript}>
          Clear
        </button>
      </div>

      {error && <p className="panel-error">{error}</p>}
      <p className="panel-note">
        Runs on your device, speaks with your browser voice, and talks to the Python API at {API_BASE_URL}.
      </p>
    </aside>
  );
}
