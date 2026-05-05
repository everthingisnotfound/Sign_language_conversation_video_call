import { useCallback, useRef, useState } from "react";
import { useLocation, useParams } from "react-router-dom";
import SignInterpreterPanel from "../components/SignInterpreterPanel";
import VideoCall from "../components/VideoCall";
import ShareMeeting from "../components/ShareMeeting";
import "./Room.css";

const initialInterpreterState = {
  isRunning: false,
  status: "Ready to start",
  transcript: "",
  currentLabel: "",
  candidateLabel: "",
  lastCommittedLabel: "",
  confidence: 0,
  hasHand: false,
  ready: false,
  framesBuffered: 0,
  detectionRatio: 0,
};

export default function Room() {
  const { id } = useParams();
  const query = new URLSearchParams(useLocation().search);
  const name = query.get("name") || "Guest";

  const videoCallRef = useRef(null);
  const [interpreterState, setInterpreterState] = useState(
    initialInterpreterState,
  );
  const [remoteCaption, setRemoteCaption] = useState(null);

  const roomId = decodeURIComponent(id || "");

  const handleRemoteCaption = useCallback((payload) => {
    setRemoteCaption(payload);
  }, []);

  const handleCommittedCaption = useCallback((fullTranscript, word, confidence) => {
    videoCallRef.current?.sendTranscript(fullTranscript, word, confidence);
  }, []);

  const activeLabel =
    interpreterState.currentLabel ||
    interpreterState.candidateLabel ||
    interpreterState.lastCommittedLabel ||
    "Waiting for sign";

  const transcriptText =
    interpreterState.transcript ||
    (interpreterState.isRunning
      ? "Hold one sign clearly for a moment and the live transcript will appear here."
      : "Start the interpreter to show live sign-to-text captions during the call.");

  return (
    <div className="room-shell">
      <div className="room-backdrop" />

      <VideoCall
        ref={videoCallRef}
        roomID={roomId}
        userName={name}
        onRemoteCaption={handleRemoteCaption}
      />

      <div className="room-hud">
        <div className="room-brand">
          <p className="room-kicker">SignBridge Live</p>
          <h1>Room {roomId}</h1>

          <div className="room-meta">
            <span>{name}</span>
            <span>
              {interpreterState.isRunning
                ? "Interpreter connected"
                : "Interpreter idle"}
            </span>
          </div>
        </div>

        <div className="room-stat-strip">
          <div className="room-stat-card">
            <span className="room-stat-label">Current sign</span>
            <strong>{activeLabel}</strong>
          </div>

          <div className="room-stat-card">
            <span className="room-stat-label">Confidence</span>
            <strong>
              {interpreterState.hasHand
                ? `${Math.round(interpreterState.confidence * 100)}%`
                : "--"}
            </strong>
          </div>
        </div>

        <ShareMeeting roomId={roomId} />
      </div>

      <SignInterpreterPanel
        userName={name}
        onInterpreterStateChange={setInterpreterState}
        onCommittedCaption={handleCommittedCaption}
      />

      <div
        className={`caption-dock ${
          interpreterState.isRunning || remoteCaption ? "active" : ""
        }`}
      >
        <div className="caption-dock__meta">
          <span
            className={`caption-pill ${
              interpreterState.hasHand ? "active" : ""
            }`}
          >
            {interpreterState.hasHand ? "Hand tracked" : "Waiting for hand"}
          </span>

          <span>{interpreterState.status}</span>
          <span>{activeLabel}</span>
        </div>

        {remoteCaption ? (
          <div className="caption-dock__remote">
            <span className="caption-dock__remote-eyebrow">
              Live from {remoteCaption.fromUserName}
            </span>
            <p className="caption-dock__text caption-dock__text--remote">
              {remoteCaption.transcript}
            </p>
          </div>
        ) : null}

        <p className="caption-dock__text">{transcriptText}</p>
      </div>
    </div>
  );
}
