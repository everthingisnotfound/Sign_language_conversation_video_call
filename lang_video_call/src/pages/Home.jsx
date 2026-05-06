import { useEffect, useMemo, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import ParticleBackground from "../components/ParticleBackground";
import "./Home.css";

const highlights = [
  {
    title: "Live sign captions",
    description:
      "Run word-level interpretation beside your call and watch the transcript build in real time.",
  },
  {
    title: "Private local pipeline",
    description:
      "Your browser preview talks to your own Python API, so you stay in control while testing.",
  },
  {
    title: "Built for accessibility demos",
    description:
      "A clearer UI, larger captions, and a calmer call layout help the experience feel presentation-ready.",
  },
];

export default function Home() {
  const [roomId, setRoomId] = useState("");
  const [name, setName] = useState("");
  const navigate = useNavigate();
  const location = useLocation();

  const sharedRoomId = useMemo(() => {
    const params = new URLSearchParams(location.search);
    return (params.get("room") || "").trim();
  }, [location.search]);

  useEffect(() => {
    if (sharedRoomId) {
      setRoomId(sharedRoomId);
    }
  }, [sharedRoomId]);

  const joinRoom = () => {
    if (!roomId.trim()) return alert("Enter Room ID");
    if (!name.trim()) return alert("Enter your name");

    navigate(
      `/room/${encodeURIComponent(roomId.trim())}?name=${encodeURIComponent(name.trim())}`,
    );
  };

  return (
    <div className="home">
      <ParticleBackground />
      <div className="home-ambient home-ambient--one" />
      <div className="home-ambient home-ambient--two" />

      <section className="home-copy">
        <p className="hero-kicker">Accessible Communication Platform</p>
        <h1 className="title">SignBridge</h1>
        <p className="subtitle">
          Pair your video calls with live sign-to-text captions in a more
          polished, presentation-ready interface.
        </p>

        <div className="hero-stats">
          <div className="hero-stat">
            <strong>46</strong>
            <span>Word gestures trained</span>
          </div>
          <div className="hero-stat">
            <strong>Live</strong>
            <span>Python API + React call view</span>
          </div>
        </div>

        <div className="feature-grid">
          {highlights.map((item) => (
            <article key={item.title} className="feature-card">
              <h2>{item.title}</h2>
              <p>{item.description}</p>
            </article>
          ))}
        </div>
      </section>

      <section className="join-card">
        <div className="join-card__header">
          <p className="hero-kicker">Join a room</p>
          <h2>Start your accessible video call</h2>
          <p>
            Enter your display name and room code to open the call and
            interpreter workspace.
          </p>
        </div>

        <div className="input-group">
          <label>
            <span>Your name</span>
            <input
              type="text"
              placeholder="Aarav"
              value={name}
              onChange={(event) => setName(event.target.value)}
            />
          </label>

          <label>
            <span>Room ID</span>
            <input
              type="text"
              placeholder="team-sync-01"
              value={roomId}
              onChange={(event) => setRoomId(event.target.value)}
              readOnly={Boolean(sharedRoomId)}
            />
          </label>

          <button type="button" onClick={joinRoom}>
            Launch Call Workspace
          </button>
        </div>

        <p className="hint">
          Tip: run your Python API before opening the interpreter inside the
          room.
        </p>
      </section>
    </div>
  );
}
