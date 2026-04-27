import { useState } from "react";
import { useNavigate } from "react-router-dom";
import "./Home.css";

export default function Home() {
  const [roomId, setRoomId] = useState("");
  const [name, setName] = useState("");
  const navigate = useNavigate();

  const joinRoom = () => {
    if (!roomId.trim()) return alert("Enter Room ID");
    if (!name.trim()) return alert("Enter your name");

    navigate(`/room/${encodeURIComponent(roomId.trim())}?name=${encodeURIComponent(name.trim())}`);
  };

  return (
    <div className="home">
      <div className="overlay" />

      <div className="card">
        <h1 className="title">Sign-Bridge</h1>
        <p className="subtitle">
          Bridging communication through vision and voice.
        </p>

        <div className="input-group">
          <input
            type="text"
            placeholder="Enter Your Name"
            value={name}
            onChange={(event) => setName(event.target.value)}
          />

          <input
            type="text"
            placeholder="Enter Room ID"
            value={roomId}
            onChange={(event) => setRoomId(event.target.value)}
          />

          <button onClick={joinRoom}>Join Call</button>
        </div>

        <p className="hint">
          Tip: Share your Room ID with others to connect
        </p>
      </div>
    </div>
  );
}