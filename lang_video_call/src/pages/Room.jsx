import { useLocation, useParams } from "react-router-dom";
import SignInterpreterPanel from "../components/SignInterpreterPanel";
import VideoCall from "../components/VideoCall";
import "./Room.css";

export default function Room() {
  const { id } = useParams();
  const query = new URLSearchParams(useLocation().search);
  const name = query.get("name") || "Guest";

  return (
    <div className="room-shell">
      <VideoCall roomID={id} userName={name} />
      <SignInterpreterPanel userName={name} />
    </div>
  );
}