import { useEffect, useState } from "react";

export default function SignOverlay() {
  const [text, setText] = useState("...");

  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        const res = await fetch("http://localhost:5000/predict");
        const data = await res.json();
        setText(data.text);
      } catch {
        console.log("API not running");
      }
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  return (
    <div
      style={{
        position: "absolute",
        bottom: "120px",
        left: "50%",
        transform: "translateX(-50%)",
        background: "rgba(0,0,0,0.7)",
        padding: "12px 20px",
        borderRadius: "10px",
        color: "white",
        fontSize: "18px",
        fontWeight: "600",
      }}
    >
      ✋ {text}
    </div>
  );
}
