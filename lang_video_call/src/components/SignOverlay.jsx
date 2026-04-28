export default function SignOverlay({ text = "" }) {
  if (!text) return null;

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
        zIndex: 10,
        pointerEvents: "none",
      }}
    >
      ✋ {text}
    </div>
  );
}
