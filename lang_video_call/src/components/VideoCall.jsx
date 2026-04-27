import { useEffect, useRef } from "react";
import { ZegoUIKitPrebuilt } from "@zegocloud/zego-uikit-prebuilt";

export default function VideoCall({ roomID, userName }) {
  const containerRef = useRef(null);

  useEffect(() => {
    const appID = Number(import.meta.env.VITE_ZEGO_APP_ID);
    const serverSecret = import.meta.env.VITE_ZEGO_SERVER_SECRET;
    const containerElement = containerRef.current;

    if (!containerElement || !appID || !serverSecret) {
      return undefined;
    }

    const userID = Date.now().toString();
    const finalUserName = userName || "Guest";

    const kitToken = ZegoUIKitPrebuilt.generateKitTokenForTest(
      appID,
      serverSecret,
      roomID,
      userID,
      finalUserName
    );

    const zp = ZegoUIKitPrebuilt.create(kitToken);

    zp.joinRoom({
      container: containerElement,
      scenario: {
        mode: ZegoUIKitPrebuilt.OneONoneCall,
      },
      showPreJoinView: false,
      layout: "Grid",
      turnOnCameraWhenJoining: true,
      turnOnMicrophoneWhenJoining: true,
    });

    return () => {
      try {
        zp.destroy();
      } catch {
        containerElement.innerHTML = "";
      }
    };
  }, [roomID, userName]);

  return (
    <div
      ref={containerRef}
      style={{
        width: "100%",
        height: "100%",
        overflow: "hidden",
        position: "absolute",
        inset: 0,
        background: "#071018",
      }}
    />
  );
}