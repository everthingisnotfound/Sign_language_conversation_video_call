import { forwardRef, useEffect, useImperativeHandle, useRef } from "react";
import { ZegoUIKitPrebuilt } from "@zegocloud/zego-uikit-prebuilt";

const VideoCall = forwardRef(({ roomID, userName }, ref) => {
  const containerRef = useRef(null);
  const zegoRef = useRef(null);

  useImperativeHandle(ref, () => ({
    sendTranscript: (text, label, confidence) => {
      if (zegoRef.current && zegoRef.current.sendTranscript) {
        zegoRef.current.sendTranscript(text, label, confidence);
      }
    },
  }));

  useEffect(() => {
    const appID = Number(import.meta.env.VITE_ZEGO_APP_ID);
    const serverSecret = import.meta.env.VITE_ZEGO_SERVER_SECRET;
    const containerElement = containerRef.current;

    if (!containerElement || !appID || !serverSecret) {
      console.error("Missing ZEGO config");
      return;
    }

    const userID = Date.now().toString();
    const finalUserName = userName || "Guest";

    const kitToken = ZegoUIKitPrebuilt.generateKitTokenForTest(
      appID,
      serverSecret,
      roomID,
      userID,
      finalUserName,
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

    zegoRef.current = zp;

    return () => {
      try {
        zp.destroy();
      } catch (err) {
        console.warn("Zego destroy failed:", err);
        if (containerElement) containerElement.innerHTML = "";
      }
    };
  }, [roomID, userName]);

  return (
    <div className="video-stage">
      <div ref={containerRef} className="video-stage__canvas" />
    </div>
  );
});

export default VideoCall;
