import { forwardRef, useEffect, useImperativeHandle, useRef } from "react";
import { ZegoUIKitPrebuilt } from "@zegocloud/zego-uikit-prebuilt";

const SIGNBRIDGE_CMD = "signbridge.v1";

const VideoCall = forwardRef(({ roomID, userName, onRemoteCaption }, ref) => {
  const containerRef = useRef(null);
  const zegoRef = useRef(null);
  const localUserIdRef = useRef("");
  const onRemoteCaptionRef = useRef(onRemoteCaption);
  const userNameRef = useRef(userName || "Guest");

  useEffect(() => {
    onRemoteCaptionRef.current = onRemoteCaption;
  }, [onRemoteCaption]);

  useEffect(() => {
    userNameRef.current = userName || "Guest";
  }, [userName]);

  useImperativeHandle(ref, () => ({
    sendTranscript: (text, label, confidence) => {
      const zp = zegoRef.current;
      if (!zp?.sendInRoomCommand) {
        return;
      }
      const payload = {
        v: 1,
        type: SIGNBRIDGE_CMD,
        transcript: text ?? "",
        label: label ?? "",
        confidence: confidence ?? 0,
        fromUserName: userNameRef.current || "Guest",
      };
      void zp.sendInRoomCommand(JSON.stringify(payload), []);
    },
  }));

  useEffect(() => {
    const appID = Number(import.meta.env.VITE_ZEGO_APP_ID);
    const serverSecret = import.meta.env.VITE_ZEGO_SERVER_SECRET;
    const containerElement = containerRef.current;

    if (!containerElement || !appID || !serverSecret) {
      console.error("Missing ZEGO config");
      return undefined;
    }

    const userID = Date.now().toString();
    localUserIdRef.current = userID;
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
        mode: ZegoUIKitPrebuilt.VideoConference,
        config: {
          role: ZegoUIKitPrebuilt.Host,
        },
      },
      showPreJoinView: false,
      layout: "Grid",
      turnOnCameraWhenJoining: true,
      turnOnMicrophoneWhenJoining: true,
      onInRoomCommandReceived: (fromUser, command) => {
        if (!fromUser || fromUser.userID === localUserIdRef.current) {
          return;
        }
        try {
          const payload = JSON.parse(command);
          if (payload?.type !== SIGNBRIDGE_CMD || payload.transcript == null) {
            return;
          }
          onRemoteCaptionRef.current?.({
            fromUserId: fromUser.userID,
            fromUserName: fromUser.userName || payload.fromUserName || "Guest",
            transcript: String(payload.transcript),
            label: payload.label != null ? String(payload.label) : "",
            confidence: typeof payload.confidence === "number" ? payload.confidence : 0,
            at: Date.now(),
          });
        } catch {
          /* ignore malformed commands */
        }
      },
    });

    zegoRef.current = zp;

    return () => {
      try {
        zp.destroy();
      } catch (err) {
        console.warn("Zego destroy failed:", err);
        if (containerElement) containerElement.innerHTML = "";
      }
      zegoRef.current = null;
    };
  }, [roomID, userName]);

  return (
    <div className="video-stage">
      <div ref={containerRef} className="video-stage__canvas" />
    </div>
  );
});

VideoCall.displayName = "VideoCall";

export default VideoCall;
