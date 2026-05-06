import { forwardRef, useEffect, useImperativeHandle, useRef } from "react";
import { ZegoUIKitPrebuilt } from "@zegocloud/zego-uikit-prebuilt";

const SIGNBRIDGE_CMD = "signbridge.v1";

const VideoCall = forwardRef(({ roomID, userName, onRemoteCaption }, ref) => {
  const containerRef = useRef(null);
  const zegoRef = useRef(null);
  const localUserIdRef = useRef("");
  const onRemoteCaptionRef = useRef(onRemoteCaption);
  const userNameRef = useRef(userName || "Guest");
  const sendSeqRef = useRef(0);

  useEffect(() => {
    onRemoteCaptionRef.current = onRemoteCaption;
  }, [onRemoteCaption]);

  useEffect(() => {
    userNameRef.current = userName || "Guest";
  }, [userName]);

  useImperativeHandle(ref, () => ({
    sendTranscript: (text, label, confidence) => {
      const zp = zegoRef.current;
      if (!zp) {
        return;
      }
      // Keep payload tiny so it always fits signaling limits.
      const clippedTranscript = String(text ?? "").slice(-180);
      const token = String(label ?? "").trim();
      const payload = {
        v: 1,
        type: SIGNBRIDGE_CMD,
        transcript: clippedTranscript,
        token,
        label: token,
        confidence: confidence ?? 0,
        fromUserName: userNameRef.current || "Guest",
        seq: (sendSeqRef.current += 1),
      };
      // Prefer custom command broadcast (more reliable than sendInRoomCommand([], ...)).
      if (typeof zp.sendInRoomCustomCommand === "function") {
        void zp.sendInRoomCustomCommand(payload);
        return;
      }
      if (typeof zp.sendInRoomCommand === "function") {
        // Some SDK builds treat [] as "send to nobody"; try undefined broadcast.
        void zp.sendInRoomCommand(JSON.stringify(payload), undefined);
      }
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

    const emitRemoteCaption = (fromUser, payload) => {
      if (!fromUser || fromUser.userID === localUserIdRef.current) {
        return;
      }
      if (payload?.type !== SIGNBRIDGE_CMD) {
        return;
      }
      const transcript =
        payload.transcript != null
          ? String(payload.transcript)
          : payload.token != null
            ? String(payload.token)
            : payload.label != null
              ? String(payload.label)
              : "";
      if (!transcript) return;
      onRemoteCaptionRef.current?.({
        fromUserId: fromUser.userID,
        fromUserName: fromUser.userName || payload.fromUserName || "Guest",
        transcript,
        label: payload.label != null ? String(payload.label) : "",
        confidence: typeof payload.confidence === "number" ? payload.confidence : 0,
        at: Date.now(),
      });
    };

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
      onInRoomCustomCommandReceived: (commands) => {
        if (!Array.isArray(commands) || !commands.length) return;
        for (const cmd of commands) {
          // Zego's custom command structure varies slightly across versions.
          const payload = cmd?.command;
          const fromUser = cmd?.fromUser || {
            userID: cmd?.senderUserID,
            userName: payload?.fromUserName,
          };
          emitRemoteCaption(fromUser, payload);
        }
      },
      onInRoomCommandReceived: (fromUser, command) => {
        try {
          const payload = JSON.parse(command);
          emitRemoteCaption(fromUser, payload);
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
