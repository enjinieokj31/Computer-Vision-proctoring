import { useEffect, useRef, useState } from "react";
import axios from "axios";
import "./App.css";

const API_URL =  import.meta.env.VITE_API_URL || "http://127.0.0.1:0000/predict";

export default function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const hiddenCanvasRef = useRef(null);
  const timerRef = useRef(null);
  const runningRef = useRef(false);
  const requestInFlightRef = useRef(false);

  const [cameraReady, setCameraReady] = useState(false);
  const [running, setRunning] = useState(false);
  const [latency, setLatency] = useState(null);
  const [faceCount, setFaceCount] = useState(0);
  const [alertOn, setAlertOn] = useState(false);
  const [error, setError] = useState("");

  // -----------------------------
  // Start camera once on mount
  // -----------------------------
  useEffect(() => {
    let mounted = true;

    const initCamera = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            width: 640,
            height: 480,
            facingMode: "user",
          },
          audio: false,
        });

        if (!mounted) return;

        if (videoRef.current) {
          videoRef.current.srcObject = stream;

          videoRef.current.onloadedmetadata = () => {
            if (!mounted) return;
            setCameraReady(true);
          };
        }
      } catch (err) {
        if (mounted) {
          setError("Camera access denied or unavailable");
        }
      }
    };

    initCamera();

    return () => {
      mounted = false;
      stopDetection();
      stopCamera();
    };
  }, []);

  // -----------------------------
  // Camera stop
  // -----------------------------
  const stopCamera = () => {
    const stream = videoRef.current?.srcObject;

    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
    }
  };

  // -----------------------------
  // Start detection loop
  // -----------------------------
  const startDetection = () => {
    if (!cameraReady || runningRef.current) return;

    runningRef.current = true;
    setRunning(true);
    setError("");

    detectionLoop();
  };

  // -----------------------------
  // Stop detection loop
  // -----------------------------
  const stopDetection = () => {
    runningRef.current = false;
    setRunning(false);

    if (timerRef.current) {
      clearTimeout(timerRef.current);
      timerRef.current = null;
    }

    requestInFlightRef.current = false;
  };

  // -----------------------------
  // No-overlap loop
  // -----------------------------
  const detectionLoop = async () => {
    if (!runningRef.current) return;

    await captureAndSend();

    if (runningRef.current) {
      timerRef.current = setTimeout(detectionLoop, 500);
    }
  };

  // -----------------------------
  // Capture frame and call backend
  // -----------------------------
  const captureAndSend = async () => {
    if (requestInFlightRef.current) return;

    const video = videoRef.current;
    const hiddenCanvas = hiddenCanvasRef.current;

    if (!video || !hiddenCanvas) return;
    if (video.videoWidth === 0 || video.videoHeight === 0) return;

    requestInFlightRef.current = true;

    try {
      hiddenCanvas.width = video.videoWidth;
      hiddenCanvas.height = video.videoHeight;

      const ctx = hiddenCanvas.getContext("2d");
      ctx.drawImage(video, 0, 0, hiddenCanvas.width, hiddenCanvas.height);

      const blob = await new Promise((resolve) =>
        hiddenCanvas.toBlob(resolve, "image/jpeg", 0.8)
      );

      if (!blob) {
        requestInFlightRef.current = false;
        return;
      }

      const formData = new FormData();
      formData.append("file", blob, "frame.jpg");

      const res = await axios.post(API_URL, formData);

      console.log("FULL RESPONSE:", res.data);

      const detections = res.data.detections || [];
      const count =
        res.data.face_count !== undefined
          ? res.data.face_count
          : detections.length;

      setLatency(res.data.latency_ms || null);
      setFaceCount(count);
      setAlertOn(count > 1);

      drawBoxes(detections);
    } catch (err) {
      console.error(err);
      setError("Prediction request failed");
    } finally {
      requestInFlightRef.current = false;
    }
  };

  // -----------------------------
  // Draw boxes (optional)
  // -----------------------------
  const drawBoxes = (boxes) => {
    const video = videoRef.current;
    const canvas = canvasRef.current;

    if (!video || !canvas) return;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const ctx = canvas.getContext("2d");

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    ctx.lineWidth = 3;
    ctx.strokeStyle = "#00ff66";
    ctx.fillStyle = "#00ff66";
    ctx.font = "16px Arial";

    boxes.forEach((box) => {
      const x = box.x1;
      const y = box.y1;
      const w = box.x2 - box.x1;
      const h = box.y2 - box.y1;

      ctx.strokeRect(x, y, w, h);
      ctx.fillText(
        `${(box.conf || 0).toFixed(2)}`,
        x,
        y > 10 ? y - 5 : 15
      );
    });
  };

  return (
    <div className="container">
      <h1>Live Face Monitoring</h1>

      <div className="controls">
        {!running ? (
          <button onClick={startDetection} disabled={!cameraReady}>
            Start Detection
          </button>
        ) : (
          <button onClick={stopDetection}>Stop Detection</button>
        )}
      </div>

      <div className={`status-card ${alertOn ? "danger" : "safe"}`}>
        {faceCount === 0 && "No Face Detected"}
        {faceCount === 1 && "One Face Detected"}
        {faceCount > 1 && `ALERT: ${faceCount} Faces Detected`}
      </div>

      {latency !== null && <p>Latency: {latency} ms</p>}
      {error && <p className="error">{error}</p>}

      <div className="video-wrapper">
        <video ref={videoRef} autoPlay playsInline muted />
        <canvas ref={canvasRef} className="overlay" />
      </div>

      <canvas ref={hiddenCanvasRef} style={{ display: "none" }} />
    </div>
  );
}