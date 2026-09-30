/* No StrictMode: react-leaflet v4 double-mounts in dev StrictMode and logs console errors. */
import ReactDOM from "react-dom/client";
import App from "./App";
import "./index.css";

ReactDOM.createRoot(document.getElementById("root")!).render(<App />);